#!/usr/bin/env python3
"""
POLARIS v4 — Non-LLM Neural Network Policy (MLP + PPO)
========================================================
Addresses: "Analyze whether a non-LLM policy, typically just a neural network
with a parametric action space, does to understand what behaviour your reward
function motivates."

Trains a small MLP policy with PPO on the POLARIS environment.
If this learns, the reward is well-designed. Then LLM failure = coordination problem.
If this fails too, the reward needs tuning.

No LLM. No transformers. Pure RL. Runs on CPU in minutes.
"""
import sys, os, io, json, time, random, math, statistics, copy
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from server.policy_environment import PolicyEnvironment
from server.config import TASK_CONFIGS, CORE_ACTIONS, STATE_BOUNDS, VALID_ACTIONS
from server.tasks import grade_trajectory

OUT = "outputs/nn_baseline"
os.makedirs(OUT, exist_ok=True)

# State keys we extract (21 dims)
STATE_KEYS = list(STATE_BOUNDS.keys())
N_STATE = len(STATE_KEYS)  # 21
N_ACTIONS = len(CORE_ACTIONS)  # 16

# PPO hyperparams
HIDDEN = 128
LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEFF = 0.01
VALUE_COEFF = 0.5
PPO_EPOCHS = 4
BATCH_SIZE = 64
TOTAL_EPISODES = 500
EVAL_EVERY = 50
MAX_STEPS = 100

TASKS = ["environmental_recovery", "negotiation_arena"]
EVAL_SEEDS = [42, 123, 777]


class ActorCritic(nn.Module):
    """Simple MLP policy + value network."""
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(N_STATE, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
        )
        self.policy = nn.Linear(HIDDEN, N_ACTIONS)
        self.value = nn.Linear(HIDDEN, 1)
    
    def forward(self, x):
        h = self.shared(x)
        return self.policy(h), self.value(h).squeeze(-1)
    
    def act(self, state_vec):
        with torch.no_grad():
            logits, value = self.forward(state_vec.unsqueeze(0))
            dist = Categorical(logits=logits)
            action = dist.sample()
            return action.item(), dist.log_prob(action).item(), value.item()


def state_to_vec(metadata):
    """Convert environment metadata to normalized state vector."""
    vec = []
    for key in STATE_KEYS:
        val = metadata.get(key, 0.0)
        lo, hi = STATE_BOUNDS[key]
        norm = (val - lo) / (hi - lo) if hi > lo else 0.0
        vec.append(max(0.0, min(1.0, norm)))
    return torch.tensor(vec, dtype=torch.float32)


def collect_episode(model, task_id, seed):
    """Run one episode, collect transitions."""
    env = PolicyEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    
    states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
    action_names = []
    step = 0
    
    while not obs.done and step < MAX_STEPS:
        step += 1
        state_vec = state_to_vec(obs.metadata)
        action_idx, log_prob, value = model.act(state_vec)
        action_name = CORE_ACTIONS[action_idx]
        
        action_data = {"action": action_name, "coalition_target": [],
                       "veto_prediction": [], "stance": "cooperative"}
        obs = env.step(action_data)
        
        states.append(state_vec)
        actions.append(action_idx)
        log_probs.append(log_prob)
        rewards.append(obs.reward)
        values.append(value)
        dones.append(obs.done)
        action_names.append(action_name)
    
    score = grade_trajectory(task_id, env.get_trajectory())
    collapsed = obs.metadata.get("collapsed", step < MAX_STEPS)
    
    return {
        "states": torch.stack(states),
        "actions": torch.tensor(actions),
        "log_probs": torch.tensor(log_probs),
        "rewards": rewards,
        "values": values,
        "dones": dones,
        "score": score,
        "collapsed": collapsed,
        "steps": step,
        "unique_actions": len(set(action_names)),
        "action_names": action_names,
    }


def compute_gae(rewards, values, dones):
    """Compute generalized advantage estimation."""
    advantages = []
    returns = []
    gae = 0.0
    next_value = 0.0
    
    for t in reversed(range(len(rewards))):
        mask = 0.0 if dones[t] else 1.0
        delta = rewards[t] + GAMMA * next_value * mask - values[t]
        gae = delta + GAMMA * GAE_LAMBDA * mask * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[t])
        next_value = values[t]
    
    return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)


def ppo_update(model, optimizer, states, actions, old_log_probs, advantages, returns):
    """PPO clipped objective update."""
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    total_loss_val = 0.0
    for _ in range(PPO_EPOCHS):
        logits, values = model(states)
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        value_loss = F.mse_loss(values, returns)
        
        loss = policy_loss + VALUE_COEFF * value_loss - ENTROPY_COEFF * entropy
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        total_loss_val += loss.item()
    
    return total_loss_val / PPO_EPOCHS


def evaluate(model, task_id, seeds):
    scores, collapses, uniques = [], [], []
    for s in seeds:
        ep = collect_episode(model, task_id, s)
        scores.append(ep["score"])
        collapses.append(1 if ep["collapsed"] else 0)
        uniques.append(ep["unique_actions"])
    return {
        "score": round(statistics.mean(scores), 4),
        "std": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0,
        "collapse_rate": round(statistics.mean(collapses), 2),
        "unique_actions": round(statistics.mean(uniques), 1),
    }


def main():
    print("="*70)
    print("  POLARIS v4 — NON-LLM NEURAL NETWORK BASELINE (MLP + PPO)")
    print(f"  State dim: {N_STATE} | Actions: {N_ACTIONS}")
    print(f"  Hidden: {HIDDEN} | LR: {LR} | Episodes: {TOTAL_EPISODES}")
    print(f"  Device: CPU (no GPU needed for MLP)")
    print("="*70)
    
    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    log = {"config": {"state_dim": N_STATE, "action_dim": N_ACTIONS, "hidden": HIDDEN,
                      "lr": LR, "gamma": GAMMA, "episodes": TOTAL_EPISODES},
           "training": [], "evals": []}
    
    # BEFORE eval
    print("\n--- BEFORE TRAINING (random init) ---")
    for tid in TASKS:
        ev = evaluate(model, tid, EVAL_SEEDS)
        print(f"  {tid}: score={ev['score']} collapse={ev['collapse_rate']:.0%} unique={ev['unique_actions']}")
        log["evals"].append({"episode": 0, "task": tid, "phase": "before", **ev})
    
    rng = random.Random(42)
    t0 = time.time()
    
    for ep in range(1, TOTAL_EPISODES + 1):
        # Alternate between tasks (70% negotiation, 30% env recovery)
        tid = "negotiation_arena" if rng.random() < 0.7 else "environmental_recovery"
        seed = rng.randint(100, 10000)
        
        episode = collect_episode(model, tid, seed)
        advantages, returns = compute_gae(episode["rewards"], episode["values"], episode["dones"])
        
        loss = ppo_update(model, optimizer, episode["states"], episode["actions"],
                         episode["log_probs"], advantages, returns)
        
        log["training"].append({
            "episode": ep, "loss": round(loss, 4), "score": round(episode["score"], 4),
            "collapsed": episode["collapsed"], "steps": episode["steps"],
            "unique_actions": episode["unique_actions"], "task": tid,
            "total_reward": round(sum(episode["rewards"]), 4),
        })
        
        if ep % 25 == 0:
            elapsed = time.time() - t0
            print(f"  [{ep}/{TOTAL_EPISODES}] loss={loss:.4f} score={episode['score']:.4f} "
                  f"{'DEAD' if episode['collapsed'] else 'ALIVE'} uniq={episode['unique_actions']} | {elapsed:.0f}s")
        
        if ep % EVAL_EVERY == 0:
            print(f"\n  --- EVAL at episode {ep} ---")
            for etid in TASKS:
                ev = evaluate(model, etid, EVAL_SEEDS)
                print(f"    {etid}: score={ev['score']} collapse={ev['collapse_rate']:.0%} unique={ev['unique_actions']}")
                log["evals"].append({"episode": ep, "task": etid, "phase": "training", **ev})
    
    # AFTER eval
    print(f"\n{'='*70}")
    print("  AFTER TRAINING")
    print(f"{'='*70}")
    for tid in TASKS:
        ev = evaluate(model, tid, EVAL_SEEDS)
        print(f"  {tid}: score={ev['score']} collapse={ev['collapse_rate']:.0%} unique={ev['unique_actions']}")
        log["evals"].append({"episode": TOTAL_EPISODES, "task": tid, "phase": "after", **ev})
    
    # Save model
    torch.save(model.state_dict(), os.path.join(OUT, "mlp_policy.pt"))
    
    # Comparison table
    print(f"\n{'='*80}")
    print(f"  NN vs LLM COMPARISON")
    print(f"{'='*80}")
    
    before = {e["task"]: e for e in log["evals"] if e["phase"] == "before"}
    after = {e["task"]: e for e in log["evals"] if e["phase"] == "after"}
    
    print(f"  {'Agent':<20} | {'Single-Agent':>14} | {'Multi-Agent':>14} | {'CCR':>8}")
    print(f"  {'-'*20}-+-{'-'*14}-+-{'-'*14}-+-{'-'*8}")
    
    # NN before
    s_b = before.get("environmental_recovery", {}).get("score", 0)
    m_b = before.get("negotiation_arena", {}).get("score", 0)
    ccr_b = m_b / s_b if s_b > 0 else 0
    print(f"  {'NN (before)': <20} | {s_b:>14.4f} | {m_b:>14.4f} | {ccr_b:>8.4f}")
    
    # NN after
    s_a = after.get("environmental_recovery", {}).get("score", 0)
    m_a = after.get("negotiation_arena", {}).get("score", 0)
    ccr_a = m_a / s_a if s_a > 0 else 0
    print(f"  {'NN (after PPO)': <20} | {s_a:>14.4f} | {m_a:>14.4f} | {ccr_a:>8.4f}")
    
    # Compare with known LLM results
    llm_path = "outputs/llm_benchmark/zero_shot_complete.json"
    if os.path.exists(llm_path):
        with open(llm_path) as f:
            llm = json.load(f)
        for row in llm.get("zero_shot_results", {}).get("unified_table", {}).get("rows", []):
            name, _, single, multi, _, ccr, typ = row
            print(f"  {name: <20} | {single:>14.4f} | {multi:>14.4f} | {ccr:>8.4f}")
    
    print(f"\n  KEY FINDING:")
    if s_a > 0.5 and m_a < 0.3:
        print(f"  NN policy LEARNS single-agent ({s_a:.4f}) but FAILS multi-agent ({m_a:.4f}).")
        print(f"  This proves: reward is learnable. Coordination collapse is the bottleneck, not reward design.")
    elif s_a > 0.5 and m_a > 0.3:
        print(f"  NN policy succeeds at BOTH tasks. LLM failure is an LLM-specific problem.")
    else:
        print(f"  NN policy struggles ({s_a:.4f}/{m_a:.4f}). Reward function may need tuning.")
    
    elapsed = time.time() - t0
    log["total_time"] = round(elapsed, 1)
    with open(os.path.join(OUT, "nn_baseline_results.json"), "w") as f:
        json.dump(log, f, indent=2, default=str)
    print(f"\n  Done in {elapsed:.0f}s | Saved: {OUT}/nn_baseline_results.json")


if __name__ == "__main__":
    main()
