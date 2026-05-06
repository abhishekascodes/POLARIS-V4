#!/usr/bin/env python3
"""
POLARIS v4 — GRPO Trainer (Research-Grade)
============================================
Trains Qwen2-0.5B with Group Relative Policy Optimization.

KEY DIFFERENCES from naive training:
  1. Composite reward: R = task + alpha*coordination - beta*collapse
  2. Real multi-agent rollout (president + minister council negotiation)
  3. CCR tracked per episode for research logging
  4. Checkpoints every N steps

Hardware: RTX 5080 (17GB VRAM) | Qwen2-0.5B (~1GB)
"""
import sys, os, io, json, time, random, math, statistics, gc, copy
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM

from server.policy_environment import PolicyEnvironment
from server.config import TASK_CONFIGS, VALID_ACTIONS, CORE_ACTIONS
from server.tasks import grade_trajectory
from polaris_bench.failure_detector import FailureDetector

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════
MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"
OUT = "outputs/training_v4"
CKPT = os.path.join(OUT, "checkpoints")
os.makedirs(CKPT, exist_ok=True)

# Reward shaping
ALPHA_COORD = 0.3       # coordination bonus weight
BETA_COLLAPSE = 5.0     # collapse penalty
GAMMA_DIVERSITY = 0.1   # action diversity bonus
DELTA_SURVIVAL = 0.02   # per-step survival bonus

# GRPO
GROUP_SIZE = 4
TOP_K = 2
LR = 3e-6
KL_COEFF = 0.04
MAX_GRAD_NORM = 1.0
ROLLOUT_STEPS = 25      # steps per training rollout
TRAIN_STEPS = 150       # total training steps
EVAL_EVERY = 30
SAVE_EVERY = 50
MAX_TOKENS = 60
TEMP = 0.6

EVAL_SEEDS = [42, 123, 777]
TRAIN_SEEDS = list(range(200, 500))

SYSTEM = """You are President. Pick ONE action. Balance GDP, pollution, satisfaction.
You MUST coordinate with your ministers. Read their proposals carefully.
Form coalitions. Predict who will veto. Prevent collapse.

Actions: """ + ", ".join(CORE_ACTIONS) + """

Reply JSON: {"action":"<name>","reasoning":"<why>","coalition_target":["<minister>"],"veto_prediction":["<minister>"],"stance":"cooperative"}"""


def fmt(meta, step):
    ev = ", ".join(str(e) for e in meta.get("active_events",[])) or "none"
    neg = meta.get("negotiation_narrative", "")
    s = (f"Step {step}|GDP:{meta.get('gdp_index',0):.0f}|"
         f"Poll:{meta.get('pollution_index',0):.0f}|"
         f"Sat:{meta.get('public_satisfaction',0):.0f}|"
         f"HP:{meta.get('healthcare_index',0):.0f}|"
         f"Unemp:{meta.get('unemployment_rate',0):.1f}%|"
         f"Events:{ev}")
    if neg:
        s += f"\n\nMINISTER COUNCIL:\n{neg[:500]}"
    return s


def parse(raw):
    raw = raw.strip()
    try:
        if "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
            if raw.startswith("json"): raw = raw[4:]
        d = json.loads(raw.strip())
        a = d.get("action", "no_action")
        if a in VALID_ACTIONS:
            ct = d.get("coalition_target", [])
            if isinstance(ct, str): ct = [ct]
            ct = [str(x) for x in ct if isinstance(x, str)]
            vp = d.get("veto_prediction", [])
            if isinstance(vp, str): vp = [vp]
            vp = [str(x) for x in vp if isinstance(x, str)]
            return a, ct, vp, d.get("stance", "cooperative")
    except:
        pass
    for a in VALID_ACTIONS:
        if a in raw.lower():
            return a, [], [], "cooperative"
    return "no_action", [], [], "cooperative"


def shaped_reward(task_reward, step_reward, coalition_formed, cooperation_score,
                  collapsed, action, prev_actions, step):
    """
    Composite reward:
      R = task_component + alpha*coordination - beta*collapse + diversity + survival
    """
    r = step_reward

    # Coordination bonus
    coord = 0.0
    if coalition_formed:
        coord += 0.3
    coord += cooperation_score * 0.2
    r += ALPHA_COORD * coord

    # Collapse penalty
    if collapsed:
        r -= BETA_COLLAPSE

    # Action diversity bonus (avoid repeating same action)
    if len(prev_actions) >= 3:
        recent = prev_actions[-3:]
        unique = len(set(recent))
        if unique >= 3:
            r += GAMMA_DIVERSITY * 0.5
        elif unique == 1:
            r -= GAMMA_DIVERSITY * 0.3

    # Survival bonus
    r += DELTA_SURVIVAL

    return r


def rollout(model, tokenizer, task_id, seed, max_steps=ROLLOUT_STEPS, collect=True):
    """Full multi-agent rollout with negotiation council."""
    env = PolicyEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)

    transitions = []
    actions_taken = []
    total_shaped = 0.0
    total_raw = 0.0
    coalitions_formed = 0
    cooperation_scores = []
    step = 0

    while not obs.done and step < max_steps:
        step += 1
        meta = obs.metadata
        state_text = fmt(meta, step)

        msgs = [{"role":"system","content":SYSTEM},{"role":"user","content":state_text}]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536).to(model.device)

        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=MAX_TOKENS,
                                 temperature=TEMP, do_sample=True, top_p=0.9,
                                 pad_token_id=tokenizer.eos_token_id)
        new_toks = out[0][inp["input_ids"].shape[1]:]
        raw = tokenizer.decode(new_toks, skip_special_tokens=True)
        action, ct, vp, stance = parse(raw)

        action_data = {"action": action, "coalition_target": ct,
                       "veto_prediction": vp, "stance": stance}
        obs = env.step(action_data)

        # Extract negotiation info
        neg_out = obs.metadata.get("negotiation_outcome", {})
        cf = neg_out.get("coalition_formed", False)
        cs = neg_out.get("cooperation_score", 0.5)
        collapsed = obs.metadata.get("collapsed", False)

        if cf: coalitions_formed += 1
        cooperation_scores.append(cs)

        sr = shaped_reward(0, obs.reward, cf, cs, collapsed, action, actions_taken, step)
        total_shaped += sr
        total_raw += obs.reward
        actions_taken.append(action)

        if collect:
            transitions.append({
                "prompt_ids": inp["input_ids"][0],
                "response_ids": new_toks,
                "shaped_reward": sr,
            })

    score = grade_trajectory(task_id, env.get_trajectory())
    collapsed = obs.metadata.get("collapsed", step < max_steps)
    avg_coop = statistics.mean(cooperation_scores) if cooperation_scores else 0.0

    return {
        "transitions": transitions,
        "score": score,
        "raw_reward": round(total_raw, 4),
        "shaped_reward": round(total_shaped, 4),
        "collapsed": collapsed,
        "steps": step,
        "coalitions": coalitions_formed,
        "cooperation": round(avg_coop, 4),
        "unique_actions": len(set(actions_taken)),
        "actions": actions_taken,
    }


def compute_loss(model, ref_model, prompts, responses, advantages):
    total_loss = torch.tensor(0.0, device="cuda", requires_grad=True)
    n = 0
    for pid, rid, adv in zip(prompts, responses, advantages):
        full = torch.cat([pid, rid]).unsqueeze(0).to(model.device)
        pl = len(pid)
        if pl >= full.shape[1] - 1:
            continue

        out = model(full)
        logits = out.logits[0, pl-1:-1]
        tgt = full[0, pl:]

        lp = F.log_softmax(logits, dim=-1)
        tlp = lp.gather(1, tgt.unsqueeze(1)).squeeze(1).sum()

        with torch.no_grad():
            ro = ref_model(full)
            rl = ro.logits[0, pl-1:-1]
            rlp = F.log_softmax(rl, dim=-1)
            rtlp = rlp.gather(1, tgt.unsqueeze(1)).squeeze(1).sum()

        kl = tlp - rtlp
        loss = -(adv * tlp) + KL_COEFF * kl
        total_loss = total_loss + loss
        n += 1

    return total_loss / max(n, 1)


def evaluate(model, tokenizer, task_id, seeds):
    scores, collapses, coops, coals = [], [], [], []
    for s in seeds:
        r = rollout(model, tokenizer, task_id, s, max_steps=50, collect=False)
        scores.append(r["score"])
        collapses.append(1 if r["collapsed"] else 0)
        coops.append(r["cooperation"])
        coals.append(r["coalitions"])
    return {
        "score": round(statistics.mean(scores), 4),
        "std": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0,
        "collapse_rate": round(statistics.mean(collapses), 2),
        "cooperation": round(statistics.mean(coops), 4),
        "coalitions": round(statistics.mean(coals), 1),
    }


def main():
    print("="*64)
    print("  POLARIS v4 GRPO TRAINING")
    print(f"  Model: {MODEL_ID}")
    print(f"  Reward: task + {ALPHA_COORD}*coord - {BETA_COLLAPSE}*collapse")
    print(f"  GRPO: G={GROUP_SIZE} K={TOP_K} LR={LR}")
    print(f"  Steps: {TRAIN_STEPS} | Eval: every {EVAL_EVERY}")
    print("="*64)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    ref_model.eval()
    for p in ref_model.parameters(): p.requires_grad = False
    model.train()
    opt = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    fd = FailureDetector()

    log = {"config": {"model": MODEL_ID, "alpha": ALPHA_COORD, "beta": BETA_COLLAPSE,
                      "group_size": GROUP_SIZE, "lr": LR, "kl": KL_COEFF,
                      "train_steps": TRAIN_STEPS},
           "steps": [], "evals": []}

    # BEFORE eval
    print("\n--- BEFORE TRAINING ---")
    tasks = ["environmental_recovery", "negotiation_arena"]
    for tid in tasks:
        model.eval()
        ev = evaluate(model, tokenizer, tid, EVAL_SEEDS)
        print(f"  {tid}: score={ev['score']} collapse={ev['collapse_rate']:.0%} coop={ev['cooperation']}")
        log["evals"].append({"step": 0, "task": tid, "phase": "before", **ev})
        model.train()

    rng = random.Random(42)
    t0 = time.time()

    for step in range(1, TRAIN_STEPS + 1):
        tid = "negotiation_arena" if rng.random() < 0.7 else "environmental_recovery"
        seed = rng.choice(TRAIN_SEEDS)

        # GRPO: generate GROUP_SIZE rollouts
        group = []
        for g in range(GROUP_SIZE):
            r = rollout(model, tokenizer, tid, seed + g*1000, max_steps=ROLLOUT_STEPS)
            group.append(r)

        # Rank by shaped reward
        rewards = [g["shaped_reward"] for g in group]
        mu = statistics.mean(rewards)
        sd = statistics.stdev(rewards) if len(rewards) > 1 else 1.0
        advantages = [(r - mu) / max(sd, 0.01) for r in rewards]

        ranked = sorted(range(GROUP_SIZE), key=lambda i: rewards[i], reverse=True)
        pos_idx = ranked[:TOP_K]
        neg_idx = ranked[-TOP_K:]

        prompts, responses, advs = [], [], []
        for i in pos_idx:
            for t in group[i]["transitions"][:4]:
                prompts.append(t["prompt_ids"])
                responses.append(t["response_ids"])
                advs.append(torch.tensor(max(advantages[i], 0.1), device="cuda"))
        for i in neg_idx:
            for t in group[i]["transitions"][:4]:
                prompts.append(t["prompt_ids"])
                responses.append(t["response_ids"])
                advs.append(torch.tensor(min(advantages[i], -0.1), device="cuda"))

        if not prompts:
            continue

        loss = compute_loss(model, ref_model, prompts, responses, advs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        opt.step()
        opt.zero_grad()

        lv = loss.item()
        best = group[ranked[0]]
        log["steps"].append({"step": step, "loss": round(lv, 6),
                            "best_score": best["score"], "best_reward": best["shaped_reward"],
                            "collapsed": best["collapsed"], "coalitions": best["coalitions"],
                            "cooperation": best["cooperation"],
                            "group_mean": round(mu, 4), "task": tid})

        if step % 5 == 0:
            el = time.time() - t0
            eta = el / step * (TRAIN_STEPS - step)
            print(f"  [{step}/{TRAIN_STEPS}] loss={lv:.4f} best={best['score']:.4f} "
                  f"coop={best['cooperation']:.2f} coal={best['coalitions']} "
                  f"{'DEAD' if best['collapsed'] else 'ALIVE'} | {el:.0f}s (ETA {eta:.0f}s)")

        if step % EVAL_EVERY == 0:
            model.eval()
            print(f"\n  --- EVAL step {step} ---")
            for etid in tasks:
                ev = evaluate(model, tokenizer, etid, EVAL_SEEDS)
                print(f"    {etid}: {ev['score']} collapse={ev['collapse_rate']:.0%} coop={ev['cooperation']}")
                log["evals"].append({"step": step, "task": etid, "phase": "training", **ev})
            model.train()

        if step % SAVE_EVERY == 0:
            cp = os.path.join(CKPT, f"step_{step}")
            model.save_pretrained(cp)
            tokenizer.save_pretrained(cp)
            print(f"  Saved: {cp}")

        if step % 10 == 0:
            with open(os.path.join(OUT, "training_log.json"), "w") as f:
                json.dump(log, f, indent=2, default=str)

    # AFTER eval
    model.eval()
    print(f"\n{'='*64}")
    print("  AFTER TRAINING")
    print(f"{'='*64}")
    for tid in tasks:
        ev = evaluate(model, tokenizer, tid, EVAL_SEEDS)
        print(f"  {tid}: {ev['score']} collapse={ev['collapse_rate']:.0%} coop={ev['cooperation']}")
        log["evals"].append({"step": TRAIN_STEPS, "task": tid, "phase": "after", **ev})

    # CCR
    before_s = [e for e in log["evals"] if e["phase"]=="before" and e["task"]=="environmental_recovery"]
    before_m = [e for e in log["evals"] if e["phase"]=="before" and e["task"]=="negotiation_arena"]
    after_s = [e for e in log["evals"] if e["phase"]=="after" and e["task"]=="environmental_recovery"]
    after_m = [e for e in log["evals"] if e["phase"]=="after" and e["task"]=="negotiation_arena"]

    if before_s and before_m and after_s and after_m:
        ccr_before = before_m[0]["score"] / max(before_s[0]["score"], 0.01)
        ccr_after = after_m[-1]["score"] / max(after_s[-1]["score"], 0.01)
        print(f"\n  CCR BEFORE: {ccr_before:.4f}")
        print(f"  CCR AFTER:  {ccr_after:.4f}")
        print(f"  CCR DELTA:  {ccr_after - ccr_before:+.4f}")
        log["ccr"] = {"before": round(ccr_before, 4), "after": round(ccr_after, 4),
                      "delta": round(ccr_after - ccr_before, 4)}

    fp = os.path.join(CKPT, "final")
    model.save_pretrained(fp)
    tokenizer.save_pretrained(fp)

    log["total_time"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, "training_log.json"), "w") as f:
        json.dump(log, f, indent=2, default=str)

    print(f"\nDone in {log['total_time']/60:.1f}min | Log: {OUT}/training_log.json")


if __name__ == "__main__":
    main()
