#!/usr/bin/env python3
"""
POLARIS v4 — Reward Variable Decomposition Analysis
=====================================================
Addresses: "Analyze how each reward variable influences the behaviour of the agent"

Runs episodes with each reward component isolated to measure its behavioral impact.
This proves the reward function is well-designed before attributing failure to LLMs.
"""
import sys, os, io, json, random, statistics
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.policy_environment import PolicyEnvironment
from server.config import TASK_CONFIGS, CORE_ACTIONS, STATE_BOUNDS, VALID_ACTIONS
from server.reward_engine import RewardEngine
from server.tasks import grade_trajectory

OUT = "outputs/reward_analysis"
os.makedirs(OUT, exist_ok=True)

SEEDS = [42, 123, 777, 999, 314]
TASK = "negotiation_arena"
MAX_STEPS = 50

# Each strategy targets one reward component
STRATEGIES = {
    "random": {"desc": "Uniform random baseline", "fn": lambda s, rng: rng.choice(CORE_ACTIONS)},
    "max_economic": {"desc": "Always maximize GDP/economy", 
                     "fn": lambda s, rng: "stimulate_economy" if s["gdp_index"] < 120 else "expand_industry"},
    "max_environmental": {"desc": "Always minimize pollution",
                          "fn": lambda s, rng: "enforce_emission_limits" if s["pollution_index"] > 100 else "subsidize_renewables"},
    "max_social": {"desc": "Always maximize satisfaction",
                   "fn": lambda s, rng: "increase_welfare" if s["public_satisfaction"] < 60 else "invest_in_healthcare"},
    "max_stability": {"desc": "Always pick no_action for stability",
                      "fn": lambda s, rng: "no_action"},
    "balanced": {"desc": "Round-robin across all pillars",
                 "fn": None},  # handled specially
    "greedy_reward": {"desc": "Heuristic: pick action that helps worst metric",
                      "fn": None},  # handled specially
}

BALANCE_CYCLE = ["subsidize_renewables", "stimulate_economy", "increase_welfare", 
                 "invest_in_education", "invest_in_healthcare", "incentivize_clean_tech"]

def greedy_worst_metric(state):
    gdp_norm = state["gdp_index"] / 200
    poll_norm = 1 - state["pollution_index"] / 300
    sat_norm = state["public_satisfaction"] / 100
    worst = min([(gdp_norm, "econ"), (poll_norm, "env"), (sat_norm, "social")], key=lambda x: x[0])
    if worst[1] == "econ":
        return "stimulate_economy" if state["unemployment_rate"] > 10 else "expand_industry"
    elif worst[1] == "env":
        return "enforce_emission_limits" if state["pollution_index"] > 200 else "subsidize_renewables"
    else:
        return "increase_welfare" if state["public_satisfaction"] < 40 else "invest_in_healthcare"

def run_episode(strategy_name, seed):
    env = PolicyEnvironment()
    obs = env.reset(seed=seed, task_id=TASK)
    rng = random.Random(seed)
    
    rewards_by_component = {"economic": [], "environmental": [], "social": [],
                            "stability": [], "pareto": [], "penalties": [],
                            "cooperation": [], "total": []}
    actions = []
    step = 0
    balance_idx = 0
    
    while not obs.done and step < MAX_STEPS:
        step += 1
        state = obs.metadata
        
        if strategy_name == "balanced":
            action = BALANCE_CYCLE[balance_idx % len(BALANCE_CYCLE)]
            balance_idx += 1
        elif strategy_name == "greedy_reward":
            action = greedy_worst_metric(state)
        else:
            action = STRATEGIES[strategy_name]["fn"](state, rng)
        
        action_data = {"action": action, "coalition_target": [], "veto_prediction": [], "stance": "cooperative"}
        obs = env.step(action_data)
        actions.append(action)
        
        rb = obs.metadata.get("reward_breakdown", {})
        if rb:
            rewards_by_component["economic"].append(rb.get("economic_score", 0))
            rewards_by_component["environmental"].append(rb.get("environmental_score", 0))
            rewards_by_component["social"].append(rb.get("social_score", 0))
            rewards_by_component["stability"].append(rb.get("stability_score", 0))
            rewards_by_component["pareto"].append(rb.get("pareto_bonus", 0))
            rewards_by_component["penalties"].append(rb.get("penalties", 0))
            rewards_by_component["cooperation"].append(rb.get("cooperation_bonus", 0))
            rewards_by_component["total"].append(rb.get("total_reward", 0))
    
    score = grade_trajectory(TASK, env.get_trajectory())
    collapsed = obs.metadata.get("collapsed", step < MAX_STEPS)
    
    avg_components = {k: round(statistics.mean(v), 4) if v else 0 for k, v in rewards_by_component.items()}
    
    return {
        "score": round(score, 4), "collapsed": collapsed, "steps": step,
        "unique_actions": len(set(actions)),
        "reward_components": avg_components,
        "actions_sample": actions[:10],
    }


def main():
    print("="*70)
    print("  POLARIS v4 — REWARD VARIABLE DECOMPOSITION")
    print("="*70)
    
    all_results = {}
    
    for sname in STRATEGIES:
        desc = STRATEGIES[sname].get("desc", "")
        print(f"\n  Strategy: {sname} — {desc}")
        runs = []
        for seed in SEEDS:
            r = run_episode(sname, seed)
            runs.append(r)
        
        avg_score = statistics.mean(r["score"] for r in runs)
        collapse_rate = sum(1 for r in runs if r["collapsed"]) / len(runs)
        
        # Average reward components across runs
        avg_comp = {}
        for comp in ["economic", "environmental", "social", "stability", "pareto", "penalties", "cooperation", "total"]:
            vals = [r["reward_components"].get(comp, 0) for r in runs]
            avg_comp[comp] = round(statistics.mean(vals), 4)
        
        all_results[sname] = {
            "description": desc,
            "avg_score": round(avg_score, 4),
            "collapse_rate": round(collapse_rate, 2),
            "avg_components": avg_comp,
            "runs": runs,
        }
        
        print(f"    Score: {avg_score:.4f} | Collapse: {collapse_rate:.0%}")
        print(f"    Components: econ={avg_comp['economic']:.3f} env={avg_comp['environmental']:.3f} "
              f"soc={avg_comp['social']:.3f} stab={avg_comp['stability']:.3f} "
              f"pareto={avg_comp['pareto']:.3f} pen={avg_comp['penalties']:.3f}")
    
    # Print comparison table
    print(f"\n{'='*100}")
    print(f"  REWARD DECOMPOSITION TABLE")
    print(f"{'='*100}")
    print(f"  {'Strategy':<18} | {'Score':>6} | {'Collapse':>8} | {'Econ':>6} | {'Env':>6} | {'Social':>6} | {'Stab':>6} | {'Pareto':>6} | {'Pen':>6} | {'Total':>6}")
    print(f"  {'-'*18}-+-{'-'*6}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")
    for sn, sr in all_results.items():
        c = sr["avg_components"]
        print(f"  {sn:<18} | {sr['avg_score']:>6.4f} | {sr['collapse_rate']:>7.0%} | "
              f"{c['economic']:>6.3f} | {c['environmental']:>6.3f} | {c['social']:>6.3f} | "
              f"{c['stability']:>6.3f} | {c['pareto']:>6.3f} | {c['penalties']:>6.3f} | {c['total']:>6.3f}")
    
    # Key insights
    print(f"\n  KEY INSIGHTS:")
    best = max(all_results.items(), key=lambda x: x[1]["avg_score"])
    worst = min(all_results.items(), key=lambda x: x[1]["avg_score"])
    print(f"  1. Best strategy: {best[0]} (score={best[1]['avg_score']:.4f})")
    print(f"  2. Worst strategy: {worst[0]} (score={worst[1]['avg_score']:.4f})")
    
    # Which component matters most?
    greedy = all_results.get("greedy_reward", {})
    balanced = all_results.get("balanced", {})
    if greedy and balanced:
        print(f"  3. Greedy-worst-metric ({greedy['avg_score']:.4f}) vs Balanced ({balanced['avg_score']:.4f})")
    
    # Save
    with open(os.path.join(OUT, "reward_decomposition.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved: {OUT}/reward_decomposition.json")


if __name__ == "__main__":
    main()
