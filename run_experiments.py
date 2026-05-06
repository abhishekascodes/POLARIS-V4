#!/usr/bin/env python3
"""
POLARIS v4 — MASTER EXPERIMENT RUNNER
No API keys needed. Runs baselines across all conditions.
Generates: scaling, chaos, adversarial, ablation data.

Usage: python run_experiments.py
       python run_experiments.py --quick
"""
import sys, os, io, json, time, random, statistics, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.policy_environment import PolicyEnvironment
from server.config import TASK_CONFIGS, CORE_ACTIONS, VALID_ACTIONS
from server.tasks import grade_trajectory

SEEDS = [42, 123, 777, 1337, 2024]
OUT = "outputs/experiments"

# ── AGENTS ──
def agent_random(meta, rng):
    return {"action": rng.choice(CORE_ACTIONS)}

def agent_heuristic(meta, rng):
    sat = meta.get("public_satisfaction", 50)
    poll = meta.get("pollution_index", 100)
    gdp = meta.get("gdp_index", 100)
    if sat < 30: return {"action": "increase_welfare", "coalition_target": [], "veto_prediction": [], "stance": "cooperative"}
    if poll > 200: return {"action": "enforce_emission_limits", "coalition_target": [], "veto_prediction": [], "stance": "cooperative"}
    if gdp < 50: return {"action": "stimulate_economy", "coalition_target": [], "veto_prediction": [], "stance": "cooperative"}
    return {"action": rng.choice(["subsidize_renewables","invest_in_education","increase_welfare","stimulate_economy"]),
            "coalition_target": [], "veto_prediction": [], "stance": "cooperative"}

def agent_greedy_gdp(meta, rng):
    return {"action": rng.choice(["stimulate_economy","decrease_tax","expand_industry","reduce_interest_rates"]),
            "coalition_target": [], "veto_prediction": [], "stance": "cooperative"}

def agent_greedy_green(meta, rng):
    return {"action": rng.choice(["subsidize_renewables","enforce_emission_limits","implement_carbon_tax","incentivize_clean_tech"]),
            "coalition_target": [], "veto_prediction": [], "stance": "cooperative"}

AGENTS = {"random": agent_random, "heuristic": agent_heuristic, "greedy_gdp": agent_greedy_gdp, "greedy_green": agent_greedy_green}

# ── RUN ONE EPISODE ──
def run_episode(task_id, seed, agent_fn, task_override=None):
    rng = random.Random(seed)
    if task_override:
        TASK_CONFIGS[task_id] = task_override
    env = PolicyEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    total_reward, step, actions = 0.0, 0, []
    tom_correct, tom_total, coalitions = 0, 0, 0
    while not obs.done:
        step += 1
        action_data = agent_fn(obs.metadata, rng)
        obs = env.step(action_data)
        total_reward += obs.reward
        actions.append(action_data.get("action","no_action"))
        outcome = obs.metadata.get("negotiation_outcome", {})
        if "veto_prediction_correct" in outcome:
            tom_total += 1
            if outcome["veto_prediction_correct"]: tom_correct += 1
        if outcome.get("coalition_formed"): coalitions += 1
    score = grade_trajectory(task_id, env.get_trajectory())
    return {"score": score, "reward": total_reward, "steps": step,
            "collapsed": obs.metadata.get("collapsed", step < TASK_CONFIGS.get(task_id,{}).get("max_steps",200)),
            "tom_acc": tom_correct/max(tom_total,1) if tom_total>0 else None,
            "coalitions": coalitions, "actions": actions, "seed": seed}

# ── EXPERIMENT 1: SCALING (2,5,8,12 agents) ──
def exp_scaling(seeds, agents_list=[2,5,8,12]):
    print("\n[EXP 1] AGENT SCALING")
    results = {}
    base = dict(TASK_CONFIGS["negotiation_arena"])
    for n in agents_list:
        cfg = dict(base)
        cfg["num_ministers"] = n
        cfg["max_steps"] = 150
        tag = f"scale_{n}"
        results[tag] = {}
        for aname, afn in AGENTS.items():
            runs = []
            for s in seeds:
                r = run_episode("negotiation_arena", s, afn, cfg)
                runs.append(r)
            avg_score = statistics.mean(r["score"] for r in runs)
            avg_reward = statistics.mean(r["reward"] for r in runs)
            collapse_rate = sum(1 for r in runs if r["collapsed"])/len(runs)
            results[tag][aname] = {"avg_score": round(avg_score,4), "avg_reward": round(avg_reward,2),
                                    "collapse_rate": round(collapse_rate,2), "runs": len(runs),
                                    "std_score": round(statistics.stdev(r["score"] for r in runs),4) if len(runs)>1 else 0}
            print(f"  {n} agents | {aname:12} | score={avg_score:.4f} +/- {results[tag][aname]['std_score']:.4f} | collapse={collapse_rate:.0%}")
    return results

# ── EXPERIMENT 2: CHAOS LEVELS ──
def exp_chaos(seeds, levels=[0.0, 0.3, 0.6, 0.9, 1.0]):
    print("\n[EXP 2] CHAOS SCALING")
    results = {}
    base = dict(TASK_CONFIGS["negotiation_arena"])
    for chaos in levels:
        cfg = dict(base); cfg["chaos_level"] = chaos; cfg["max_steps"] = 150
        tag = f"chaos_{chaos}"
        results[tag] = {}
        for aname in ["random","heuristic"]:
            runs = [run_episode("negotiation_arena", s, AGENTS[aname], cfg) for s in seeds]
            avg = statistics.mean(r["score"] for r in runs)
            cr = sum(1 for r in runs if r["collapsed"])/len(runs)
            std = statistics.stdev(r["score"] for r in runs) if len(runs)>1 else 0
            results[tag][aname] = {"avg_score": round(avg,4), "collapse_rate": round(cr,2), "std": round(std,4)}
            print(f"  chaos={chaos:.1f} | {aname:12} | score={avg:.4f} | collapse={cr:.0%}")
    return results

# ── EXPERIMENT 3: TASK DIFFICULTY ──
def exp_tasks(seeds):
    print("\n[EXP 3] TASK DIFFICULTY COMPARISON")
    results = {}
    for tid in ["environmental_recovery","balanced_economy","sustainable_governance","negotiation_arena"]:
        results[tid] = {}
        for aname in ["random","heuristic"]:
            runs = [run_episode(tid, s, AGENTS[aname]) for s in seeds]
            avg = statistics.mean(r["score"] for r in runs)
            cr = sum(1 for r in runs if r["collapsed"])/len(runs)
            std = statistics.stdev(r["score"] for r in runs) if len(runs)>1 else 0
            results[tid][aname] = {"avg_score": round(avg,4), "collapse_rate": round(cr,2), "std": round(std,4)}
            print(f"  {tid:35} | {aname:12} | score={avg:.4f} | collapse={cr:.0%}")
    return results

# ── EXPERIMENT 4: ABLATION (reward components) ──
def exp_ablation(seeds):
    print("\n[EXP 4] REWARD ABLATION")
    results = {}
    configs = {
        "full": {},
        "no_events": {"events_enabled": False, "event_frequency_multiplier": 0.0},
        "no_drift": {"drift_enabled": False},
        "max_chaos": {"chaos_level": 1.0, "satisfaction_event_scale": 1.0},
        "no_negotiation": {"negotiation_enabled": False, "num_ministers": 1},
    }
    base = dict(TASK_CONFIGS["negotiation_arena"])
    for label, overrides in configs.items():
        cfg = {**base, **overrides}; cfg["max_steps"] = 150
        runs = [run_episode("negotiation_arena", s, AGENTS["heuristic"], cfg) for s in seeds]
        avg = statistics.mean(r["score"] for r in runs)
        cr = sum(1 for r in runs if r["collapsed"])/len(runs)
        std = statistics.stdev(r["score"] for r in runs) if len(runs)>1 else 0
        results[label] = {"avg_score": round(avg,4), "collapse_rate": round(cr,2), "std": round(std,4)}
        print(f"  {label:20} | score={avg:.4f} +/- {std:.4f} | collapse={cr:.0%}")
    return results

# ── EXPERIMENT 5: EPISODE TRACES (qualitative) ──
def exp_traces(seed=42):
    print("\n[EXP 5] EPISODE TRACES")
    traces = {}
    for tid in ["environmental_recovery", "negotiation_arena"]:
        r = run_episode(tid, seed, AGENTS["heuristic"])
        traces[tid] = {
            "score": r["score"], "reward": round(r["reward"],2), "steps": r["steps"],
            "collapsed": r["collapsed"], "actions": r["actions"][:30],
            "unique_actions": len(set(r["actions"])), "total_actions": len(r["actions"]),
        }
        status = "COLLAPSED" if r["collapsed"] else "SURVIVED"
        print(f"  {tid}: {status} | score={r['score']:.4f} | {r['steps']} steps | {len(set(r['actions']))} unique actions")
    return traces

# ── MAIN ──
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="3 seeds instead of 5")
    args = parser.parse_args()
    
    seeds = SEEDS[:3] if args.quick else SEEDS
    os.makedirs(OUT, exist_ok=True)
    
    print("="*64)
    print("  POLARIS v4 — MASTER EXPERIMENT SUITE")
    print(f"  Seeds: {seeds}")
    print("="*64)
    
    t0 = time.time()
    all_results = {}
    all_results["scaling"] = exp_scaling(seeds)
    all_results["chaos"] = exp_chaos(seeds)
    all_results["tasks"] = exp_tasks(seeds)
    all_results["ablation"] = exp_ablation(seeds)
    all_results["traces"] = exp_traces()
    elapsed = time.time() - t0
    
    # Save
    path = os.path.join(OUT, "experiment_results.json")
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {path}")
    print(f"Total time: {elapsed:.1f}s")
    
    # Generate plots
    try:
        from generate_plots import generate_all_plots
        generate_all_plots(path)
    except ImportError:
        print("Run generate_plots.py separately for visualizations.")
    
    print(f"\n{'='*64}")
    print(f"  ALL EXPERIMENTS COMPLETE")
    print(f"{'='*64}")

if __name__ == "__main__":
    main()
