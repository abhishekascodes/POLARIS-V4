#!/usr/bin/env python3
"""
POLARIS v4 — LOCAL LLM BENCHMARK (RTX 5080)
=============================================
Runs downloaded Qwen models through POLARIS benchmark using transformers.
No API keys. No internet. Pure local GPU evaluation.

Models: Qwen2-0.5B, Qwen2.5-0.5B, Qwen2.5-3B, Qwen2.5-7B

Usage:
    python run_local_llm_benchmark.py              # all models, quick
    python run_local_llm_benchmark.py --full        # all models, full 20 scenarios
    python run_local_llm_benchmark.py --model 0.5b  # single model
"""
import sys, os, io, json, time, random, statistics, argparse, gc
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from server.policy_environment import PolicyEnvironment
from server.config import TASK_CONFIGS, VALID_ACTIONS, CORE_ACTIONS, ACTION_DESCRIPTIONS
from server.tasks import grade_trajectory
from polaris_bench.failure_detector import FailureDetector

OUT = "outputs/llm_benchmark"
os.makedirs(OUT, exist_ok=True)

MODELS = {
    "qwen2-0.5b": "Qwen/Qwen2-0.5B-Instruct",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
}

ACTION_LIST = "\n".join(f"  - {n}: {d}" for n, d in ACTION_DESCRIPTIONS.items())

SYSTEM_PROMPT = f"""You are the President of a simulated nation. Choose ONE policy action each turn.
Balance GDP, pollution, and satisfaction. Prevent collapse.

AVAILABLE ACTIONS:
{ACTION_LIST}

Respond with ONLY valid JSON:
{{"action": "<action_name>", "reasoning": "<1 sentence>", "coalition_target": [], "veto_prediction": [], "stance": "cooperative"}}"""

SYSTEM_SIMPLE = f"""Choose ONE policy action. Respond with ONLY the action name.
Actions: {', '.join(CORE_ACTIONS)}"""


def load_model(model_id):
    """Load model onto GPU."""
    print(f"  Loading {model_id}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    elapsed = time.time() - t0
    params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Loaded in {elapsed:.1f}s | {params:.1f}B params | Device: {next(model.parameters()).device}")
    return model, tokenizer


def call_local_llm(model, tokenizer, obs_text, use_negotiation=True):
    """Generate action from local model."""
    system = SYSTEM_PROMPT if use_negotiation else SYSTEM_SIMPLE
    messages = [{"role": "system", "content": system}, {"role": "user", "content": obs_text}]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=150 if use_negotiation else 20,
            temperature=0.1, do_sample=True, top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    tokens_used = len(new_tokens)
    
    if use_negotiation:
        try:
            if "```" in raw:
                raw = raw.split("```")[1].split("```")[0]
                if raw.startswith("json"): raw = raw[4:]
            data = json.loads(raw.strip())
            action = data.get("action", "no_action")
            if action not in VALID_ACTIONS:
                for a in VALID_ACTIONS:
                    if a in action: action = a; break
                else: action = "no_action"
            return {"action": action, "reasoning": data.get("reasoning",""),
                    "coalition_target": data.get("coalition_target",[]),
                    "veto_prediction": data.get("veto_prediction",[]),
                    "stance": data.get("stance","cooperative"), "_tokens": tokens_used, "_raw": raw[:200]}
        except (json.JSONDecodeError, IndexError):
            pass
    
    # Fallback: find action in raw text
    raw_lower = raw.lower().strip("'\"` \n")
    for a in VALID_ACTIONS:
        if a == raw_lower or a in raw_lower:
            return {"action": a, "_tokens": tokens_used, "_raw": raw[:200],
                    "coalition_target": [], "veto_prediction": [], "stance": "cooperative"}
    return {"action": "no_action", "_tokens": tokens_used, "_raw": raw[:200],
            "coalition_target": [], "veto_prediction": [], "stance": "cooperative"}


def format_obs(meta, step, max_steps):
    events = ", ".join(str(e) for e in meta.get("active_events", [])) or "none"
    neg = meta.get("negotiation_narrative", "")
    text = (f"Step {step}/{max_steps}\n"
            f"GDP: {meta.get('gdp_index',0):.0f}/200 | Pollution: {meta.get('pollution_index',0):.0f}/300 | "
            f"Satisfaction: {meta.get('public_satisfaction',0):.0f}/100\n"
            f"Healthcare: {meta.get('healthcare_index',0):.0f} | Education: {meta.get('education_index',0):.0f} | "
            f"Unemployment: {meta.get('unemployment_rate',0):.1f}%\n"
            f"Events: {events}")
    if neg: text += f"\n\nCOUNCIL:\n{neg[:400]}"
    return text


def run_episode(model, tokenizer, task_id, seed, use_neg=True, max_steps_override=None):
    """Run one full episode with a local LLM."""
    rng = random.Random(seed)
    env = PolicyEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    cfg = TASK_CONFIGS.get(task_id, {})
    max_steps = max_steps_override or cfg.get("max_steps", 200)
    
    total_reward, step = 0.0, 0
    actions, raw_outputs = [], []
    tom_correct, tom_total, coalitions = 0, 0, 0
    total_tokens = 0
    
    while not obs.done:
        step += 1
        meta = obs.metadata
        obs_text = format_obs(meta, step, max_steps)
        action_data = call_local_llm(model, tokenizer, obs_text, use_neg)
        
        obs = env.step(action_data)
        total_reward += obs.reward
        actions.append(action_data.get("action", "no_action"))
        raw_outputs.append(action_data.get("_raw", ""))
        total_tokens += action_data.get("_tokens", 0)
        
        outcome = obs.metadata.get("negotiation_outcome", {})
        if outcome.get("coalition_formed"): coalitions += 1
    
    score = grade_trajectory(task_id, env.get_trajectory())
    collapsed = obs.metadata.get("collapsed", step < max_steps)
    
    return {"score": score, "reward": round(total_reward, 4), "steps": step,
            "collapsed": collapsed, "actions": actions, "seed": seed,
            "total_tokens": total_tokens, "coalitions": coalitions,
            "unique_actions": len(set(actions)),
            "raw_samples": raw_outputs[:3]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all", choices=["all"]+list(MODELS.keys()))
    parser.add_argument("--full", action="store_true", help="Full 20-scenario benchmark")
    parser.add_argument("--seeds", type=str, default="42,123,777")
    args = parser.parse_args()
    
    seeds = [int(s) for s in args.seeds.split(",")]
    fd = FailureDetector()
    
    # Tasks to evaluate
    if args.full:
        tasks = ["environmental_recovery", "balanced_economy", "sustainable_governance",
                 "sustainable_governance_extreme", "negotiation_arena"]
    else:
        tasks = ["environmental_recovery", "negotiation_arena"]
    
    models_to_run = [args.model] if args.model != "all" else list(MODELS.keys())
    all_results = {}
    
    print("="*64)
    print("  POLARIS v4 — LOCAL LLM BENCHMARK")
    print(f"  Models: {models_to_run}")
    print(f"  Tasks: {tasks}")
    print(f"  Seeds: {seeds}")
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB" if torch.cuda.is_available() else "")
    print("="*64)
    
    for model_key in models_to_run:
        model_id = MODELS[model_key]
        print(f"\n{'#'*64}")
        print(f"  MODEL: {model_key} ({model_id})")
        print(f"{'#'*64}")
        
        try:
            model, tokenizer = load_model(model_id)
        except Exception as e:
            print(f"  FAILED to load: {e}")
            continue
        
        model_results = {"model": model_key, "model_id": model_id, "tasks": {}}
        
        for task_id in tasks:
            use_neg = task_id in ("negotiation_arena", "sustainable_governance_extreme")
            print(f"\n  --- {task_id} {'(with negotiation)' if use_neg else '(simple)'} ---")
            
            task_runs = []
            for seed in seeds:
                print(f"    seed={seed} ... ", end="", flush=True)
                t0 = time.time()
                result = run_episode(model, tokenizer, task_id, seed, use_neg)
                elapsed = time.time() - t0
                
                # Detect failures
                failures = fd.detect_all([], result["actions"])
                result["failure_modes"] = [f.mode for f in failures]
                result["wall_time"] = round(elapsed, 1)
                
                status = "COLLAPSED" if result["collapsed"] else "SURVIVED"
                print(f"{status} score={result['score']:.4f} | {result['steps']} steps | {elapsed:.1f}s | {result['unique_actions']} unique actions")
                task_runs.append(result)
            
            # Aggregate
            avg_score = statistics.mean(r["score"] for r in task_runs)
            std_score = statistics.stdev(r["score"] for r in task_runs) if len(task_runs) > 1 else 0
            collapse_rate = sum(1 for r in task_runs if r["collapsed"]) / len(task_runs)
            avg_time = statistics.mean(r["wall_time"] for r in task_runs)
            
            model_results["tasks"][task_id] = {
                "avg_score": round(avg_score, 4),
                "std_score": round(std_score, 4),
                "collapse_rate": round(collapse_rate, 2),
                "avg_wall_time": round(avg_time, 1),
                "runs": task_runs,
            }
            
            print(f"    => AVG: {avg_score:.4f} +/- {std_score:.4f} | collapse={collapse_rate:.0%} | {avg_time:.1f}s/ep")
        
        all_results[model_key] = model_results
        
        # Free GPU memory
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  GPU memory freed.")
    
    # Save all results
    path = os.path.join(OUT, "llm_results.json")
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Load baseline data if available
    baselines = {}
    baseline_path = "outputs/experiments/experiment_results.json"
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            bdata = json.load(f)
        tasks_data = bdata.get("tasks", {})
        for agent_name in ["random", "heuristic"]:
            baselines[agent_name] = {}
            for tid, td in tasks_data.items():
                if agent_name in td:
                    baselines[agent_name][tid] = td[agent_name]
    
    # Print THE comparison table (baselines + LLMs)
    print(f"\n{'='*90}")
    print(f"  POLARIS v4 — UNIFIED COMPARISON TABLE (Baselines + LLMs)")
    print(f"{'='*90}")
    
    # Negotiation arena comparison (THE key table)
    print(f"\n  NEGOTIATION ARENA (Multi-Agent, 5 Ministers)")
    print(f"  {'Model':<20} | {'Score':>12} | {'Collapse':>10} | {'CCR':>8} | {'Type':>10}")
    print(f"  {'-'*20}-+-{'-'*12}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}")
    
    # Baselines first
    for bname in ["random", "heuristic"]:
        if bname in baselines:
            neg = baselines[bname].get("negotiation_arena", {})
            env = baselines[bname].get("environmental_recovery", {})
            s_neg = neg.get("avg_score", 0)
            s_env = env.get("avg_score", 0)
            ccr = s_neg / s_env if s_env > 0 else 0
            cr = neg.get("collapse_rate", 0)
            print(f"  {bname.upper():<20} | {s_neg:>12.4f} | {cr:>9.0%} | {ccr:>8.4f} | {'baseline':>10}")
    
    # LLM models
    for mk, mr in all_results.items():
        neg = mr["tasks"].get("negotiation_arena", {})
        env = mr["tasks"].get("environmental_recovery", {})
        s_neg = neg.get("avg_score", 0)
        s_env = env.get("avg_score", 0)
        ccr = s_neg / s_env if s_env > 0 else 0
        cr = neg.get("collapse_rate", 0)
        std = neg.get("std_score", 0)
        print(f"  {mk:<20} | {s_neg:>7.4f}+/-{std:<4.4f} | {cr:>9.0%} | {ccr:>8.4f} | {'LLM':>10}")
    
    print(f"{'='*90}")
    
    # Single-agent comparison
    print(f"\n  ENVIRONMENTAL RECOVERY (Single-Agent, No Ministers)")
    print(f"  {'Model':<20} | {'Score':>12} | {'Collapse':>10}")
    print(f"  {'-'*20}-+-{'-'*12}-+-{'-'*10}")
    for bname in ["random", "heuristic"]:
        if bname in baselines:
            env = baselines[bname].get("environmental_recovery", {})
            print(f"  {bname.upper():<20} | {env.get('avg_score',0):>12.4f} | {env.get('collapse_rate',0):>9.0%}")
    for mk, mr in all_results.items():
        env = mr["tasks"].get("environmental_recovery", {})
        std = env.get("std_score", 0)
        print(f"  {mk:<20} | {env.get('avg_score',0):>7.4f}+/-{std:<4.4f} | {env.get('collapse_rate',0):>9.0%}")
    
    # KEY INSIGHT auto-generation
    print(f"\n{'='*90}")
    print(f"  KEY FINDINGS")
    print(f"{'='*90}")
    
    if len(all_results) >= 2:
        scores_by_size = [(mk, mr["tasks"].get("negotiation_arena",{}).get("avg_score",0)) for mk,mr in all_results.items()]
        scores_by_size.sort(key=lambda x: x[1], reverse=True)
        best = scores_by_size[0]
        worst = scores_by_size[-1]
        print(f"  1. Best LLM: {best[0]} (score={best[1]:.4f})")
        print(f"  2. Worst LLM: {worst[0]} (score={worst[1]:.4f})")
        
        # Check if scaling helps
        if "qwen2.5-0.5b" in all_results and "qwen2.5-7b" in all_results:
            s_small = all_results["qwen2.5-0.5b"]["tasks"].get("negotiation_arena",{}).get("avg_score",0)
            s_large = all_results["qwen2.5-7b"]["tasks"].get("negotiation_arena",{}).get("avg_score",0)
            if s_large > s_small * 1.1:
                print(f"  3. Scaling HELPS: 7B ({s_large:.4f}) > 0.5B ({s_small:.4f})")
            else:
                print(f"  3. Scaling does NOT fix coordination: 7B ({s_large:.4f}) vs 0.5B ({s_small:.4f})")
    
    # Universal collapse check
    all_collapse = all(mr["tasks"].get("negotiation_arena",{}).get("collapse_rate",1) >= 0.9 for mr in all_results.values())
    if all_collapse:
        print(f"  4. UNIVERSAL COLLAPSE: ALL models collapse >90% in multi-agent negotiation")
    
    print(f"\n  Results saved: {path}")
    print(f"  Run 'python generate_plots.py' for visualizations.")


if __name__ == "__main__":
    main()
