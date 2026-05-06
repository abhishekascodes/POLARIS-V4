"""
POLARIS-Bench v4 — Multi-Model Evaluator
==========================================

The core engine that runs ANY LLM through the full POLARIS benchmark.
Supports local models (via vLLM/transformers), OpenAI-compatible APIs
(Groq, Together, OpenAI, etc.), and Anthropic/Google APIs.

Usage:
    evaluator = PolarisEvaluator()
    
    # Evaluate via OpenAI-compatible API
    results = evaluator.evaluate_model(
        model_name="llama-3.3-70b-versatile",
        api_base="https://api.groq.com/openai/v1",
        api_key="gsk_...",
        scenarios="all",  # or ["coord_resource_allocation", "tom_veto_prediction"]
        seeds=[42, 123, 777],
    )
    
    # Generate report
    results.compute_composites()
    report = BenchmarkReport(results)
    report.save("outputs/llama_70b_results/")
"""

from __future__ import annotations

import json
import os
import sys
import time
import random
import statistics
from typing import Any, Dict, List, Optional, Tuple

# Add parent dir for server imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI

from server.policy_environment import PolicyEnvironment
from server.config import VALID_ACTIONS, ACTION_DESCRIPTIONS, TASK_CONFIGS
from server.tasks import grade_trajectory

from .scenarios import SCENARIOS, get_all_scenario_ids, DIMENSIONS
from .metrics import (
    CoordinationMetrics, ModelResults, extract_metrics,
    compute_ccr, compute_ccr_from_results,
)
from .failure_detector import FailureDetector


# ═══════════════════════════════════════════════════════════════
# LLM ADAPTERS
# ═══════════════════════════════════════════════════════════════

MINISTERS = ["Chancellor Voss", "Director Okafor", "Dr. Vasquez",
             "General Tanaka", "Senator Mwangi",
             # Extended for 8+ agent scenarios
             "Minister Patel", "Advisor Chen", "Secretary Kim",
             "Envoy Müller", "Commissioner Osei", "Delegate Yamamoto",
             "Attaché da Silva"]

ACTION_LIST_STR = "\n".join(f"  - {n}: {d}" for n, d in ACTION_DESCRIPTIONS.items())

SYSTEM_NEGOTIATION = """You are the President of a simulated nation. Each turn, your council of ministers presents proposals. You must:
1. Read each minister's proposal, argument, and coalition offer
2. Decide which policy action to take
3. Choose which ministers to form a coalition with
4. Predict which ministers might veto your decision

AVAILABLE ACTIONS:
{actions}

RESPONSE FORMAT — respond with valid JSON only, no markdown:
{{
  "action": "<action_name>",
  "reasoning": "<1-2 sentences>",
  "coalition_target": ["<minister_name>"],
  "veto_prediction": ["<minister_name_who_might_veto>"],
  "stance": "cooperative"
}}

RULES:
- action MUST be one of the valid actions listed above
- Balance GDP, pollution, and satisfaction
- Prevent collapse: GDP > 15, pollution < 290, satisfaction > 5
- React to briefings and events
Respond with ONLY valid JSON."""

SYSTEM_SIMPLE = """You are an expert AI policy advisor governing a simulated nation.
Each turn you must choose EXACTLY ONE policy action.

AVAILABLE ACTIONS:
{actions}

Respond with ONLY the action name. Nothing else."""


def format_obs_negotiation(meta: Dict, step: int, max_steps: int) -> str:
    lines = [
        f"--- STEP {step}/{max_steps} ---",
        f"GDP: {meta.get('gdp_index',0):.0f}/200 | Pollution: {meta.get('pollution_index',0):.0f}/300 | Satisfaction: {meta.get('public_satisfaction',0):.0f}/100",
        f"Healthcare: {meta.get('healthcare_index',0):.0f} | Education: {meta.get('education_index',0):.0f} | Unemployment: {meta.get('unemployment_rate',0):.1f}%",
    ]
    events = meta.get("active_events", [])
    if events:
        lines.append(f"Events: {', '.join(str(e) for e in events)}")
    neg = meta.get("negotiation_narrative", "")
    if neg:
        lines.append(f"\n{neg[:600]}")
    briefings = meta.get("active_briefings", [])
    if briefings:
        lines.append("\nBRIEFINGS:")
        for b in briefings[:3]:
            if isinstance(b, dict):
                lines.append(f"  [{b.get('category','')}] ...deadline step {b.get('deadline_step','')} ({b.get('steps_remaining','')} left)")
    new_b = meta.get("new_briefing", "")
    if new_b:
        lines.append(f"\nNEW INTEL: {new_b[:200]}")
    return "\n".join(lines)


def format_obs_simple(meta: Dict, step: int, max_steps: int) -> str:
    events = ", ".join(str(e) for e in meta.get("active_events", [])) or "none"
    return (
        f"Step {step}/{max_steps} | GDP: {meta.get('gdp_index',0):.0f} | "
        f"Pollution: {meta.get('pollution_index',0):.0f} | "
        f"Satisfaction: {meta.get('public_satisfaction',0):.0f} | Events: {events}"
    )


def call_llm(client: OpenAI, obs_text: str, model: str, use_negotiation: bool) -> Dict:
    """Call LLM and parse response."""
    system = SYSTEM_NEGOTIATION.format(actions=ACTION_LIST_STR) if use_negotiation else SYSTEM_SIMPLE.format(actions=ACTION_LIST_STR)
    
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": obs_text},
            ],
            temperature=0.1,
            max_tokens=250 if use_negotiation else 30,
        )
        raw = resp.choices[0].message.content.strip()
        tokens = getattr(resp.usage, 'total_tokens', 0) if resp.usage else 0
        
        if use_negotiation:
            # Parse JSON response
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                for a in VALID_ACTIONS:
                    if a in raw.lower():
                        return {"action": a, "reasoning": "parse fallback", "coalition_target": [], "veto_prediction": [], "stance": "cooperative", "_tokens": tokens}
                return {"action": "no_action", "reasoning": "parse error", "coalition_target": [], "veto_prediction": [], "stance": "cooperative", "_tokens": tokens}
            
            action = data.get("action", "no_action")
            if action not in VALID_ACTIONS:
                for a in VALID_ACTIONS:
                    if a in action:
                        action = a
                        break
                else:
                    action = "no_action"
            data["action"] = action
            data.setdefault("reasoning", "")
            data.setdefault("coalition_target", [])
            data.setdefault("veto_prediction", [])
            data.setdefault("stance", "cooperative")
            data["_tokens"] = tokens
            return data
        else:
            raw_clean = raw.lower().strip("'\"` \n")
            for a in VALID_ACTIONS:
                if a == raw_clean or a in raw_clean:
                    return {"action": a, "_tokens": tokens}
            return {"action": "no_action", "_tokens": tokens}
    except Exception as e:
        return {"action": "no_action", "reasoning": f"error: {e}", "coalition_target": [], "veto_prediction": [], "stance": "cooperative", "_tokens": 0}


# ═══════════════════════════════════════════════════════════════
# HEURISTIC BASELINES
# ═══════════════════════════════════════════════════════════════

def agent_random(meta: Dict, rng: random.Random) -> Dict:
    from server.config import CORE_ACTIONS
    return {"action": rng.choice(CORE_ACTIONS)}

def agent_heuristic(meta: Dict, rng: random.Random) -> Dict:
    sat = meta.get("public_satisfaction", 50)
    poll = meta.get("pollution_index", 100)
    gdp = meta.get("gdp_index", 100)
    if sat < 30: action = "increase_welfare"
    elif poll > 200: action = "enforce_emission_limits"
    elif gdp < 50: action = "stimulate_economy"
    else: action = rng.choice(["subsidize_renewables", "invest_in_education", "increase_welfare", "stimulate_economy"])
    return {"action": action, "reasoning": "heuristic", "coalition_target": [MINISTERS[1]], "veto_prediction": [], "stance": "cooperative"}


# ═══════════════════════════════════════════════════════════════
# SCENARIO → TASK ADAPTER
# ═══════════════════════════════════════════════════════════════

def scenario_to_task_config(scenario: Dict) -> Dict:
    """Convert a POLARIS-Bench scenario to a task config the env understands."""
    # Map scenario to closest existing task config, with overrides
    num_ministers = scenario.get("num_ministers", 5)
    neg_enabled = scenario.get("negotiation_enabled", True)
    
    if num_ministers <= 1:
        base_task = "environmental_recovery"
    elif neg_enabled:
        base_task = "negotiation_arena"
    else:
        base_task = "sustainable_governance"
    
    # Build config with scenario overrides
    config = dict(TASK_CONFIGS.get(base_task, TASK_CONFIGS["negotiation_arena"]))
    
    # Apply scenario-specific settings
    config["max_steps"] = scenario.get("max_steps", config["max_steps"])
    config["num_ministers"] = num_ministers
    config["events_enabled"] = scenario.get("events_enabled", True)
    config["event_frequency_multiplier"] = scenario.get("event_frequency_multiplier", 1.0)
    config["chaos_level"] = scenario.get("chaos_level", 0.6)
    config["drift_enabled"] = scenario.get("drift_enabled", True)
    config["negotiation_enabled"] = scenario.get("negotiation_enabled", True)
    config["briefing_enabled"] = scenario.get("briefing_enabled", True)
    config["minister_mode"] = scenario.get("minister_mode", "scripted")
    
    if "satisfaction_event_scale" in scenario:
        config["satisfaction_event_scale"] = scenario["satisfaction_event_scale"]
    if "satisfaction_floor_damping" in scenario:
        config["satisfaction_floor_damping"] = scenario["satisfaction_floor_damping"]
    if "crisis_welfare_bonus" in scenario:
        config["crisis_welfare_bonus"] = scenario["crisis_welfare_bonus"]
    if "initial_state_overrides" in scenario:
        config["initial_state_overrides"] = scenario["initial_state_overrides"]
    
    return config


# ═══════════════════════════════════════════════════════════════
# MAIN EVALUATOR
# ═══════════════════════════════════════════════════════════════

class PolarisEvaluator:
    """
    The POLARIS-Bench evaluation engine.
    
    Runs any LLM through the full 20-scenario benchmark suite
    and produces comprehensive metrics, failure analysis, and
    publication-quality reports.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.failure_detector = FailureDetector()
    
    def evaluate_model(
        self,
        model_name: str,
        api_base: str = "https://api.groq.com/openai/v1",
        api_key: str = "",
        model_family: str = "",
        model_params: str = "",
        scenarios: str | List[str] = "all",
        seeds: List[int] = [42, 123, 777],
        include_baselines: bool = True,
        output_dir: str = "outputs/polaris_bench",
    ) -> ModelResults:
        """
        Run the full POLARIS-Bench evaluation for a model.
        
        Args:
            model_name: Model identifier (e.g., "llama-3.3-70b-versatile")
            api_base: OpenAI-compatible API base URL
            api_key: API key
            model_family: Model family (e.g., "llama", "qwen", "gpt")
            model_params: Parameter count string (e.g., "70B")
            scenarios: "all" or list of scenario IDs
            seeds: Random seeds for statistical significance
            include_baselines: Also run heuristic and random baselines
            output_dir: Directory to save results
            
        Returns:
            ModelResults with all metrics computed
        """
        client = OpenAI(api_key=api_key, base_url=api_base)
        
        if scenarios == "all":
            scenario_ids = get_all_scenario_ids()
        else:
            scenario_ids = scenarios
        
        results = ModelResults(
            model_name=model_name,
            model_family=model_family or self._infer_family(model_name),
            model_params=model_params or self._infer_params(model_name),
        )
        
        total = len(scenario_ids) * len(seeds)
        done = 0
        
        self._print(f"\n{'='*64}")
        self._print(f"  POLARIS-Bench v4 — Multi-Agent LLM Coordination Benchmark")
        self._print(f"  Model: {model_name}")
        self._print(f"  API: {api_base}")
        self._print(f"  Scenarios: {len(scenario_ids)} | Seeds: {len(seeds)} | Total: {total} episodes")
        self._print(f"{'='*64}\n")
        
        for scenario_id in scenario_ids:
            scenario = SCENARIOS[scenario_id]
            use_neg = scenario.get("negotiation_enabled", True)
            
            self._print(f"\n{'─'*48}")
            self._print(f"  [{scenario['dimension'].upper()}] {scenario['name']}")
            self._print(f"  {scenario['difficulty'].upper()} | {scenario['max_steps']} steps | {scenario['num_ministers']} ministers")
            self._print(f"{'─'*48}")
            
            scenario_metrics = []
            
            for seed in seeds:
                done += 1
                self._print(f"  [{done}/{total}] seed={seed} ...", end="", flush=True)
                
                t0 = time.time()
                metrics = self._run_single_episode(
                    client=client,
                    model_name=model_name,
                    scenario=scenario,
                    seed=seed,
                    use_negotiation=use_neg,
                )
                elapsed = time.time() - t0
                metrics.wall_time_seconds = elapsed
                
                status = "SURVIVED" if not metrics.collapsed else "COLLAPSED"
                tom_str = f" ToM={metrics.tom_accuracy:.0%}" if metrics.tom_predictions > 0 else ""
                self._print(f" {status} score={metrics.score:.4f}{tom_str} ({elapsed:.1f}s)")
                
                # Failure detection
                failures = self.failure_detector.detect_all(
                    trajectory=[],  # will use internal trajectory
                    actions=metrics.action_sequence,
                )
                metrics.failure_modes = [f.mode for f in failures]
                
                scenario_metrics.append(metrics)
            
            results.scenario_results[scenario_id] = scenario_metrics
            
            # Scenario summary
            avg_score = statistics.mean(m.score for m in scenario_metrics)
            collapse_rate = sum(1 for m in scenario_metrics if m.collapsed) / len(scenario_metrics)
            self._print(f"  → Avg score: {avg_score:.4f} | Collapse rate: {collapse_rate:.0%}")
        
        # Compute composites
        results.compute_composites()
        
        # Compute CCR (need single-agent baseline)
        if include_baselines:
            single_results = self._run_single_agent_baseline(client, model_name, seeds)
            multi_results = [m for ms in results.scenario_results.values() for m in ms]
            results.ccr = compute_ccr_from_results(single_results, multi_results)
        
        # Save results
        self._save_results(results, output_dir)
        
        self._print(f"\n{'='*64}")
        self._print(f"  POLARIS-Bench COMPLETE")
        self._print(f"  Overall: {results.polaris_overall:.4f}")
        self._print(f"  CCR: {results.ccr:.4f}")
        self._print(f"  Coord: {results.polaris_coord:.4f} | ToM: {results.polaris_tom:.4f}")
        self._print(f"  Plan: {results.polaris_plan:.4f} | Adv: {results.polaris_adv:.4f}")
        self._print(f"  Scale: {results.polaris_scale:.4f}")
        self._print(f"{'='*64}\n")
        
        return results
    
    def _run_single_episode(
        self,
        client: OpenAI,
        model_name: str,
        scenario: Dict,
        seed: int,
        use_negotiation: bool,
    ) -> CoordinationMetrics:
        """Run a single episode and extract metrics."""
        
        # Map scenario to task config
        task_config = scenario_to_task_config(scenario)
        scenario_id = scenario["id"]
        max_steps = task_config["max_steps"]
        
        # Temporarily register the scenario as a task
        task_id = scenario_id
        TASK_CONFIGS[task_id] = task_config
        
        try:
            env = PolicyEnvironment()
            obs = env.reset(seed=seed, task_id=task_id)
            
            total_reward = 0.0
            step = 0
            actions_taken = []
            trajectory_meta = []
            
            while not obs.done:
                step += 1
                meta = obs.metadata
                trajectory_meta.append(meta)
                
                if use_negotiation:
                    obs_text = format_obs_negotiation(meta, step, max_steps)
                    action_data = call_llm(client, obs_text, model_name, True)
                else:
                    obs_text = format_obs_simple(meta, step, max_steps)
                    action_data = call_llm(client, obs_text, model_name, False)
                
                obs = env.step(action_data)
                total_reward += obs.reward
                actions_taken.append(action_data.get("action", "no_action"))
            
            # Final metadata
            trajectory_meta.append(obs.metadata)
            
            # Grade using closest grader
            try:
                grader_task = self._get_grader_task(scenario_id)
                score = grade_trajectory(grader_task, env.get_trajectory())
            except Exception:
                score = 0.0
            
            # Extract comprehensive metrics
            metrics = extract_metrics(
                trajectory=trajectory_meta,
                task_score=score,
                scenario_id=scenario_id,
                model_name=model_name,
                seed=seed,
                max_steps=max_steps,
                action_data_list=actions_taken,
            )
            metrics.total_reward = total_reward
            
            return metrics
            
        finally:
            # Clean up temp task config
            if task_id in TASK_CONFIGS and task_id not in (
                "environmental_recovery", "balanced_economy",
                "sustainable_governance", "sustainable_governance_extreme",
                "multi_agent_council", "negotiation_arena",
            ):
                del TASK_CONFIGS[task_id]
    
    def _run_single_agent_baseline(
        self,
        client: OpenAI,
        model_name: str,
        seeds: List[int],
    ) -> List[CoordinationMetrics]:
        """Run single-agent (no ministers) episodes for CCR computation."""
        self._print(f"\n  Running single-agent baseline for CCR...")
        results = []
        for seed in seeds[:3]:
            env = PolicyEnvironment()
            obs = env.reset(seed=seed, task_id="environmental_recovery")
            
            step = 0
            actions = []
            trajectory = []
            while not obs.done:
                step += 1
                meta = obs.metadata
                trajectory.append(meta)
                obs_text = format_obs_simple(meta, step, 50)
                action_data = call_llm(client, obs_text, model_name, False)
                obs = env.step(action_data)
                actions.append(action_data.get("action", "no_action"))
            
            score = grade_trajectory("environmental_recovery", env.get_trajectory())
            metrics = extract_metrics(
                trajectory=trajectory,
                task_score=score,
                scenario_id="single_agent_baseline",
                model_name=model_name,
                seed=seed,
                max_steps=50,
                action_data_list=actions,
            )
            results.append(metrics)
        
        avg = statistics.mean(m.score for m in results)
        self._print(f"  Single-agent baseline: {avg:.4f}")
        return results
    
    def _get_grader_task(self, scenario_id: str) -> str:
        """Map scenario to the closest available grader."""
        scenario = SCENARIOS.get(scenario_id, {})
        dim = scenario.get("dimension", "")
        num_min = scenario.get("num_ministers", 5)
        
        if num_min <= 1:
            return "environmental_recovery"
        if scenario.get("negotiation_enabled", True):
            return "negotiation_arena"
        return "sustainable_governance"
    
    def _save_results(self, results: ModelResults, output_dir: str):
        """Save results to disk."""
        os.makedirs(output_dir, exist_ok=True)
        
        safe_name = results.model_name.replace("/", "_").replace(":", "_")
        path = os.path.join(output_dir, f"{safe_name}_results.json")
        
        data = {
            "model": results.model_name,
            "family": results.model_family,
            "params": results.model_params,
            "polaris_overall": results.polaris_overall,
            "polaris_coord": results.polaris_coord,
            "polaris_tom": results.polaris_tom,
            "polaris_plan": results.polaris_plan,
            "polaris_adv": results.polaris_adv,
            "polaris_scale": results.polaris_scale,
            "ccr": results.ccr,
            "total_episodes": results.total_episodes,
            "total_collapses": results.total_collapses,
            "avg_tom_accuracy": results.avg_tom_accuracy,
            "leaderboard": results.to_leaderboard_row(),
            "scenarios": {},
        }
        
        for sid, metrics_list in results.scenario_results.items():
            data["scenarios"][sid] = [m.to_dict() for m in metrics_list]
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        
        self._print(f"  Results saved: {path}")
    
    def _infer_family(self, model_name: str) -> str:
        name = model_name.lower()
        if "llama" in name: return "llama"
        if "qwen" in name: return "qwen"
        if "mistral" in name or "mixtral" in name: return "mistral"
        if "gemma" in name: return "gemma"
        if "gpt" in name: return "gpt"
        if "claude" in name: return "claude"
        if "gemini" in name: return "gemini"
        return "unknown"
    
    def _infer_params(self, model_name: str) -> str:
        import re
        match = re.search(r'(\d+)[bB]', model_name)
        return f"{match.group(1)}B" if match else "unknown"
    
    def _print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)
