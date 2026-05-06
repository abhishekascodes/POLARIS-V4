"""
POLARIS-Bench v4 — Metrics Engine
==================================

Defines the core metrics that make POLARIS a real benchmark:
  - Coordination Collapse Ratio (CCR) — THE signature metric
  - Theory-of-Mind accuracy
  - Coalition efficiency
  - Governance stability
  - Composite POLARIS Score

Each metric is independently computed and normalized to [0, 1].
"""

from __future__ import annotations
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class CoordinationMetrics:
    """Complete metrics for a single evaluation run."""
    
    # Identity
    model_name: str = ""
    scenario_id: str = ""
    seed: int = 0
    
    # Core performance
    score: float = 0.0              # task grader score [0,1]
    total_reward: float = 0.0       # cumulative reward
    steps_survived: int = 0         # how many steps before done
    max_steps: int = 0              # maximum possible steps
    collapsed: bool = True          # did the system collapse?
    survival_rate: float = 0.0      # steps_survived / max_steps
    
    # Coordination metrics
    coalition_count: int = 0        # total coalitions formed
    coalition_rate: float = 0.0     # coalitions / steps
    veto_count: int = 0             # total vetoes received
    veto_rate: float = 0.0          # vetoes / steps
    betrayal_count: int = 0         # coalition betrayals
    approval_rate: float = 0.0      # policies approved / total
    
    # Theory-of-Mind
    tom_predictions: int = 0        # total ToM predictions made
    tom_correct: int = 0            # correct predictions
    tom_accuracy: float = 0.0       # tom_correct / tom_predictions
    tom_reward_sum: float = 0.0     # cumulative ToM reward
    
    # Governance stability
    gdp_mean: float = 0.0
    gdp_std: float = 0.0
    pollution_mean: float = 0.0
    pollution_std: float = 0.0
    satisfaction_mean: float = 0.0
    satisfaction_std: float = 0.0
    volatility: float = 0.0         # average normalized std across metrics
    
    # Trust dynamics
    trust_initial: float = 0.6
    trust_final: float = 0.0
    trust_min: float = 0.0
    trust_trajectory: List[float] = field(default_factory=list)
    
    # Briefing compliance
    briefings_total: int = 0
    briefings_resolved: int = 0
    briefing_compliance: float = 0.0
    
    # Action diversity
    unique_actions: int = 0
    total_actions: int = 0
    action_diversity: float = 0.0   # unique / total possible
    oscillation_count: int = 0      # action flip-flops
    
    # Timing
    wall_time_seconds: float = 0.0
    tokens_used: int = 0
    api_calls: int = 0
    
    # Raw data
    action_sequence: List[str] = field(default_factory=list)
    reward_sequence: List[float] = field(default_factory=list)
    failure_modes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict (excludes large sequences for storage)."""
        d = {}
        for k, v in self.__dict__.items():
            if k in ("action_sequence", "reward_sequence", "trust_trajectory"):
                d[k + "_len"] = len(v)
            elif k == "failure_modes":
                d[k] = v
            else:
                d[k] = v
        return d


@dataclass
class ModelResults:
    """Aggregated results for a single model across all scenarios."""
    
    model_name: str = ""
    model_family: str = ""     # llama, qwen, gpt, claude, etc.
    model_params: str = ""     # "8B", "70B", "unknown"
    
    # Per-scenario results
    scenario_results: Dict[str, List[CoordinationMetrics]] = field(default_factory=dict)
    
    # Composite scores (computed)
    polaris_coord: float = 0.0     # coordination dimension score
    polaris_tom: float = 0.0       # theory-of-mind dimension score
    polaris_plan: float = 0.0      # long-horizon planning score
    polaris_adv: float = 0.0       # adversarial robustness score
    polaris_scale: float = 0.0     # scaling dimension score
    polaris_overall: float = 0.0   # weighted composite
    
    # THE metric
    ccr: float = 0.0               # Coordination Collapse Ratio
    
    # Aggregate stats
    total_episodes: int = 0
    total_collapses: int = 0
    avg_survival_rate: float = 0.0
    avg_tom_accuracy: float = 0.0
    avg_coalition_rate: float = 0.0
    
    def compute_composites(self):
        """Compute composite scores from scenario results."""
        from .scenarios import DIMENSIONS
        
        dim_scores = {}
        for dim, scenario_ids in DIMENSIONS.items():
            scores = []
            for sid in scenario_ids:
                if sid in self.scenario_results:
                    for m in self.scenario_results[sid]:
                        scores.append(m.score)
            dim_scores[dim] = statistics.mean(scores) if scores else 0.0
        
        self.polaris_coord = dim_scores.get("coordination", 0.0)
        self.polaris_tom = dim_scores.get("theory_of_mind", 0.0)
        self.polaris_plan = dim_scores.get("long_horizon", 0.0)
        self.polaris_adv = dim_scores.get("adversarial", 0.0)
        self.polaris_scale = dim_scores.get("scaling", 0.0)
        
        # Weighted composite
        self.polaris_overall = (
            0.25 * self.polaris_coord +
            0.25 * self.polaris_tom +
            0.20 * self.polaris_plan +
            0.15 * self.polaris_adv +
            0.15 * self.polaris_scale
        )
        
        # Aggregate stats
        all_metrics = [m for results in self.scenario_results.values() for m in results]
        if all_metrics:
            self.total_episodes = len(all_metrics)
            self.total_collapses = sum(1 for m in all_metrics if m.collapsed)
            self.avg_survival_rate = statistics.mean(m.survival_rate for m in all_metrics)
            
            tom_metrics = [m for m in all_metrics if m.tom_predictions > 0]
            self.avg_tom_accuracy = statistics.mean(m.tom_accuracy for m in tom_metrics) if tom_metrics else 0.0
            
            coal_metrics = [m for m in all_metrics if m.steps_survived > 0]
            self.avg_coalition_rate = statistics.mean(m.coalition_rate for m in coal_metrics) if coal_metrics else 0.0
    
    def to_leaderboard_row(self) -> Dict[str, Any]:
        """Generate a single row for the leaderboard."""
        return {
            "model": self.model_name,
            "family": self.model_family,
            "params": self.model_params,
            "coord": round(self.polaris_coord, 4),
            "tom": round(self.polaris_tom, 4),
            "plan": round(self.polaris_plan, 4),
            "adv": round(self.polaris_adv, 4),
            "scale": round(self.polaris_scale, 4),
            "overall": round(self.polaris_overall, 4),
            "ccr": round(self.ccr, 4),
            "episodes": self.total_episodes,
            "collapse_rate": round(self.total_collapses / max(self.total_episodes, 1), 4),
            "avg_tom": round(self.avg_tom_accuracy, 4),
        }


# ═══════════════════════════════════════════════════════════════
# COORDINATION COLLAPSE RATIO (CCR)
# ═══════════════════════════════════════════════════════════════

def compute_ccr(
    single_agent_score: float,
    multi_agent_score: float,
) -> float:
    """
    Compute the Coordination Collapse Ratio (CCR).
    
    CCR = multi_agent_score / single_agent_score
    
    CCR = 1.0 → perfect coordination retention
    CCR < 0.5 → significant coordination failure
    CCR < 0.3 → catastrophic coordination collapse
    
    This is THE signature metric of POLARIS-Bench.
    """
    if single_agent_score <= 0:
        return 0.0
    return min(1.0, multi_agent_score / single_agent_score)


def compute_ccr_from_results(
    single_results: List[CoordinationMetrics],
    multi_results: List[CoordinationMetrics],
) -> float:
    """Compute CCR from lists of single-agent and multi-agent results."""
    if not single_results or not multi_results:
        return 0.0
    single_avg = statistics.mean(m.score for m in single_results)
    multi_avg = statistics.mean(m.score for m in multi_results)
    return compute_ccr(single_avg, multi_avg)


# ═══════════════════════════════════════════════════════════════
# METRIC EXTRACTION FROM EPISODE DATA
# ═══════════════════════════════════════════════════════════════

def extract_metrics(
    trajectory: List[Dict],
    task_score: float,
    scenario_id: str,
    model_name: str,
    seed: int,
    max_steps: int,
    wall_time: float = 0.0,
    action_data_list: Optional[List[Dict]] = None,
) -> CoordinationMetrics:
    """
    Extract comprehensive metrics from an episode trajectory.
    
    Args:
        trajectory: List of observation metadata dicts (one per step)
        task_score: Score from the task grader [0,1]
        scenario_id: Which scenario was run
        model_name: Name of the model being evaluated
        seed: Random seed used
        max_steps: Maximum steps for the scenario
        wall_time: Wall clock time in seconds
        action_data_list: List of action dicts taken by the agent
    """
    m = CoordinationMetrics()
    m.model_name = model_name
    m.scenario_id = scenario_id
    m.seed = seed
    m.score = task_score
    m.max_steps = max_steps
    m.steps_survived = len(trajectory)
    m.survival_rate = m.steps_survived / max(max_steps, 1)
    m.wall_time_seconds = wall_time
    
    if not trajectory:
        return m
    
    # Collapse detection
    final = trajectory[-1]
    m.collapsed = final.get("collapsed", False)
    if not m.collapsed:
        # Check collapse conditions
        m.collapsed = (
            final.get("gdp_index", 100) < 15 or
            final.get("pollution_index", 100) > 290 or
            final.get("public_satisfaction", 50) < 5
        )
    
    # Rewards
    m.total_reward = sum(t.get("reward", 0) for t in trajectory)
    m.reward_sequence = [t.get("reward", 0) for t in trajectory]
    
    # Governance metrics
    gdps = [t.get("gdp_index", 100) for t in trajectory]
    polls = [t.get("pollution_index", 100) for t in trajectory]
    sats = [t.get("public_satisfaction", 50) for t in trajectory]
    
    m.gdp_mean = statistics.mean(gdps)
    m.gdp_std = statistics.stdev(gdps) if len(gdps) >= 2 else 0
    m.pollution_mean = statistics.mean(polls)
    m.pollution_std = statistics.stdev(polls) if len(polls) >= 2 else 0
    m.satisfaction_mean = statistics.mean(sats)
    m.satisfaction_std = statistics.stdev(sats) if len(sats) >= 2 else 0
    
    # Volatility (normalized)
    vol_parts = []
    if m.gdp_std > 0: vol_parts.append(m.gdp_std / 200)
    if m.pollution_std > 0: vol_parts.append(m.pollution_std / 300)
    if m.satisfaction_std > 0: vol_parts.append(m.satisfaction_std / 100)
    m.volatility = statistics.mean(vol_parts) if vol_parts else 0.0
    
    # Trust dynamics
    trusts = [t.get("institutional_trust", t.get("council", {}).get("institutional_trust", 0.6)) for t in trajectory]
    if trusts:
        m.trust_initial = trusts[0]
        m.trust_final = trusts[-1]
        m.trust_min = min(trusts)
        m.trust_trajectory = trusts
    
    # Negotiation metrics
    for t in trajectory:
        outcome = t.get("negotiation_outcome", {})
        if outcome.get("coalition_formed"):
            m.coalition_count += 1
        if outcome.get("vetoed"):
            m.veto_count += 1
        if outcome.get("betrayal_occurred"):
            m.betrayal_count += 1
        if "veto_prediction_correct" in outcome:
            m.tom_predictions += 1
            if outcome["veto_prediction_correct"]:
                m.tom_correct += 1
            m.tom_reward_sum += outcome.get("tom_reward", 0)
        if outcome.get("approved"):
            m.approval_rate += 1
    
    steps = max(m.steps_survived, 1)
    m.coalition_rate = m.coalition_count / steps
    m.veto_rate = m.veto_count / steps
    m.tom_accuracy = m.tom_correct / max(m.tom_predictions, 1) if m.tom_predictions > 0 else 0.0
    m.approval_rate = m.approval_rate / steps
    
    # Briefing compliance
    bs = final.get("briefing_stats", {})
    m.briefings_total = bs.get("total_briefings", 0)
    m.briefings_resolved = bs.get("resolved", 0)
    m.briefing_compliance = m.briefings_resolved / max(m.briefings_total, 1) if m.briefings_total > 0 else 0.5
    
    # Action analysis
    if action_data_list:
        actions = [a.get("action", "no_action") if isinstance(a, dict) else a for a in action_data_list]
        m.action_sequence = actions
        m.unique_actions = len(set(actions))
        m.total_actions = len(actions)
        m.action_diversity = m.unique_actions / 19  # 19 possible actions
        
        # Oscillation detection
        for i in range(2, len(actions)):
            if actions[i] == actions[i-2] and actions[i] != actions[i-1]:
                m.oscillation_count += 1
    
    return m
