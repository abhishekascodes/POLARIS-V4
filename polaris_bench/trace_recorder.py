"""
POLARIS-Bench v4 — Episode Trace Recorder
===========================================

Records complete episode traces for the open dataset (POLARIS-Traces).
Each trace includes:
  - Full action sequence with model reasoning
  - Complete state trajectory
  - Negotiation transcripts (minister proposals, votes, vetoes)
  - Failure mode annotations
  - Model metadata

Traces are saved as JSONL for efficient streaming and HuggingFace Datasets.
"""

from __future__ import annotations
import json
import os
import time
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional


@dataclass
class StepTrace:
    """A single step in an episode trace."""
    step: int
    state: Dict[str, float]
    action: str
    action_data: Dict[str, Any]      # full model output (reasoning, coalition, etc.)
    reward: float
    cumulative_reward: float
    
    # Negotiation data
    minister_proposals: List[Dict[str, Any]] = field(default_factory=list)
    vote_result: Dict[str, Any] = field(default_factory=dict)
    vetoes: List[str] = field(default_factory=list)
    coalition_formed: bool = False
    coalition_members: List[str] = field(default_factory=list)
    
    # Events & briefings
    active_events: List[str] = field(default_factory=list)
    new_briefing: str = ""
    
    # ToM
    veto_prediction: List[str] = field(default_factory=list)
    veto_prediction_correct: Optional[bool] = None
    tom_reward: float = 0.0


@dataclass 
class EpisodeTrace:
    """Complete trace of a single episode."""
    
    # Identity
    trace_id: str = ""
    model_name: str = ""
    model_family: str = ""
    model_params: str = ""
    scenario_id: str = ""
    seed: int = 0
    timestamp: str = ""
    
    # Config
    max_steps: int = 0
    num_ministers: int = 0
    chaos_level: float = 0.0
    
    # Results
    score: float = 0.0
    total_reward: float = 0.0
    steps_completed: int = 0
    collapsed: bool = True
    survival_rate: float = 0.0
    
    # ToM aggregate
    tom_predictions: int = 0
    tom_correct: int = 0
    tom_accuracy: float = 0.0
    
    # Negotiation aggregate
    coalition_count: int = 0
    veto_count: int = 0
    betrayal_count: int = 0
    
    # Failure modes detected
    failure_modes: List[str] = field(default_factory=list)
    
    # Full step-by-step trace
    steps: List[Dict[str, Any]] = field(default_factory=list)
    
    # Wall time
    wall_time_seconds: float = 0.0
    total_tokens: int = 0
    
    def add_step(self, step_trace: StepTrace):
        """Add a step to the trace."""
        self.steps.append(asdict(step_trace))
        self.steps_completed = len(self.steps)
    
    def finalize(self):
        """Compute aggregate metrics after episode ends."""
        if self.steps:
            self.survival_rate = self.steps_completed / max(self.max_steps, 1)
            self.tom_accuracy = (
                self.tom_correct / max(self.tom_predictions, 1)
                if self.tom_predictions > 0 else 0.0
            )
    
    def to_jsonl_record(self) -> str:
        """Serialize to a single JSONL line (for dataset)."""
        record = {
            "trace_id": self.trace_id,
            "model_name": self.model_name,
            "model_family": self.model_family,
            "model_params": self.model_params,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "timestamp": self.timestamp,
            "max_steps": self.max_steps,
            "num_ministers": self.num_ministers,
            "chaos_level": self.chaos_level,
            "score": self.score,
            "total_reward": self.total_reward,
            "steps_completed": self.steps_completed,
            "collapsed": self.collapsed,
            "survival_rate": self.survival_rate,
            "tom_predictions": self.tom_predictions,
            "tom_correct": self.tom_correct,
            "tom_accuracy": self.tom_accuracy,
            "coalition_count": self.coalition_count,
            "veto_count": self.veto_count,
            "betrayal_count": self.betrayal_count,
            "failure_modes": self.failure_modes,
            "wall_time_seconds": self.wall_time_seconds,
            "total_tokens": self.total_tokens,
            "num_steps": len(self.steps),
            # Include full steps for detailed analysis
            "steps": self.steps,
        }
        return json.dumps(record, default=str)


class TraceRecorder:
    """
    Records and manages episode traces for the open dataset.
    
    Usage:
        recorder = TraceRecorder("outputs/traces")
        trace = recorder.start_episode("gpt-4o", "coord_crisis_response", seed=42)
        # ... run episode, calling trace.add_step() each step ...
        recorder.finish_episode(trace, score=0.45, collapsed=False)
    """
    
    def __init__(self, output_dir: str = "outputs/polaris_traces"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.episode_count = 0
    
    def start_episode(
        self,
        model_name: str,
        scenario_id: str,
        seed: int,
        max_steps: int = 200,
        num_ministers: int = 5,
        chaos_level: float = 0.6,
        model_family: str = "",
        model_params: str = "",
    ) -> EpisodeTrace:
        """Start recording a new episode."""
        
        # Generate unique trace ID
        raw = f"{model_name}:{scenario_id}:{seed}:{time.time()}"
        trace_id = hashlib.md5(raw.encode()).hexdigest()[:12]
        
        trace = EpisodeTrace(
            trace_id=trace_id,
            model_name=model_name,
            model_family=model_family,
            model_params=model_params,
            scenario_id=scenario_id,
            seed=seed,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            max_steps=max_steps,
            num_ministers=num_ministers,
            chaos_level=chaos_level,
        )
        
        return trace
    
    def record_step(
        self,
        trace: EpisodeTrace,
        step: int,
        state: Dict[str, float],
        action_data: Dict[str, Any],
        reward: float,
        cumulative_reward: float,
        meta: Dict[str, Any],
    ):
        """Record a single step."""
        
        # Extract negotiation data from metadata
        outcome = meta.get("negotiation_outcome", {})
        council = meta.get("council", {})
        
        step_trace = StepTrace(
            step=step,
            state={k: round(v, 4) if isinstance(v, float) else v 
                   for k, v in state.items() if isinstance(v, (int, float))},
            action=action_data.get("action", "no_action"),
            action_data={k: v for k, v in action_data.items() if k != "_tokens"},
            reward=round(reward, 6),
            cumulative_reward=round(cumulative_reward, 4),
            minister_proposals=meta.get("minister_proposals", []),
            vote_result=outcome,
            vetoes=council.get("vetoes", []),
            coalition_formed=outcome.get("coalition_formed", False),
            coalition_members=outcome.get("supporters", []),
            active_events=[str(e) for e in meta.get("active_events", [])],
            new_briefing=meta.get("new_briefing", ""),
            veto_prediction=action_data.get("veto_prediction", []),
            veto_prediction_correct=outcome.get("veto_prediction_correct"),
            tom_reward=outcome.get("tom_reward", 0),
        )
        
        # Update aggregate counters
        if step_trace.coalition_formed:
            trace.coalition_count += 1
        if step_trace.vetoes:
            trace.veto_count += len(step_trace.vetoes)
        if outcome.get("betrayal_occurred"):
            trace.betrayal_count += 1
        if step_trace.veto_prediction_correct is not None:
            trace.tom_predictions += 1
            if step_trace.veto_prediction_correct:
                trace.tom_correct += 1
        
        # Track tokens
        trace.total_tokens += action_data.get("_tokens", 0)
        
        trace.add_step(step_trace)
    
    def finish_episode(
        self,
        trace: EpisodeTrace,
        score: float,
        collapsed: bool,
        failure_modes: List[str] = None,
        wall_time: float = 0.0,
    ):
        """Finalize and save an episode trace."""
        trace.score = round(score, 6)
        trace.collapsed = collapsed
        trace.total_reward = round(
            sum(s.get("reward", 0) for s in trace.steps), 4
        )
        trace.failure_modes = failure_modes or []
        trace.wall_time_seconds = round(wall_time, 2)
        trace.finalize()
        
        # Append to JSONL file (one file per model)
        safe_model = trace.model_name.replace("/", "_").replace(":", "_")
        jsonl_path = os.path.join(self.output_dir, f"{safe_model}_traces.jsonl")
        
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(trace.to_jsonl_record() + "\n")
        
        self.episode_count += 1
        
        return jsonl_path
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get statistics about all recorded traces."""
        stats = {
            "total_episodes": 0,
            "models": {},
            "scenarios": {},
            "total_steps": 0,
        }
        
        for fname in os.listdir(self.output_dir):
            if not fname.endswith("_traces.jsonl"):
                continue
            
            path = os.path.join(self.output_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        stats["total_episodes"] += 1
                        stats["total_steps"] += record.get("num_steps", 0)
                        
                        model = record.get("model_name", "unknown")
                        if model not in stats["models"]:
                            stats["models"][model] = {"episodes": 0, "avg_score": 0, "scores": []}
                        stats["models"][model]["episodes"] += 1
                        stats["models"][model]["scores"].append(record.get("score", 0))
                        
                        scenario = record.get("scenario_id", "unknown")
                        if scenario not in stats["scenarios"]:
                            stats["scenarios"][scenario] = 0
                        stats["scenarios"][scenario] += 1
                    except json.JSONDecodeError:
                        continue
        
        # Compute averages
        for model_data in stats["models"].values():
            scores = model_data.pop("scores", [])
            model_data["avg_score"] = round(sum(scores) / max(len(scores), 1), 4) if scores else 0
        
        return stats
