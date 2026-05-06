"""
POLARIS-Bench v4 — The Multi-Agent LLM Coordination Benchmark
==============================================================

The first comprehensive benchmark for evaluating LLM coordination,
theory-of-mind, and multi-agent negotiation under pressure.

20 scenarios across 5 dimensions:
  - Coordination (can agents work together?)
  - Theory-of-Mind (can agents model other agents?)
  - Long-Horizon Planning (can agents think ahead?)
  - Adversarial Robustness (can agents handle hostile agents?)
  - Scaling (how does agent count affect performance?)

Core metric: Coordination Collapse Ratio (CCR)
  CCR = Score_multi / Score_single
  CCR < 0.5 → significant coordination failure

Usage:
    from polaris_bench import PolarisEvaluator
    
    evaluator = PolarisEvaluator()
    results = evaluator.evaluate_model(
        model_name="llama-3.3-70b-versatile",
        api_base="https://api.groq.com/openai/v1",
        api_key="your-key"
    )
"""

__version__ = "4.0.0"
__author__ = "Abhishek A S"
__benchmark__ = "POLARIS-Bench"

from .evaluator import PolarisEvaluator
from .metrics import CoordinationMetrics, ModelResults, compute_ccr
from .failure_detector import FailureDetector
from .scenarios import SCENARIOS, get_scenario, get_all_scenario_ids, DIMENSIONS
from .report import BenchmarkReport
from .trace_recorder import TraceRecorder
from .stats import confidence_interval, paired_t_test, cohens_d, scaling_law_fit
from .formal_spec import SPEC, FormalSpec
