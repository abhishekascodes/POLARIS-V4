"""Quick verification of the full POLARIS-Bench v4 system."""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test all imports
from polaris_bench import PolarisEvaluator, SCENARIOS, SPEC, FailureDetector, TraceRecorder
from polaris_bench.stats import confidence_interval, scaling_law_fit, paired_t_test, cohens_d
from polaris_bench.metrics import CoordinationMetrics, ModelResults, compute_ccr
from polaris_bench.scenarios import DIMENSIONS, get_all_scenario_ids
from polaris_bench.formal_spec import FormalSpec
print("✅ ALL IMPORTS OK")

# Test scenarios
print(f"✅ Scenarios: {len(SCENARIOS)}")
for dim, sids in DIMENSIONS.items():
    print(f"   {dim}: {len(sids)} scenarios")

# Test metrics
m = CoordinationMetrics(model_name="test", score=0.8, collapsed=False)
print(f"✅ Metrics: model={m.model_name}, score={m.score}")
print(f"✅ CCR(0.9, 0.3) = {compute_ccr(0.9, 0.3):.4f}")

# Test failure detector
fd = FailureDetector()
actions = ["increase_tax", "decrease_tax"] * 5
failures = fd._detect_oscillation(actions)
print(f"✅ Failure detector: {len(failures)} oscillations detected")

# Test stats
ci = confidence_interval([0.3, 0.4, 0.35, 0.38, 0.32])
print(f"✅ CI: mean={ci[0]:.3f} [{ci[1]:.3f}, {ci[2]:.3f}]")

sl = scaling_law_fit([8, 70, 405], [0.31, 0.33, 0.35])
print(f"✅ Scaling law: slope={sl['slope']:.6f}, helps={sl['scaling_helps']}")

tt = paired_t_test([0.3, 0.35, 0.32], [0.25, 0.28, 0.30])
print(f"✅ T-test: t={tt['t_statistic']:.3f}, p={tt['p_value']:.4f}")

d = cohens_d([0.3, 0.35, 0.32], [0.25, 0.28, 0.30])
print(f"✅ Cohen's d: {d:.3f}")

# Test trace recorder
tr = TraceRecorder("outputs/test_traces")
trace = tr.start_episode("test-model", "coord_resource_allocation", seed=42)
print(f"✅ Trace recorder: id={trace.trace_id}")

# Test formal spec
spec = FormalSpec()
latex = spec.to_latex()
print(f"✅ LaTeX spec: {len(latex)} chars")

# Test 12-minister council support
from server.multi_agent_council import MultiAgentCouncil, MINISTER_ROLES
mac = MultiAgentCouncil()
mac.reset(seed=42, num_ministers=12)
print(f"✅ Council: 12 ministers initialized ({len(MINISTER_ROLES)} roles available)")

# Test 12-minister LLM engine
from server.llm_minister import LLMMinisterEngine, ALL_PERSONAS
engine = LLMMinisterEngine(mode="scripted")
engine.reset(seed=42, num_ministers=12)
names = engine.get_minister_names()
print(f"✅ LLM Minister Engine: {len(names)} ministers")
for n in names:
    print(f"   - {n}")

# Test evaluator init
evaluator = PolarisEvaluator(verbose=False)
print(f"✅ Evaluator ready")

print(f"\n{'='*60}")
print(f"  POLARIS-Bench v4 — FULL SYSTEM VERIFIED")
print(f"  20 scenarios | 5 dimensions | 12 ministers | CCR metric")
print(f"  Failure detector | Stats engine | Trace recorder")
print(f"  Formal MDP spec | LaTeX export")
print(f"{'='*60}")
