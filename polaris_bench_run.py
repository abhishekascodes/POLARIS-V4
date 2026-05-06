#!/usr/bin/env python3
"""
POLARIS-Bench v4 — CLI Runner
===============================

Run the full 20-scenario benchmark for any LLM.

Usage:
    # Full benchmark with Groq (free)
    set API_BASE_URL=https://api.groq.com/openai/v1
    set API_KEY=gsk_...
    python polaris_bench_run.py --model llama-3.3-70b-versatile

    # Quick test (3 scenarios only)
    python polaris_bench_run.py --model llama-3.3-70b-versatile --quick

    # Specific dimension
    python polaris_bench_run.py --model llama-3.3-70b-versatile --dimension coordination

    # Compare multiple models (run each, then generate comparison)
    python polaris_bench_run.py --compare outputs/polaris_bench/

    # Custom API (OpenAI, Together, etc.)
    python polaris_bench_run.py --model gpt-4o --api-base https://api.openai.com/v1 --api-key sk-...
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from polaris_bench.evaluator import PolarisEvaluator
from polaris_bench.scenarios import get_all_scenario_ids, get_scenarios_by_dimension, DIMENSIONS
from polaris_bench.metrics import ModelResults
from polaris_bench.report import BenchmarkReport


def main():
    parser = argparse.ArgumentParser(
        description="POLARIS-Bench v4 — Multi-Agent LLM Coordination Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python polaris_bench_run.py --model llama-3.3-70b-versatile
  python polaris_bench_run.py --model gpt-4o --api-base https://api.openai.com/v1
  python polaris_bench_run.py --model llama-3.3-70b-versatile --quick
  python polaris_bench_run.py --compare outputs/polaris_bench/
        """
    )
    
    parser.add_argument("--model", type=str, help="Model name/ID")
    parser.add_argument("--api-base", type=str, default=os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1"),
                       help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", type=str, default=os.getenv("API_KEY", os.getenv("HF_TOKEN", "")),
                       help="API key")
    parser.add_argument("--family", type=str, default="", help="Model family (llama, qwen, gpt, etc.)")
    parser.add_argument("--params", type=str, default="", help="Model params (8B, 70B, etc.)")
    
    parser.add_argument("--dimension", type=str, choices=list(DIMENSIONS.keys()),
                       help="Run only scenarios from this dimension")
    parser.add_argument("--scenario", type=str, help="Run a single specific scenario")
    parser.add_argument("--quick", action="store_true",
                       help="Quick mode: 1 scenario per dimension, 1 seed")
    parser.add_argument("--seeds", type=str, default="42,123,777",
                       help="Comma-separated seeds (default: 42,123,777)")
    
    parser.add_argument("--output", type=str, default="outputs/polaris_bench",
                       help="Output directory")
    parser.add_argument("--no-baselines", action="store_true",
                       help="Skip single-agent baseline (no CCR)")
    
    parser.add_argument("--compare", type=str,
                       help="Compare all results in a directory")
    parser.add_argument("--report", type=str,
                       help="Generate report from existing results")
    
    args = parser.parse_args()
    
    # ─── Compare mode ─────────────────────────────────────────
    if args.compare:
        compare_results(args.compare)
        return
    
    # ─── Report mode ──────────────────────────────────────────
    if args.report:
        generate_report(args.report)
        return
    
    # ─── Evaluate mode ────────────────────────────────────────
    if not args.model:
        parser.error("--model is required for evaluation")
    
    seeds = [int(s) for s in args.seeds.split(",")]
    
    # Determine scenarios
    if args.scenario:
        scenarios = [args.scenario]
    elif args.dimension:
        scenarios = get_scenarios_by_dimension(args.dimension)
    elif args.quick:
        # One scenario per dimension
        scenarios = []
        for dim, sids in DIMENSIONS.items():
            scenarios.append(sids[0])
        seeds = [42]  # single seed for quick mode
    else:
        scenarios = "all"
    
    evaluator = PolarisEvaluator(verbose=True)
    
    results = evaluator.evaluate_model(
        model_name=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        model_family=args.family,
        model_params=args.params,
        scenarios=scenarios,
        seeds=seeds,
        include_baselines=not args.no_baselines,
        output_dir=args.output,
    )
    
    # Generate report
    report = BenchmarkReport(results)
    report.generate_all(os.path.join(args.output, "report"))
    
    print(f"\n✅ POLARIS-Bench complete for {args.model}")
    print(f"   Overall Score: {results.polaris_overall:.4f}")
    print(f"   CCR: {results.ccr:.4f}")
    print(f"   Results: {args.output}/")


def compare_results(results_dir: str):
    """Compare all model results in a directory."""
    report = BenchmarkReport()
    
    for fname in os.listdir(results_dir):
        if fname.endswith("_results.json"):
            path = os.path.join(results_dir, fname)
            with open(path) as f:
                data = json.load(f)
            
            results = ModelResults(
                model_name=data["model"],
                model_family=data.get("family", ""),
                model_params=data.get("params", ""),
            )
            results.polaris_overall = data.get("polaris_overall", 0)
            results.polaris_coord = data.get("polaris_coord", 0)
            results.polaris_tom = data.get("polaris_tom", 0)
            results.polaris_plan = data.get("polaris_plan", 0)
            results.polaris_adv = data.get("polaris_adv", 0)
            results.polaris_scale = data.get("polaris_scale", 0)
            results.ccr = data.get("ccr", 0)
            results.total_episodes = data.get("total_episodes", 0)
            results.total_collapses = data.get("total_collapses", 0)
            results.avg_tom_accuracy = data.get("avg_tom_accuracy", 0)
            
            report.add_model(results)
    
    if not report.all_model_results:
        print(f"No results found in {results_dir}")
        return
    
    report.generate_all(os.path.join(results_dir, "report"))
    print(f"\n✅ Comparison report generated for {len(report.all_model_results)} models")


def generate_report(results_path: str):
    """Generate report from a single results file."""
    with open(results_path) as f:
        data = json.load(f)
    
    results = ModelResults(model_name=data["model"])
    results.polaris_overall = data.get("polaris_overall", 0)
    results.polaris_coord = data.get("polaris_coord", 0)
    results.polaris_tom = data.get("polaris_tom", 0)
    results.polaris_plan = data.get("polaris_plan", 0)
    results.polaris_adv = data.get("polaris_adv", 0)
    results.polaris_scale = data.get("polaris_scale", 0)
    results.ccr = data.get("ccr", 0)
    
    report = BenchmarkReport(results)
    output_dir = os.path.dirname(results_path)
    report.generate_all(os.path.join(output_dir, "report"))


if __name__ == "__main__":
    main()
