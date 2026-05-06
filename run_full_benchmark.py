#!/usr/bin/env python3
"""
POLARIS-Bench v4 — MASTER MULTI-MODEL BENCHMARK
=================================================

Evaluates EVERY accessible model through the full benchmark.
Uses local GPU (RTX 5080) for small models + free APIs for large ones.

Hardware: OMEN Max 15 | RTX 5080 16GB | Ultra 9 275HX | 32GB RAM

Models evaluated:
  LOCAL (your 5080):
    - Qwen2.5-3B-Instruct   (tiny, ~2GB VRAM)
    - Qwen2.5-7B-Instruct   (small, ~5GB VRAM)
    - Llama-3.2-3B-Instruct  (tiny)
    - Mistral-7B-Instruct    (small, ~5GB VRAM)

  GROQ API (FREE, fast):
    - Llama-3.3-70B          (large)
    - Llama-3.1-8B           (medium)
    - Mixtral-8x7B           (MoE)
    - Gemma2-9B              (medium)

  (Optional — set keys):
    - GPT-4o-mini            (OpenAI, ~$2)
    - Gemini-1.5-Flash       (Google, free tier)

Usage:
    # Set your Groq API key (free at console.groq.com)
    set GROQ_API_KEY=gsk_your_key_here

    # Run full benchmark (all accessible models)
    python run_full_benchmark.py

    # Quick test (1 scenario per dimension, 1 seed)
    python run_full_benchmark.py --quick

    # Only API models (skip local)
    python run_full_benchmark.py --api-only

    # Only local models (skip API)
    python run_full_benchmark.py --local-only
"""

import argparse
import json
import os
import sys
import time
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from polaris_bench.evaluator import PolarisEvaluator
from polaris_bench.scenarios import get_all_scenario_ids, DIMENSIONS
from polaris_bench.report import BenchmarkReport
from polaris_bench.stats import full_statistical_report


# ═══════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ═══════════════════════════════════════════════════════════════

GROQ_MODELS = [
    {
        "model_name": "llama-3.3-70b-versatile",
        "api_base": "https://api.groq.com/openai/v1",
        "family": "llama", "params": "70B",
    },
    {
        "model_name": "llama-3.1-8b-instant",
        "api_base": "https://api.groq.com/openai/v1",
        "family": "llama", "params": "8B",
    },
    {
        "model_name": "mixtral-8x7b-32768",
        "api_base": "https://api.groq.com/openai/v1",
        "family": "mistral", "params": "47B",
    },
    {
        "model_name": "gemma2-9b-it",
        "api_base": "https://api.groq.com/openai/v1",
        "family": "gemma", "params": "9B",
    },
]

# Local models via vLLM OpenAI-compatible server
# Start vLLM: python -m vllm.entrypoints.openai.api_server --model <model> --port 8000
LOCAL_MODELS = [
    {
        "model_name": "Qwen/Qwen2.5-3B-Instruct",
        "api_base": "http://localhost:8000/v1",
        "family": "qwen", "params": "3B",
    },
    {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "api_base": "http://localhost:8000/v1",
        "family": "qwen", "params": "7B",
    },
    {
        "model_name": "meta-llama/Llama-3.2-3B-Instruct",
        "api_base": "http://localhost:8000/v1",
        "family": "llama", "params": "3B",
    },
    {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "api_base": "http://localhost:8000/v1",
        "family": "mistral", "params": "7B",
    },
]

# Optional paid/free-tier API models
OPTIONAL_MODELS = [
    {
        "model_name": "gpt-4o-mini",
        "api_base": "https://api.openai.com/v1",
        "family": "gpt", "params": "unknown",
        "env_key": "OPENAI_API_KEY",
    },
    {
        "model_name": "gemini-1.5-flash",
        "api_base": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "family": "gemini", "params": "unknown",
        "env_key": "GEMINI_API_KEY",
    },
]


def get_api_key(model_config):
    """Get API key for a model."""
    env_key = model_config.get("env_key", "")
    if env_key:
        return os.getenv(env_key, "")
    # Default to GROQ for groq models
    if "groq" in model_config.get("api_base", ""):
        return os.getenv("GROQ_API_KEY", os.getenv("HF_TOKEN", ""))
    if "localhost" in model_config.get("api_base", ""):
        return "not-needed"
    return ""


def run_full_benchmark(args):
    """Run the complete multi-model benchmark."""
    
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine which models to run
    models_to_run = []
    
    if not args.local_only:
        for m in GROQ_MODELS:
            key = get_api_key(m)
            if key:
                models_to_run.append({**m, "api_key": key})
            else:
                print(f"  SKIP {m['model_name']} — no GROQ_API_KEY set")
    
    if not args.api_only:
        for m in LOCAL_MODELS:
            models_to_run.append({**m, "api_key": "not-needed"})
    
    if not args.local_only and not args.api_only:
        for m in OPTIONAL_MODELS:
            key = get_api_key(m)
            if key:
                models_to_run.append({**m, "api_key": key})
    
    if not models_to_run:
        print("ERROR: No models available. Set GROQ_API_KEY or start local vLLM server.")
        print("  Get free key: https://console.groq.com/keys")
        return
    
    # Determine scenarios
    if args.quick:
        scenarios = [sids[0] for sids in DIMENSIONS.values()]
        seeds = [42]
    elif args.dimension:
        scenarios = list(DIMENSIONS.get(args.dimension, []))
        seeds = [42, 123, 777]
    else:
        scenarios = "all"
        seeds = [42, 123, 777]
    
    print(f"\n{'='*64}")
    print(f"  POLARIS-Bench v4 — MULTI-MODEL BENCHMARK")
    print(f"  Models: {len(models_to_run)}")
    print(f"  Scenarios: {'ALL 20' if scenarios == 'all' else len(scenarios)}")
    print(f"  Seeds: {seeds}")
    print(f"  Output: {output_dir}")
    print(f"{'='*64}\n")
    
    evaluator = PolarisEvaluator(verbose=True)
    all_results = []
    
    for i, model_config in enumerate(models_to_run):
        model_name = model_config["model_name"]
        print(f"\n{'#'*64}")
        print(f"  MODEL {i+1}/{len(models_to_run)}: {model_name}")
        print(f"  Family: {model_config['family']} | Params: {model_config['params']}")
        print(f"  API: {model_config['api_base']}")
        print(f"{'#'*64}")
        
        try:
            results = evaluator.evaluate_model(
                model_name=model_name,
                api_base=model_config["api_base"],
                api_key=model_config["api_key"],
                model_family=model_config["family"],
                model_params=model_config["params"],
                scenarios=scenarios,
                seeds=seeds,
                include_baselines=(i == 0),  # only compute CCR baseline once
                output_dir=output_dir,
            )
            all_results.append(results)
            
        except Exception as e:
            print(f"  ERROR evaluating {model_name}: {e}")
            if "localhost" in model_config["api_base"]:
                print(f"  TIP: Start vLLM server first:")
                print(f"    python -m vllm.entrypoints.openai.api_server --model {model_name} --port 8000")
            continue
    
    if not all_results:
        print("No models completed evaluation.")
        return
    
    # Generate comparison report
    print(f"\n{'='*64}")
    print(f"  GENERATING COMPARISON REPORT...")
    print(f"{'='*64}")
    
    report = BenchmarkReport()
    for r in all_results:
        report.add_model(r)
    report.generate_all(os.path.join(output_dir, "report"))
    
    # Statistical analysis
    if len(all_results) >= 2:
        stats = full_statistical_report(all_results)
        stats_path = os.path.join(output_dir, "report", "statistical_analysis.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2, default=str)
        
        print(f"\n  KEY FINDINGS:")
        for finding in stats.get("key_findings", []):
            print(f"    -> {finding}")
    
    print(f"\n{'='*64}")
    print(f"  BENCHMARK COMPLETE")
    print(f"  Results: {output_dir}/")
    print(f"  Report: {output_dir}/report/")
    print(f"  Models evaluated: {len(all_results)}")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="POLARIS-Bench v4 Multi-Model Benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick: 1 scenario/dim, 1 seed")
    parser.add_argument("--api-only", action="store_true", help="Skip local models")
    parser.add_argument("--local-only", action="store_true", help="Skip API models")
    parser.add_argument("--dimension", type=str, help="Run single dimension only")
    parser.add_argument("--output", type=str, default="outputs/polaris_bench", help="Output dir")
    args = parser.parse_args()
    
    run_full_benchmark(args)
