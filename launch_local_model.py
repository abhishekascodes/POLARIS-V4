#!/usr/bin/env python3
"""
POLARIS-Bench v4 — Local Model Server
=======================================

Launches models on your RTX 5080 via vLLM for benchmark evaluation.
Each model runs as an OpenAI-compatible API server on localhost.

Hardware: RTX 5080 (16GB VRAM)
  - 3B models: ~2GB VRAM, fast
  - 7B models: ~5GB VRAM (4-bit quantized), moderate  
  - 14B models: ~8GB VRAM (4-bit quantized), slower

Usage:
    # Install vLLM first (one time)
    pip install vllm

    # Launch a model
    python launch_local_model.py --model qwen-3b
    python launch_local_model.py --model qwen-7b
    python launch_local_model.py --model llama-3b
    python launch_local_model.py --model mistral-7b

    # Then in another terminal:
    python run_full_benchmark.py --local-only --quick
"""

import argparse
import subprocess
import sys
import os

MODELS = {
    "qwen-3b": {
        "hf_id": "Qwen/Qwen2.5-3B-Instruct",
        "vram": "~2GB",
        "quantization": None,
    },
    "qwen-7b": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "vram": "~5GB",
        "quantization": "awq",
    },
    "llama-3b": {
        "hf_id": "meta-llama/Llama-3.2-3B-Instruct",
        "vram": "~2GB",
        "quantization": None,
    },
    "mistral-7b": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "vram": "~5GB",
        "quantization": "awq",
    },
    "gemma-2b": {
        "hf_id": "google/gemma-2-2b-it",
        "vram": "~2GB",
        "quantization": None,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Launch local model for POLARIS-Bench")
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()),
                       help="Model to launch")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                       help="GPU memory utilization (0-1)")
    args = parser.parse_args()
    
    model = MODELS[args.model]
    hf_id = model["hf_id"]
    
    print(f"\n{'='*60}")
    print(f"  Launching: {hf_id}")
    print(f"  VRAM: {model['vram']}")
    print(f"  Port: {args.port}")
    print(f"  Quantization: {model['quantization'] or 'none (fp16)'}")
    print(f"{'='*60}\n")
    
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", hf_id,
        "--port", str(args.port),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-model-len", "4096",
        "--trust-remote-code",
    ]
    
    if model["quantization"]:
        cmd.extend(["--quantization", model["quantization"]])
    
    hf_token = os.getenv("HF_TOKEN", "")
    env = os.environ.copy()
    if hf_token:
        env["HF_TOKEN"] = hf_token
    
    print(f"  Command: {' '.join(cmd)}\n")
    print(f"  Server will be available at: http://localhost:{args.port}/v1")
    print(f"  Then run: python run_full_benchmark.py --local-only --quick\n")
    
    try:
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        print("\n  Server stopped.")
    except FileNotFoundError:
        print("  ERROR: vLLM not installed. Run: pip install vllm")


if __name__ == "__main__":
    main()
