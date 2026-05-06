"""POLARIS-Bench: The Multi-Agent LLM Coordination Benchmark"""
from setuptools import setup, find_packages

setup(
    name="polaris-bench",
    version="4.0.0",
    author="Abhishek A S",
    author_email="abhishekascodes@gmail.com",
    description="The first comprehensive benchmark for LLM multi-agent coordination",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/abhishekascodes/POLARIS-V4",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "openai>=1.0",
        "pydantic>=2.0",
        "matplotlib>=3.7",
        "numpy>=1.24",
    ],
    extras_require={
        "local": ["vllm>=0.4", "transformers>=4.40", "torch>=2.1"],
        "full": ["datasets>=2.18", "huggingface_hub>=0.20"],
    },
    entry_points={
        "console_scripts": [
            "polaris-bench=polaris_bench_run:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="benchmark llm multi-agent coordination ai-safety",
)
