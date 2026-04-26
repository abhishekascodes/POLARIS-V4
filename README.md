---
title: POLARIS v3
emoji: 🌐
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: true
license: mit
short_description: Multi-Agent AI Governance Engine with Theory-of-Mind
tags:
  - openenv
  - reinforcement-learning
  - multi-agent
  - theory-of-mind
  - negotiation
  - governance
---

<div align="center">

# POLARIS v3 — Multi-Agent AI Governance Engine

**The first OpenEnv environment where LLM agents negotiate, form coalitions, predict vetoes, and learn governance through multi-agent interaction.**

> **This is the final hackathon submission. POLARIS v1 and v2 on this GitHub are earlier iterations — this v3 repo is the only one to be evaluated.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-4f46e5?style=for-the-badge)](https://github.com/OpenEnv-ai/openenv)
[![HF Space](https://img.shields.io/badge/Live_Demo-yellow?style=for-the-badge)](https://huggingface.co/spaces/asabhishek/polaris-v3)
[![Blog](https://img.shields.io/badge/Blog-orange?style=for-the-badge)](https://github.com/abhishekascodes/POLARIS-V3/blob/main/BLOG.md)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

**Built solo by [Abhishek A S](https://github.com/abhishekascodes) (17) · Meta PyTorch OpenEnv Hackathon 2026**

</div>

---

## What is POLARIS?

POLARIS simulates a nation with 21 economic metrics and **5 AI minister agents** — each with their own priorities, hidden agendas, and veto power. The LLM agent must negotiate with them, predict their behavior, and keep the nation alive.

Unlike simple RL environments, the challenge here is **other intelligent agents inside the environment** — creating real multi-agent pressure that even Llama 70B struggles with.

---

## Try It Now

| | |
|---|---|
| **Demo Video** | [**Watch on YouTube**](https://youtu.be/jP-cAZvJ7aU) |
| **Live Demo** | [**asabhishek-polaris-v3.hf.space**](https://asabhishek-polaris-v3.hf.space) |
| **Colab Notebook** | [**Open in Colab**](https://colab.research.google.com/github/abhishekascodes/POLARIS-V3/blob/main/POLARIS_v3_Demo.ipynb) |
| **GitHub Repo** | [**github.com/abhishekascodes/POLARIS-V3**](https://github.com/abhishekascodes/POLARIS-V3) |
| **Blog** | [**BLOG.md**](https://github.com/abhishekascodes/POLARIS-V3/blob/main/BLOG.md) |

> If Colab doesn't open, [view the notebook directly on GitHub](https://github.com/abhishekascodes/POLARIS-V3/blob/main/POLARIS_v3_Demo.ipynb).

**Advanced Dashboard:** A full real-time control panel with live metrics, coalition tracking, and system internals is available via the [control panel](https://asabhishek-polaris-v3.hf.space/control).

---

## Key Features

- **5 AI minister agents** with natural language negotiation, coalition offers, and veto threats
- **Theory-of-Mind training** — agent learns to predict who will veto and why
- **21-metric simulation** with 4-layer transitions, delayed effects, and non-stationary drift
- **6-component reward** — governance + Pareto + ToM + coalition + briefing + oscillation penalty
- **Auto-curriculum** — difficulty escalates from Easy to Extreme as the agent improves
- **Collapse mechanics** — GDP crashes, pollution spikes, or public revolt = game over
- **6 tasks** across 4 difficulty tiers with up to 300-step episodes
- **Causal explainability** — every step shows why things happened

---

## Training Results

Trained **Qwen 2.5 3B** with GRPO + QLoRA (4-bit) on RTX 5080 in 13 minutes:

| Metric | Before | After GRPO | Change |
|--------|:------:|:----------:|:------:|
| Avg Reward | 13.4 | **30.2** | **+126.3%** |
| Survival | 0/5 | **1/5** | First survival |
| Coalitions | 12 | **35** | 2.9x |

Training improves coordination efficiency, increasing average reward by ~29% and stabilizing performance across episodes. Scaling to larger training runs significantly amplifies these effects, with performance gains reaching +126%.

<p align="center">
  <img src="outputs/grpo_training/grpo_training_results.png" alt="GRPO Training Results — Before vs After comparison showing reward improvement, per-episode breakdown, curriculum escalation, and Theory-of-Mind metrics" width="800"/>
</p>

**Frontier model benchmark:** Llama 3.3 70B scores 0.96 on easy governance but collapses to 0.22 on multi-agent negotiation. Theory-of-Mind accuracy: 0%. POLARIS creates genuine difficulty that scales with model sophistication.

Training logs, evaluation outputs, and benchmark data are available in [`/outputs`](outputs/) for full transparency.

---

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Run dashboard
python dashboard_server.py
# Open http://localhost:8765

# 3. Train with GRPO
python train_grpo.py --model Qwen/Qwen2.5-3B-Instruct --steps 100
```

---

## Architecture

```
POLARIS v3 Engine
 ├── 5 AI Ministers — negotiate, propose, veto, form coalitions
 ├── 3-Phase Protocol — PROPOSE > NEGOTIATE > RESOLVE
 ├── 4-Layer Transitions — immediate > delayed > drift > cascade
 ├── Briefing Engine — timed intelligence with deadlines
 ├── 6-Component Reward — governance + ToM + Pareto + coalition + briefing
 └── Explainability — causal chains + counterfactual analysis
```

---

## Hackathon Themes Covered

| Theme | How POLARIS Addresses It |
|-------|------------------------|
| Multi-Agent Interactions | 5 minister agents with negotiation, coalitions, vetoes |
| Long-Horizon Planning | 200-300 step episodes with timed briefing deadlines |
| World Modeling | 21-metric simulation with non-stationary drift |
| Self-Improvement | Auto-curriculum from Easy to Extreme |

---

## All Links

| Resource | URL |
|----------|-----|
| Demo Video | [youtu.be/jP-cAZvJ7aU](https://youtu.be/jP-cAZvJ7aU) |
| HuggingFace Space | [huggingface.co/spaces/asabhishek/polaris-v3](https://huggingface.co/spaces/asabhishek/polaris-v3) |
| Blog | [BLOG.md](https://github.com/abhishekascodes/POLARIS-V3/blob/main/BLOG.md) |
| GitHub Repository | [github.com/abhishekascodes/POLARIS-V3](https://github.com/abhishekascodes/POLARIS-V3) |
| Colab Demo | [POLARIS_v3_Demo.ipynb](https://colab.research.google.com/github/abhishekascodes/POLARIS-V3/blob/main/POLARIS_v3_Demo.ipynb) |

---

<div align="center">

**POLARIS v3 — Where every policy decision is a negotiation, every minister has an agenda, and every veto tests your theory of mind.**

Built for the Meta PyTorch OpenEnv Hackathon × Scaler 2026

</div>
