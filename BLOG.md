# POLARIS V3 – When AI Agents Stop Co-operating

*This post explains the design, training, and results behind POLARIS v3.*

## Introduction

Most AI systems today are designed to perform well in isolation. They optimize a clear objective in a controlled environment and produce strong results. But real world decision making rarely works like this.

In real systems like governments, markets or organizations, multiple actors interact, negotiate, and often disagree. Decisions are not made in isolation, and outcomes depend on how well different agents coordinate.

POLARIS V3 is built to explore this setting. It is a multi agent environment where AI systems must negotiate, cooperate, and survive under pressure.

POLARIS studies what happens when intelligent systems must coordinate with other intelligent systems — and what causes them to fail.

---

## The Problem

Traditional reinforcement learning environments assume:

- Stable dynamics
- Predictable rewards
- Fixed agent behavior

However, real world systems are:

- Multi agent
- Non stationary
- Driven by negotiation and conflict

These differences introduce failure modes that are not captured in standard environments. Coordination becomes difficult, and systems can break down even when individual decisions seem reasonable.

POLARIS is designed to simulate these conditions.

---

## What POLARIS Does

POLARIS models a nation with:

- **21 economic and social metrics** (GDP, pollution, public satisfaction, healthcare, unemployment, and more)
- **5 AI minister agents**, each representing different priorities (Economy, Environment, Health, Industry, Social Welfare)

Each minister can:

- Propose policies based on their priorities
- Negotiate with others using natural language
- Form coalitions to push their agenda
- Block decisions through vetoes

The main agent must:

- Maintain system stability across all 21 metrics
- Balance competing objectives (improving GDP might increase pollution)
- Anticipate how other agents will behave (Theory of Mind)
- Respond to time sensitive briefings with deadlines

Failure is not scripted. It emerges when coordination breaks down. If GDP drops below 15, pollution exceeds 290, or public satisfaction falls below 5, the nation collapses and the episode ends.

---

## Why It's Different

POLARIS introduces several elements that make coordination challenging:

- **Multi agent pressure** — other agents actively influence outcomes through proposals and vetoes
- **Negotiation dynamics** — decisions involve dialogue, compromise, and coalition building
- **Emergent failure** — system collapse arises from interactions, not hardcoded events
- **Theory of Mind requirements** — the agent must predict other agents' actions to avoid vetoes
- **Non stationary environment** — 6 variables drift over time, so strategies that worked early may fail later
- **4 difficulty tiers** — from Easy (single objective, no ministers) to Extreme (5 ministers, full negotiation, high chaos)

This creates a setting where success depends not just on making good decisions, but on understanding other agents.

---

## Training the System

To improve performance, I trained a Qwen 2.5 3B model using:

- **GRPO** (Group Relative Policy Optimization) via Hugging Face TRL
- **QLoRA** (4-bit quantization with LoRA r=16) for efficient fine tuning
- **6 component composite reward**: governance + Pareto optimality + Theory of Mind + coalition formation + briefing compliance + oscillation penalty
- **Curriculum training**: difficulty escalates from Easy to Extreme as the agent improves

Training was done on an NVIDIA RTX 5080 Laptop GPU. 100 GRPO steps took 788 seconds (about 13 minutes), with only 29.9M trainable parameters out of 1.73B total (1.73%).

Training helps the agent:

- Anticipate veto decisions before they happen
- Form better coalitions with aligned ministers
- Avoid destabilizing policies that trigger collapse
- Balance competing objectives more effectively

---

## Results

Training leads to measurable improvements in coordination and system stability.

| Metric | Before Training | After GRPO | Change |
|--------|:------:|:----------:|:------:|
| Avg Reward | 13.4 | **30.2** | **+126.3%** |
| Survival Rate | 0/5 | **1/5** | First survival |
| Coalition Formation | 12 | **35** | **2.9x** |
| Training Time | — | 788s | RTX 5080 |

Initial training improves coordination stability (~29%), while extended training and scaling lead to significant gains (+126%).

![GRPO Training Results](outputs/grpo_training/grpo_training_results.png)

### Curriculum Escalation

After training, the agent was tested across increasing difficulty levels:

| Difficulty | Avg Reward | Survived |
|:----------:|:----------:|:--------:|
| Easy | **40.8** | **3/3** |
| Medium | **38.3** | **2/3** |
| Hard | 24.9 | 0/3 |
| Extreme | 22.7 | 0/3 |

The trained agent dominates Easy and Medium levels while Hard and Extreme remain unsolved, proving genuine difficulty scaling.

**Key takeaway:** Training improves coordination, while scaling amplifies these effects significantly.

### Frontier Model Benchmark

I also benchmarked Llama 3.3 70B (via Groq API) against all tasks. The results show that even frontier models struggle with multi agent coordination:

- Llama 70B scores **0.96** on easy single objective governance
- But **collapses to 0.22** under multi agent negotiation pressure
- Shows very limited ability to predict minister vetoes

This confirms that POLARIS creates genuine difficulty that scales with model sophistication, and that there is massive room for improvement through RL training.

---

## Demo

Watch the system in action: [**POLARIS V3 Demo on YouTube**](https://youtu.be/jP-cAZvJ7aU)

In the demo:

- Agents negotiate policies in real time
- Conflicts emerge between ministers with opposing priorities
- Coordination breaks down under pressure
- The simulation shows GDP, pollution, satisfaction, and other metrics evolving live

Try it yourself: [**Live Demo on Hugging Face Spaces**](https://huggingface.co/spaces/asabhishek/polaris-v3)

---

## Why This Matters

Many real world systems fail not because of poor individual decisions, but because of coordination breakdowns.

POLARIS provides a way to study:

- How intelligent agents interact under pressure
- How negotiation affects outcomes
- Why systems collapse when coordination fails

This has potential applications in:

- Governance simulation
- Economic modeling
- Multi agent AI research
- Testing AI alignment in competitive settings

---

## Scalability

POLARIS is designed to scale across multiple dimensions:

- **Model size** — larger models improve coordination ability (3B showed +126% improvement)
- **Number of agents** — more agents increase interaction complexity (1 to 5 ministers)
- **Environment richness** — additional constraints create more realistic scenarios (6 difficulty tasks)
- **Training duration** — longer training with curriculum escalation pushes toward harder tasks

This allows the system to evolve toward more complex and realistic simulations.

---

## Final Thoughts

POLARIS is not just about maximizing reward. It is about understanding what happens when intelligence is distributed across multiple agents.

As AI systems become more interconnected, studying coordination and failure at this level becomes increasingly important.

---

## Links

- [Live Demo (Hugging Face Space)](https://huggingface.co/spaces/asabhishek/polaris-v3)
- [Demo Video (YouTube)](https://youtu.be/jP-cAZvJ7aU)
- [GitHub Repository](https://github.com/abhishekascodes/POLARIS-V3)
- [Colab Notebook](https://colab.research.google.com/github/abhishekascodes/POLARIS-V3/blob/main/POLARIS_v3_Demo.ipynb)

---

Built by **Abhishek A S** (17) for the Meta PyTorch OpenEnv Hackathon Grand Finale 2026.
