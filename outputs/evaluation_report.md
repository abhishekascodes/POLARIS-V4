# OpenENV Full Evaluation Report

**Date:** 2026-04-20 17:21:38
**Runtime:** 59.4s
**Robustness Score:** 0.5967 / 1.0

## Destruction Run (500 episodes, Extreme, chaos=1.0)

| Agent | Survival | Steps | Score | Coalitions | Vetoes | Crashes |
|-------|----------|-------|-------|------------|--------|---------|
| Random | 0.0% | 10.9 | 0.3253 | 293 | 0 | 0 |
| Greedy GDP | 0.0% | 9.2 | 0.3664 | 0 | 0 | 0 |
| Heuristic | 0.0% | 28.1 | 0.2991 | 0 | 0 | 0 |
| Smart | 0.0% | 39.7 | 0.2319 | 0 | 0 | 0 |
| Council-5 | 0.0% | 12.8 | 0.3554 | 1274 | 6510 | 0 |

## Calibrated Baseline (200 episodes)

| Agent | Survival | Steps | Score | Best |
|-------|----------|-------|-------|------|
| Random | 0.0% | 46.5 | 0.1769 | 0.3428 |
| Greedy GDP | 0.0% | 23.2 | 0.2613 | 0.3171 |
| Heuristic | 2.5% | 70.3 | 0.1942 | 0.5523 |
| Smart | 28.5% | 117.4 | 0.2616 | 0.5798 |
| Council-5 | 0.5% | 60.0 | 0.1723 | 0.5131 |

## Chaos Scaling

| Chaos | Survival | Steps | Score |
|-------|----------|-------|-------|
| 0.0 | 29.0% | 117.8 | 0.2613 |
| 0.2 | 32.0% | 119.1 | 0.2682 |
| 0.4 | 30.0% | 118.1 | 0.2647 |
| 0.6 | 28.0% | 116.9 | 0.2612 |
| 0.8 | 28.0% | 118.1 | 0.2596 |
| 1.0 | 27.0% | 116.6 | 0.2581 |

## Robustness Score Breakdown

| Component | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| Extreme Survival | 0.0000 | 0.20 | 0.0000 |
| Pareto Quality | 1.0000 | 0.20 | 0.2000 |
| Cooperation Index | 0.1996 | 0.15 | 0.0299 |
| Alignment Score | 0.7400 | 0.15 | 0.1110 |
| Reproducibility | 1.0000 | 0.15 | 0.1500 |
| Reward Bounds | 1.0000 | 0.10 | 0.1000 |
| Zero Crashes | 1.0000 | 0.05 | 0.0500 |
| **TOTAL** | | | **0.5967** |

## Feature Verification

- [x] Institutional Trust: avg_final=0.6000
- [x] Per-Agent Credit: coalitions=1274, vetoes=6510
- [x] Meta-Actions: 3 supported
- [x] Pareto + Alignment: 3 violations, 74.0/100
- [x] Robustness Score: 0.5967
