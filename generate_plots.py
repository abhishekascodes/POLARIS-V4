#!/usr/bin/env python3
"""
POLARIS v4 — Publication-Quality Plot Generator
Reads experiment_results.json and produces all figures.
"""
import sys, os, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 11, 'axes.titleweight': 'bold', 'figure.facecolor': '#fafafa',
                     'axes.facecolor': '#fafafa', 'axes.grid': True, 'grid.alpha': 0.3})

COLORS = {'random': '#a1a1aa', 'heuristic': '#4f46e5', 'greedy_gdp': '#d97706', 'greedy_green': '#059669'}
OUT = "outputs/experiments/plots"

def generate_all_plots(results_path="outputs/experiments/experiment_results.json"):
    os.makedirs(OUT, exist_ok=True)
    with open(results_path) as f:
        data = json.load(f)
    
    plot_scaling(data.get("scaling", {}))
    plot_chaos(data.get("chaos", {}))
    plot_tasks(data.get("tasks", {}))
    plot_ablation(data.get("ablation", {}))
    plot_scaling_ccr(data.get("scaling", {}))
    print(f"All plots saved to: {OUT}/")

def plot_scaling(data):
    """Agent count vs performance — THE money plot."""
    if not data: return
    agents = sorted(set(int(k.split("_")[1]) for k in data.keys()))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for aname in ["random", "heuristic"]:
        scores = [data[f"scale_{n}"].get(aname, {}).get("avg_score", 0) for n in agents]
        stds = [data[f"scale_{n}"].get(aname, {}).get("std_score", 0) for n in agents]
        collapses = [data[f"scale_{n}"].get(aname, {}).get("collapse_rate", 0) for n in agents]
        ax1.errorbar(agents, scores, yerr=stds, marker='o', linewidth=2, capsize=5,
                    label=aname.title(), color=COLORS.get(aname, '#333'))
        ax2.plot(agents, [c*100 for c in collapses], marker='s', linewidth=2,
                label=aname.title(), color=COLORS.get(aname, '#333'))
    
    ax1.set_xlabel("Number of Agents"); ax1.set_ylabel("Score (0-1)")
    ax1.set_title("Performance vs Agent Count"); ax1.legend(); ax1.set_ylim(0, 1)
    ax2.set_xlabel("Number of Agents"); ax2.set_ylabel("Collapse Rate (%)")
    ax2.set_title("Collapse Rate vs Agent Count"); ax2.legend(); ax2.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(f"{OUT}/scaling_performance.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  [1/5] scaling_performance.png")

def plot_scaling_ccr(data):
    """CCR across agent counts — shows coordination collapse."""
    if not data: return
    agents = sorted(set(int(k.split("_")[1]) for k in data.keys()))
    if len(agents) < 2: return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    for aname in ["heuristic"]:
        base = data.get("scale_2", {}).get(aname, {}).get("avg_score", 0.5)
        if base <= 0: base = 0.5
        ccrs = [data[f"scale_{n}"].get(aname, {}).get("avg_score", 0) / base for n in agents]
        ax.plot(agents, ccrs, marker='o', linewidth=2.5, color='#4f46e5', label='CCR (Heuristic)')
    
    ax.axhline(y=0.5, color='#e11d48', linestyle='--', alpha=0.7, label='CCR = 0.5 (failure)')
    ax.axhline(y=0.3, color='#dc2626', linestyle='--', alpha=0.7, label='CCR = 0.3 (catastrophic)')
    ax.fill_between(agents, 0, 0.3, alpha=0.05, color='red')
    ax.fill_between(agents, 0.3, 0.5, alpha=0.03, color='orange')
    ax.set_xlabel("Number of Agents"); ax.set_ylabel("Coordination Collapse Ratio (CCR)")
    ax.set_title("Coordination Collapse Ratio vs Agent Count\nCCR < 0.5 = Coordination Failure")
    ax.legend(); ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(f"{OUT}/ccr_scaling.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  [2/5] ccr_scaling.png")

def plot_chaos(data):
    """Chaos level vs performance."""
    if not data: return
    levels = sorted(float(k.split("_")[1]) for k in data.keys())
    fig, ax = plt.subplots(figsize=(8, 6))
    for aname in ["random", "heuristic"]:
        scores = [data[f"chaos_{c}"].get(aname, {}).get("avg_score", 0) for c in levels]
        stds = [data[f"chaos_{c}"].get(aname, {}).get("std", 0) for c in levels]
        ax.errorbar(levels, scores, yerr=stds, marker='o', linewidth=2, capsize=5,
                    label=aname.title(), color=COLORS.get(aname, '#333'))
    ax.set_xlabel("Chaos Level"); ax.set_ylabel("Score (0-1)")
    ax.set_title("Performance Degrades with Chaos\nEnvironment Difficulty vs Agent Score")
    ax.legend(); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{OUT}/chaos_scaling.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  [3/5] chaos_scaling.png")

def plot_tasks(data):
    """Task difficulty comparison."""
    if not data: return
    tasks = list(data.keys())
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(tasks)); w = 0.35
    for i, aname in enumerate(["random", "heuristic"]):
        scores = [data[t].get(aname, {}).get("avg_score", 0) for t in tasks]
        stds = [data[t].get(aname, {}).get("std", 0) for t in tasks]
        ax.bar(x + i*w - w/2, scores, w, yerr=stds, label=aname.title(),
               color=COLORS.get(aname, '#333'), edgecolor='white', capsize=4)
    ax.set_xticks(x); ax.set_xticklabels([t.replace("_","\n") for t in tasks], fontsize=8)
    ax.set_ylabel("Score (0-1)"); ax.set_title("Performance Across Task Difficulty Levels")
    ax.legend(); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{OUT}/task_comparison.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  [4/5] task_comparison.png")

def plot_ablation(data):
    """Ablation study — what components matter."""
    if not data: return
    labels = list(data.keys())
    scores = [data[l]["avg_score"] for l in labels]
    stds = [data[l]["std"] for l in labels]
    collapse = [data[l]["collapse_rate"]*100 for l in labels]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = ['#4f46e5' if l == 'full' else '#d97706' for l in labels]
    ax1.barh(range(len(labels)), scores, xerr=stds, color=colors, edgecolor='white', capsize=4)
    ax1.set_yticks(range(len(labels))); ax1.set_yticklabels([l.replace("_"," ").title() for l in labels])
    ax1.set_xlabel("Score (0-1)"); ax1.set_title("Ablation: Score Impact"); ax1.invert_yaxis()
    
    ax2.barh(range(len(labels)), collapse, color=colors, edgecolor='white')
    ax2.set_yticks(range(len(labels))); ax2.set_yticklabels([l.replace("_"," ").title() for l in labels])
    ax2.set_xlabel("Collapse Rate (%)"); ax2.set_title("Ablation: Collapse Rate"); ax2.invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{OUT}/ablation.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  [5/5] ablation.png")

if __name__ == "__main__":
    generate_all_plots()
