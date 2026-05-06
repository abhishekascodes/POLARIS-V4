"""
POLARIS-Bench v4 — Publication-Quality Report Generator
========================================================

Generates charts, tables, and analysis suitable for arXiv papers.
Produces:
  - Coordination Collapse plot (THE money plot)
  - Per-dimension radar chart
  - Failure mode distribution
  - Model comparison tables
  - Scaling law curves
"""

from __future__ import annotations
import json
import os
import statistics
from typing import Dict, List, Optional, Any

from .metrics import ModelResults, CoordinationMetrics
from .scenarios import SCENARIOS, DIMENSIONS


class BenchmarkReport:
    """Generates publication-quality reports from evaluation results."""
    
    def __init__(self, results: Optional[ModelResults] = None):
        self.results = results
        self.all_model_results: List[ModelResults] = []
        if results:
            self.all_model_results.append(results)
    
    def add_model(self, results: ModelResults):
        """Add another model's results for comparison."""
        self.all_model_results.append(results)
    
    def generate_all(self, output_dir: str = "outputs/polaris_bench/report"):
        """Generate all report artifacts."""
        os.makedirs(output_dir, exist_ok=True)
        
        self.generate_leaderboard(output_dir)
        self.generate_summary_table(output_dir)
        
        try:
            self.generate_plots(output_dir)
        except ImportError:
            print("  matplotlib not available — skipping plots")
        
        print(f"\n  Report generated in: {output_dir}")
    
    def generate_leaderboard(self, output_dir: str):
        """Generate leaderboard JSON."""
        rows = [r.to_leaderboard_row() for r in self.all_model_results]
        rows.sort(key=lambda x: x["overall"], reverse=True)
        
        path = os.path.join(output_dir, "leaderboard.json")
        with open(path, "w") as f:
            json.dump(rows, f, indent=2)
        
        # Also print to console
        print(f"\n{'═'*80}")
        print(f"  POLARIS-Bench LEADERBOARD")
        print(f"{'═'*80}")
        print(f"  {'Model':<30} {'Overall':>8} {'Coord':>7} {'ToM':>7} {'Plan':>7} {'Adv':>7} {'CCR':>7}")
        print(f"  {'─'*30} {'─'*8} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")
        for row in rows:
            print(f"  {row['model']:<30} {row['overall']:>8.4f} {row['coord']:>7.4f} {row['tom']:>7.4f} {row['plan']:>7.4f} {row['adv']:>7.4f} {row['ccr']:>7.4f}")
        print(f"{'═'*80}")
    
    def generate_summary_table(self, output_dir: str):
        """Generate detailed summary table."""
        summary = []
        for r in self.all_model_results:
            entry = {
                "model": r.model_name,
                "family": r.model_family,
                "params": r.model_params,
                "overall": round(r.polaris_overall, 4),
                "ccr": round(r.ccr, 4),
                "dimensions": {
                    "coordination": round(r.polaris_coord, 4),
                    "theory_of_mind": round(r.polaris_tom, 4),
                    "long_horizon": round(r.polaris_plan, 4),
                    "adversarial": round(r.polaris_adv, 4),
                    "scaling": round(r.polaris_scale, 4),
                },
                "episodes": r.total_episodes,
                "collapses": r.total_collapses,
                "collapse_rate": round(r.total_collapses / max(r.total_episodes, 1), 4),
                "avg_tom_accuracy": round(r.avg_tom_accuracy, 4),
                "avg_coalition_rate": round(r.avg_coalition_rate, 4),
                "scenario_breakdown": {},
            }
            
            for sid, metrics in r.scenario_results.items():
                avg_score = statistics.mean(m.score for m in metrics)
                collapse_rate = sum(1 for m in metrics if m.collapsed) / len(metrics)
                
                # Failure mode analysis
                all_failures = []
                for m in metrics:
                    all_failures.extend(m.failure_modes)
                failure_counts = {}
                for f in all_failures:
                    failure_counts[f] = failure_counts.get(f, 0) + 1
                
                entry["scenario_breakdown"][sid] = {
                    "avg_score": round(avg_score, 4),
                    "collapse_rate": round(collapse_rate, 4),
                    "failure_modes": failure_counts,
                }
            
            summary.append(entry)
        
        path = os.path.join(output_dir, "summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
    
    def generate_plots(self, output_dir: str):
        """Generate publication-quality matplotlib plots."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Inter', 'Arial', 'Helvetica'],
            'font.size': 11,
            'axes.titlesize': 13,
            'axes.titleweight': 'bold',
            'figure.facecolor': '#fafafa',
            'axes.facecolor': '#fafafa',
            'axes.grid': True,
            'grid.alpha': 0.3,
        })
        
        if len(self.all_model_results) >= 2:
            self._plot_model_comparison(output_dir)
            self._plot_ccr_scaling(output_dir)
        
        for results in self.all_model_results:
            safe_name = results.model_name.replace("/", "_").replace(":", "_")
            self._plot_dimension_radar(results, output_dir, safe_name)
            self._plot_scenario_heatmap(results, output_dir, safe_name)
            self._plot_failure_distribution(results, output_dir, safe_name)
    
    def _plot_dimension_radar(self, results: ModelResults, output_dir: str, name: str):
        """Radar chart of 5 dimension scores."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        dims = ["Coordination", "Theory\nof Mind", "Long-Horizon\nPlanning", "Adversarial", "Scaling"]
        vals = [results.polaris_coord, results.polaris_tom, results.polaris_plan, results.polaris_adv, results.polaris_scale]
        
        angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
        vals_closed = vals + [vals[0]]
        angles_closed = angles + [angles[0]]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.fill(angles_closed, vals_closed, alpha=0.2, color='#4f46e5')
        ax.plot(angles_closed, vals_closed, color='#4f46e5', linewidth=2)
        ax.scatter(angles, vals, color='#4f46e5', s=60, zorder=5)
        
        ax.set_xticks(angles)
        ax.set_xticklabels(dims, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title(f"POLARIS-Bench: {results.model_name}\nOverall: {results.polaris_overall:.4f}", fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{name}_radar.png"), dpi=200, bbox_inches='tight')
        plt.close()
    
    def _plot_model_comparison(self, output_dir: str):
        """Bar chart comparing models across dimensions."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        models = [r.model_name.split("/")[-1][:20] for r in self.all_model_results]
        dims = ["Coord", "ToM", "Plan", "Adv", "Scale", "Overall"]
        
        n_models = len(models)
        n_dims = len(dims)
        x = np.arange(n_dims)
        width = 0.8 / n_models
        
        colors = ['#4f46e5', '#d97706', '#059669', '#e11d48', '#7c3aed', '#0284c7', '#ea580c', '#65a30d']
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        for i, results in enumerate(self.all_model_results):
            vals = [results.polaris_coord, results.polaris_tom, results.polaris_plan,
                    results.polaris_adv, results.polaris_scale, results.polaris_overall]
            offset = (i - n_models/2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=models[i], color=colors[i % len(colors)], edgecolor='white')
        
        ax.set_xticks(x)
        ax.set_xticklabels(dims)
        ax.set_ylabel("Score (0-1)")
        ax.set_title("POLARIS-Bench: Multi-Model Comparison", fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=200, bbox_inches='tight')
        plt.close()
    
    def _plot_ccr_scaling(self, output_dir: str):
        """THE MONEY PLOT: Model size vs CCR."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        param_map = {"1B": 1, "3B": 3, "7B": 7, "8B": 8, "9B": 9, "14B": 14,
                     "27B": 27, "32B": 32, "70B": 70, "72B": 72, "405B": 405}
        
        points = []
        for r in self.all_model_results:
            params = param_map.get(r.model_params, None)
            if params and r.ccr > 0:
                points.append((params, r.ccr, r.model_name.split("/")[-1][:15]))
        
        if len(points) < 2:
            return
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        xs, ys, labels = zip(*points)
        ax.scatter(xs, ys, s=120, c='#4f46e5', zorder=5, edgecolors='white', linewidths=1.5)
        
        for x, y, label in points:
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(8, 8), fontsize=8, color='#71717a')
        
        ax.set_xscale('log')
        ax.set_xlabel("Model Parameters (B)", fontsize=12)
        ax.set_ylabel("Coordination Collapse Ratio (CCR)", fontsize=12)
        ax.set_title("Scaling Does NOT Fix Coordination\nCoordination Collapse Ratio vs Model Size", fontsize=14, fontweight='bold')
        ax.axhline(y=0.5, color='#e11d48', linestyle='--', alpha=0.5, label='CCR = 0.5 (significant failure)')
        ax.axhline(y=0.3, color='#dc2626', linestyle='--', alpha=0.5, label='CCR = 0.3 (catastrophic failure)')
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "ccr_scaling.png"), dpi=200, bbox_inches='tight')
        plt.close()
    
    def _plot_scenario_heatmap(self, results: ModelResults, output_dir: str, name: str):
        """Heatmap of scores across all scenarios."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        scenario_ids = list(results.scenario_results.keys())
        if not scenario_ids:
            return
        
        scores = []
        labels = []
        for sid in scenario_ids:
            metrics = results.scenario_results[sid]
            avg = statistics.mean(m.score for m in metrics)
            scores.append(avg)
            scenario = SCENARIOS.get(sid, {})
            labels.append(scenario.get("name", sid)[:18])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        colors_arr = ['#dc2626' if s < 0.3 else '#d97706' if s < 0.5 else '#059669' for s in scores]
        
        bars = ax.barh(range(len(labels)), scores, color=colors_arr, edgecolor='white')
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Score (0-1)")
        ax.set_title(f"POLARIS-Bench Scenario Scores: {results.model_name}", fontweight='bold')
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(score + 0.02, i, f"{score:.3f}", va='center', fontsize=9, color='#3f3f46')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{name}_scenarios.png"), dpi=200, bbox_inches='tight')
        plt.close()
    
    def _plot_failure_distribution(self, results: ModelResults, output_dir: str, name: str):
        """Bar chart of failure mode frequency."""
        import matplotlib.pyplot as plt
        
        all_failures = {}
        for metrics_list in results.scenario_results.values():
            for m in metrics_list:
                for f in m.failure_modes:
                    all_failures[f] = all_failures.get(f, 0) + 1
        
        if not all_failures:
            return
        
        sorted_failures = sorted(all_failures.items(), key=lambda x: x[1], reverse=True)
        modes, counts = zip(*sorted_failures)
        
        # Pretty names
        mode_names = {
            "oscillation_trap": "Oscillation\nTrap",
            "appeasement_spiral": "Appeasement\nSpiral",
            "tunnel_vision": "Tunnel\nVision",
            "trust_death_spiral": "Trust Death\nSpiral",
            "coalition_betrayal_loop": "Coalition\nBetrayal",
            "veto_blindness": "Veto\nBlindness",
            "cascading_collapse": "Cascading\nCollapse",
            "premature_convergence": "Premature\nConvergence",
            "deadline_blindness": "Deadline\nBlindness",
            "metric_fixation": "Metric\nFixation",
        }
        
        labels = [mode_names.get(m, m) for m in modes]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        severity_colors = {
            "oscillation_trap": "#e11d48",
            "trust_death_spiral": "#dc2626",
            "cascading_collapse": "#dc2626",
            "tunnel_vision": "#e11d48",
            "veto_blindness": "#e11d48",
            "coalition_betrayal_loop": "#e11d48",
            "metric_fixation": "#e11d48",
            "appeasement_spiral": "#d97706",
            "premature_convergence": "#d97706",
            "deadline_blindness": "#d97706",
        }
        colors = [severity_colors.get(m, "#a1a1aa") for m in modes]
        
        ax.bar(range(len(labels)), counts, color=colors, edgecolor='white')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Occurrences")
        ax.set_title(f"Failure Mode Distribution: {results.model_name}", fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{name}_failures.png"), dpi=200, bbox_inches='tight')
        plt.close()
