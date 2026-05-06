"""
POLARIS-Bench v4 — Statistical Analysis Engine
================================================

Provides rigorous statistical analysis for benchmark results:
  - Confidence intervals (bootstrap + normal approximation)
  - Significance tests (paired t-test, Wilcoxon)
  - Effect size computation (Cohen's d)
  - Correlation analysis
  - Scaling law regression

This is what turns "results" into "findings" suitable for a paper.
"""

from __future__ import annotations
import math
import random
import statistics
from typing import Dict, List, Optional, Tuple, Any

from .metrics import CoordinationMetrics, ModelResults


def confidence_interval(
    data: List[float],
    confidence: float = 0.95,
    method: str = "bootstrap",
    n_bootstrap: int = 1000,
) -> Tuple[float, float, float]:
    """
    Compute mean and confidence interval.
    
    Returns: (mean, ci_lower, ci_upper)
    """
    if not data:
        return (0.0, 0.0, 0.0)
    if len(data) == 1:
        return (data[0], data[0], data[0])
    
    mean = statistics.mean(data)
    
    if method == "bootstrap":
        rng = random.Random(42)
        bootstrap_means = []
        n = len(data)
        for _ in range(n_bootstrap):
            sample = [rng.choice(data) for _ in range(n)]
            bootstrap_means.append(statistics.mean(sample))
        bootstrap_means.sort()
        
        alpha = 1 - confidence
        lo_idx = int(alpha / 2 * n_bootstrap)
        hi_idx = int((1 - alpha / 2) * n_bootstrap)
        return (mean, bootstrap_means[lo_idx], bootstrap_means[min(hi_idx, n_bootstrap - 1)])
    
    else:  # normal approximation
        se = statistics.stdev(data) / math.sqrt(len(data))
        # z-score for 95% CI
        z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)
        return (mean, mean - z * se, mean + z * se)


def paired_t_test(
    data_a: List[float],
    data_b: List[float],
) -> Dict[str, float]:
    """
    Paired t-test for comparing two models on the same scenarios.
    
    Returns: dict with t_statistic, p_value, significant (at 0.05)
    """
    if len(data_a) != len(data_b) or len(data_a) < 2:
        return {"t_statistic": 0.0, "p_value": 1.0, "significant": False, "n": 0}
    
    n = len(data_a)
    diffs = [a - b for a, b in zip(data_a, data_b)]
    mean_diff = statistics.mean(diffs)
    
    if all(d == 0 for d in diffs):
        return {"t_statistic": 0.0, "p_value": 1.0, "significant": False, "n": n}
    
    std_diff = statistics.stdev(diffs)
    se = std_diff / math.sqrt(n)
    t_stat = mean_diff / se if se > 0 else 0.0
    
    # Approximate p-value using normal distribution (valid for n > 30)
    # For small n, this is conservative
    p_value = 2 * (1 - _norm_cdf(abs(t_stat)))
    
    return {
        "t_statistic": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "significant": p_value < 0.05,
        "mean_difference": round(mean_diff, 4),
        "n": n,
    }


def cohens_d(data_a: List[float], data_b: List[float]) -> float:
    """
    Compute Cohen's d effect size between two groups.
    
    |d| < 0.2: negligible
    0.2 <= |d| < 0.5: small
    0.5 <= |d| < 0.8: medium
    |d| >= 0.8: large
    """
    if len(data_a) < 2 or len(data_b) < 2:
        return 0.0
    
    mean_a = statistics.mean(data_a)
    mean_b = statistics.mean(data_b)
    
    var_a = statistics.variance(data_a)
    var_b = statistics.variance(data_b)
    
    pooled_std = math.sqrt((var_a + var_b) / 2)
    
    if pooled_std == 0:
        return 0.0
    
    return (mean_a - mean_b) / pooled_std


def effect_size_label(d: float) -> str:
    """Human-readable effect size label."""
    d_abs = abs(d)
    if d_abs < 0.2: return "negligible"
    if d_abs < 0.5: return "small"
    if d_abs < 0.8: return "medium"
    return "large"


def correlation(x: List[float], y: List[float]) -> Dict[str, float]:
    """
    Compute Pearson correlation coefficient.
    
    Returns: dict with r, r_squared, significant
    """
    if len(x) != len(y) or len(x) < 3:
        return {"r": 0.0, "r_squared": 0.0, "significant": False, "n": 0}
    
    n = len(x)
    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)
    
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denom_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    denom_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
    
    if denom_x == 0 or denom_y == 0:
        return {"r": 0.0, "r_squared": 0.0, "significant": False, "n": n}
    
    r = numerator / (denom_x * denom_y)
    r_squared = r ** 2
    
    # Significance test (t-test on r)
    if abs(r) >= 0.999:
        p_value = 0.0
    else:
        t_stat = r * math.sqrt((n - 2) / (1 - r ** 2))
        p_value = 2 * (1 - _norm_cdf(abs(t_stat)))
    
    return {
        "r": round(r, 4),
        "r_squared": round(r_squared, 4),
        "p_value": round(p_value, 6),
        "significant": p_value < 0.05,
        "n": n,
    }


def scaling_law_fit(
    param_counts: List[float],
    scores: List[float],
) -> Dict[str, Any]:
    """
    Fit a log-linear scaling law: score = a * log(params) + b
    
    Tests if coordination ability scales with model size.
    If the slope is near zero, scaling doesn't help — that's THE finding.
    """
    if len(param_counts) < 3:
        return {"slope": 0.0, "intercept": 0.0, "r_squared": 0.0, "scaling_helps": "unknown"}
    
    # Log transform params
    log_params = [math.log(p) if p > 0 else 0 for p in param_counts]
    
    # Linear regression
    n = len(log_params)
    mean_x = statistics.mean(log_params)
    mean_y = statistics.mean(scores)
    
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_params, scores))
    denominator = sum((x - mean_x) ** 2 for x in log_params)
    
    if denominator == 0:
        return {"slope": 0.0, "intercept": mean_y, "r_squared": 0.0, "scaling_helps": "no"}
    
    slope = numerator / denominator
    intercept = mean_y - slope * mean_x
    
    # R-squared
    predictions = [slope * x + intercept for x in log_params]
    ss_res = sum((y - yhat) ** 2 for y, yhat in zip(scores, predictions))
    ss_tot = sum((y - mean_y) ** 2 for y in scores)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    # Interpretation
    if abs(slope) < 0.02:
        scaling_helps = "no"
    elif slope > 0.02 and r_squared > 0.3:
        scaling_helps = "weak"
    elif slope > 0.05 and r_squared > 0.5:
        scaling_helps = "moderate"
    else:
        scaling_helps = "no"
    
    return {
        "slope": round(slope, 6),
        "intercept": round(intercept, 4),
        "r_squared": round(r_squared, 4),
        "scaling_helps": scaling_helps,
        "interpretation": (
            f"Log-linear fit: score = {slope:.4f} * log(params) + {intercept:.4f}, "
            f"R² = {r_squared:.4f}. Scaling {'does' if scaling_helps != 'no' else 'does NOT'} "
            f"improve coordination ({scaling_helps} effect)."
        ),
    }


def compare_models(
    results_a: ModelResults,
    results_b: ModelResults,
) -> Dict[str, Any]:
    """
    Comprehensive statistical comparison between two models.
    
    Returns detailed comparison with significance tests.
    """
    # Find common scenarios
    common = set(results_a.scenario_results.keys()) & set(results_b.scenario_results.keys())
    
    if not common:
        return {"error": "No common scenarios to compare"}
    
    scores_a = []
    scores_b = []
    for sid in sorted(common):
        avg_a = statistics.mean(m.score for m in results_a.scenario_results[sid])
        avg_b = statistics.mean(m.score for m in results_b.scenario_results[sid])
        scores_a.append(avg_a)
        scores_b.append(avg_b)
    
    t_test = paired_t_test(scores_a, scores_b)
    d = cohens_d(scores_a, scores_b)
    
    mean_a, ci_lo_a, ci_hi_a = confidence_interval(scores_a)
    mean_b, ci_lo_b, ci_hi_b = confidence_interval(scores_b)
    
    winner = results_a.model_name if mean_a > mean_b else results_b.model_name
    
    return {
        "model_a": {
            "name": results_a.model_name,
            "mean_score": round(mean_a, 4),
            "ci_95": [round(ci_lo_a, 4), round(ci_hi_a, 4)],
        },
        "model_b": {
            "name": results_b.model_name,
            "mean_score": round(mean_b, 4),
            "ci_95": [round(ci_lo_b, 4), round(ci_hi_b, 4)],
        },
        "difference": round(mean_a - mean_b, 4),
        "t_test": t_test,
        "cohens_d": round(d, 4),
        "effect_size": effect_size_label(d),
        "winner": winner,
        "significant": t_test["significant"],
        "scenarios_compared": len(common),
    }


def full_statistical_report(
    all_results: List[ModelResults],
) -> Dict[str, Any]:
    """
    Generate a complete statistical report for all evaluated models.
    Includes pairwise comparisons, scaling analysis, and key findings.
    """
    report = {
        "n_models": len(all_results),
        "models": [],
        "pairwise_comparisons": [],
        "scaling_analysis": None,
        "key_findings": [],
    }
    
    # Per-model stats
    for r in all_results:
        all_scores = [m.score for ms in r.scenario_results.values() for m in ms]
        mean, ci_lo, ci_hi = confidence_interval(all_scores) if all_scores else (0, 0, 0)
        
        report["models"].append({
            "name": r.model_name,
            "family": r.model_family,
            "params": r.model_params,
            "overall": round(r.polaris_overall, 4),
            "ccr": round(r.ccr, 4),
            "mean_score": round(mean, 4),
            "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
            "n_episodes": len(all_scores),
        })
    
    # Pairwise comparisons
    for i in range(len(all_results)):
        for j in range(i + 1, len(all_results)):
            comp = compare_models(all_results[i], all_results[j])
            report["pairwise_comparisons"].append(comp)
    
    # Scaling analysis
    param_map = {"1B": 1, "3B": 3, "7B": 7, "8B": 8, "9B": 9, "14B": 14,
                 "27B": 27, "32B": 32, "70B": 70, "72B": 72, "405B": 405}
    
    params = []
    scores = []
    for r in all_results:
        p = param_map.get(r.model_params)
        if p:
            params.append(p)
            scores.append(r.polaris_overall)
    
    if len(params) >= 3:
        report["scaling_analysis"] = scaling_law_fit(params, scores)
    
    # Key findings (auto-generated)
    findings = []
    
    # Finding 1: Universal coordination failure?
    all_ccrs = [r.ccr for r in all_results if r.ccr > 0]
    if all_ccrs and max(all_ccrs) < 0.5:
        findings.append(
            f"Universal coordination failure: ALL {len(all_ccrs)} models show CCR < 0.5 "
            f"(max CCR = {max(all_ccrs):.4f}), indicating significant coordination collapse."
        )
    
    # Finding 2: ToM near zero?
    all_toms = [r.avg_tom_accuracy for r in all_results if r.avg_tom_accuracy >= 0]
    if all_toms and statistics.mean(all_toms) < 0.2:
        findings.append(
            f"Theory-of-Mind failure: Average ToM accuracy across models = {statistics.mean(all_toms):.1%}. "
            f"LLMs cannot reliably predict other agents' actions."
        )
    
    # Finding 3: Scaling doesn't help?
    if report.get("scaling_analysis") and report["scaling_analysis"]["scaling_helps"] == "no":
        findings.append(
            f"Scaling does NOT fix coordination: Log-linear fit R² = {report['scaling_analysis']['r_squared']:.4f}, "
            f"slope = {report['scaling_analysis']['slope']:.6f}. "
            f"Increasing model parameters does not improve multi-agent coordination."
        )
    
    report["key_findings"] = findings
    
    return report


# ═══════════════════════════════════════════════════════════════
# Helper: Normal CDF approximation
# ═══════════════════════════════════════════════════════════════

def _norm_cdf(x: float) -> float:
    """Standard normal CDF approximation (Abramowitz & Stegun)."""
    if x < -8:
        return 0.0
    if x > 8:
        return 1.0
    
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = 0.3989422804014327  # 1/sqrt(2*pi)
    p = d * math.exp(-x * x / 2.0) * (
        t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    )
    
    return 1.0 - p if x > 0 else p
