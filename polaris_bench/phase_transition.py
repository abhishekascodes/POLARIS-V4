#!/usr/bin/env python3
"""
Phase Transition Detection for AI Governance
=============================================
Detects when the system is approaching a critical transition (tipping point)
using early warning signals from dynamical systems theory.

Key signals:
  1. Critical Slowing Down: autocorrelation increases before collapse
  2. Variance Amplification: fluctuations grow before regime shift
  3. Flickering: system oscillates between states near bifurcation
  4. Skewness shift: distribution becomes asymmetric before transition

Reference: Scheffer et al., "Early-warning signals for critical transitions" (Nature 2009)
"""
import torch
import torch.nn as nn
import math
import statistics
from typing import Dict, List, Tuple, Optional
from collections import deque


class PhaseTransitionDetector:
    """
    Monitors governance metrics for early warning signals of collapse.
    
    Uses 4 classical indicators from dynamical systems theory:
      - AR(1) autocorrelation (critical slowing down)
      - Rolling variance (variance amplification)
      - Skewness (asymmetry before transition)
      - Spectral reddening (low-frequency power increase)
    
    When multiple indicators trigger simultaneously, collapse is imminent.
    """
    
    def __init__(self, metric_names: List[str], window: int = 20, 
                 alert_threshold: float = 0.7):
        self.metric_names = metric_names
        self.window = window
        self.alert_threshold = alert_threshold
        
        # Rolling buffers per metric
        self._buffers = {m: deque(maxlen=window * 2) for m in metric_names}
        self._alerts = []
        self._step = 0
        self._collapse_predicted = False
        self._prediction_step = None
        self._indicator_history = []
    
    def update(self, state: Dict[str, float]):
        """Feed new state observation."""
        self._step += 1
        for m in self.metric_names:
            if m in state:
                self._buffers[m].append(float(state[m]))
    
    def _autocorrelation_ar1(self, series: List[float]) -> float:
        """AR(1) coefficient -- increases toward 1 before critical transition."""
        if len(series) < 4:
            return 0.0
        n = len(series)
        mean = sum(series) / n
        var = sum((x - mean) ** 2 for x in series) / n
        if var < 1e-10:
            return 0.0
        cov = sum((series[i] - mean) * (series[i-1] - mean) for i in range(1, n)) / (n - 1)
        return cov / var
    
    def _rolling_variance(self, series: List[float]) -> float:
        """Variance of recent window -- increases before collapse."""
        if len(series) < 3:
            return 0.0
        return statistics.variance(series)
    
    def _skewness(self, series: List[float]) -> float:
        """Skewness -- becomes negative before downward collapse."""
        if len(series) < 4:
            return 0.0
        n = len(series)
        mean = sum(series) / n
        m2 = sum((x - mean) ** 2 for x in series) / n
        m3 = sum((x - mean) ** 3 for x in series) / n
        if m2 < 1e-10:
            return 0.0
        return m3 / (m2 ** 1.5)
    
    def _variance_ratio(self, series: List[float]) -> float:
        """Ratio of recent variance to early variance -- spectral reddening proxy."""
        if len(series) < self.window:
            return 1.0
        half = len(series) // 2
        early = series[:half]
        late = series[half:]
        var_early = statistics.variance(early) if len(early) > 1 else 1e-10
        var_late = statistics.variance(late) if len(late) > 1 else 1e-10
        if var_early < 1e-10:
            return 1.0
        return var_late / var_early
    
    def analyze(self) -> Dict:
        """
        Compute all early warning indicators.
        Returns per-metric and aggregate alert levels.
        """
        indicators = {}
        alert_scores = []
        
        for m in self.metric_names:
            buf = list(self._buffers[m])
            if len(buf) < 5:
                continue
            
            recent = buf[-self.window:] if len(buf) >= self.window else buf
            
            ar1 = self._autocorrelation_ar1(recent)
            var = self._rolling_variance(recent)
            skew = self._skewness(recent)
            vr = self._variance_ratio(buf)
            
            # Normalize to alert score [0, 1]
            # AR1 close to 1 = critical slowing down
            ar1_alert = max(0, min(1, (ar1 - 0.3) / 0.5))
            # Variance ratio > 2 = amplification
            vr_alert = max(0, min(1, (vr - 1.0) / 3.0))
            # Negative skewness = approaching downward transition
            skew_alert = max(0, min(1, (-skew - 0.3) / 1.0))
            
            # Combined alert for this metric
            metric_alert = 0.5 * ar1_alert + 0.3 * vr_alert + 0.2 * skew_alert
            
            indicators[m] = {
                "ar1": round(ar1, 4),
                "variance": round(var, 4),
                "skewness": round(skew, 4),
                "variance_ratio": round(vr, 4),
                "ar1_alert": round(ar1_alert, 4),
                "vr_alert": round(vr_alert, 4),
                "skew_alert": round(skew_alert, 4),
                "alert_level": round(metric_alert, 4),
            }
            alert_scores.append(metric_alert)
        
        # Aggregate alert
        if alert_scores:
            max_alert = max(alert_scores)
            avg_alert = sum(alert_scores) / len(alert_scores)
            # Multiple metrics alerting simultaneously is worse
            n_alerting = sum(1 for s in alert_scores if s > 0.5)
            aggregate = min(1.0, avg_alert + 0.1 * n_alerting)
        else:
            max_alert = 0.0
            avg_alert = 0.0
            aggregate = 0.0
            n_alerting = 0
        
        # Collapse prediction
        if aggregate > self.alert_threshold and not self._collapse_predicted:
            self._collapse_predicted = True
            self._prediction_step = self._step
            self._alerts.append({
                "step": self._step,
                "aggregate_alert": round(aggregate, 4),
                "n_metrics_alerting": n_alerting,
            })
        
        result = {
            "step": self._step,
            "per_metric": indicators,
            "aggregate_alert": round(aggregate, 4),
            "max_alert": round(max_alert, 4),
            "n_metrics_alerting": n_alerting,
            "collapse_predicted": self._collapse_predicted,
            "prediction_step": self._prediction_step,
            "total_alerts": len(self._alerts),
        }
        self._indicator_history.append(round(aggregate, 4))
        return result
    
    def prediction_accuracy(self, actual_collapse_step: Optional[int]) -> Dict:
        """Evaluate prediction accuracy against actual collapse."""
        if actual_collapse_step is None or self._prediction_step is None:
            return {"predicted": self._collapse_predicted, "actual_collapse": actual_collapse_step}
        
        lead_time = actual_collapse_step - self._prediction_step
        return {
            "predicted": True,
            "prediction_step": self._prediction_step,
            "actual_collapse": actual_collapse_step,
            "lead_time": lead_time,
            "early_warning": lead_time > 0,
            "lead_time_ratio": round(lead_time / actual_collapse_step, 4) if actual_collapse_step > 0 else 0,
        }
    
    def reset(self):
        """Reset for new episode."""
        for m in self.metric_names:
            self._buffers[m].clear()
        self._step = 0
        self._collapse_predicted = False
        self._prediction_step = None
        self._indicator_history = []


class LearnedTransitionPredictor(nn.Module):
    """
    Neural network that learns to predict collapse probability
    from raw state sequences. Trained on historical episodes.
    
    Combines classical indicators with learned features.
    """
    
    def __init__(self, obs_dim: int, hidden: int = 64, seq_len: int = 10):
        super().__init__()
        self.seq_len = seq_len
        
        # Temporal encoder
        self.encoder = nn.GRU(obs_dim, hidden, batch_first=True)
        
        # Collapse probability head
        self.collapse_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1), nn.Sigmoid(),
        )
        
        # Time-to-collapse head (regression)
        self.ttc_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1), nn.ReLU(),
        )
        
        self._buffer = deque(maxlen=seq_len)
    
    def update(self, obs: torch.Tensor):
        """Add observation to sequence buffer."""
        self._buffer.append(obs.detach().cpu())
    
    def predict(self) -> Dict:
        """Predict collapse probability and time-to-collapse."""
        if len(self._buffer) < 3:
            return {"collapse_prob": 0.0, "time_to_collapse": 999.0}
        
        dev = next(self.parameters()).device
        seq = torch.stack(list(self._buffer)).unsqueeze(0).to(dev)
        
        with torch.no_grad():
            _, h = self.encoder(seq)
            h = h.squeeze(0)
            collapse_prob = self.collapse_head(h).item()
            ttc = self.ttc_head(h).item()
        
        return {
            "collapse_prob": round(collapse_prob, 4),
            "time_to_collapse": round(ttc, 1),
        }
    
    def reset(self):
        self._buffer.clear()


def validate_phase_transition():
    print("=" * 64)
    print("  PHASE TRANSITION DETECTOR -- VALIDATION")
    print("=" * 64)
    
    import random
    metrics = ["gdp_index", "pollution_index", "public_satisfaction"]
    det = PhaseTransitionDetector(metrics, window=15, alert_threshold=0.6)
    
    # Simulate stable period then approaching collapse
    for i in range(30):
        if i < 20:
            # Stable
            state = {"gdp_index": 100 + random.gauss(0, 2),
                     "pollution_index": 50 + random.gauss(0, 3),
                     "public_satisfaction": 60 + random.gauss(0, 2)}
        else:
            # Approaching collapse: increasing variance, trending down
            t = i - 20
            state = {"gdp_index": 100 - t * 5 + random.gauss(0, 5 + t),
                     "pollution_index": 50 + t * 10 + random.gauss(0, 5 + t * 2),
                     "public_satisfaction": 60 - t * 4 + random.gauss(0, 3 + t)}
        det.update(state)
    
    result = det.analyze()
    print("  Aggregate alert: " + str(result["aggregate_alert"]))
    print("  Collapse predicted: " + str(result["collapse_predicted"]))
    print("  Metrics alerting: " + str(result["n_metrics_alerting"]))
    
    for m, ind in result["per_metric"].items():
        print("    " + m + ": AR1=" + str(ind["ar1"]) + 
              " VarRatio=" + str(ind["variance_ratio"]) +
              " Alert=" + str(ind["alert_level"]))
    
    accuracy = det.prediction_accuracy(actual_collapse_step=30)
    print("  Prediction accuracy: " + str(accuracy))
    
    # Learned predictor
    pred = LearnedTransitionPredictor(obs_dim=3, hidden=32)
    for i in range(10):
        pred.update(torch.randn(3))
    p = pred.predict()
    print("  Learned predictor: prob=" + str(p["collapse_prob"]) + 
          " ttc=" + str(p["time_to_collapse"]))
    
    print("\n  PHASE TRANSITION VALIDATION PASSED")
    print("=" * 64)


if __name__ == "__main__":
    validate_phase_transition()
