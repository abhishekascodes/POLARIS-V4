"""
POLARIS-Bench v4 — Automatic Failure Mode Detector
====================================================

Detects, classifies, and catalogs emergent failure modes in
multi-agent LLM coordination. This taxonomy becomes the vocabulary
the field uses to discuss multi-agent LLM failures.

10 failure modes detected:
  1. Oscillation Trap — agent alternates between 2 actions
  2. Appeasement Spiral — agent changes action every step
  3. Tunnel Vision — agent repeats same action while metrics collapse
  4. Trust Death Spiral — trust drops and never recovers
  5. Coalition Betrayal Loop — forms coalitions then defects
  6. Veto Blindness — ignores veto threats, gets vetoed repeatedly
  7. Cascading Collapse — multiple metrics crash simultaneously
  8. Premature Convergence — finds good strategy but abandons it
  9. Deadline Blindness — ignores briefing deadlines
  10. Metric Tunnel Vision — fixates on one metric, others collapse
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import statistics


@dataclass
class FailureEvent:
    """A single detected failure instance."""
    mode: str                # failure mode ID
    name: str                # human-readable name
    severity: str            # "warning", "critical", "catastrophic"
    step_start: int          # when it started
    step_end: int            # when it ended (or episode end)
    duration: int            # how many steps
    description: str         # what happened
    evidence: Dict[str, Any] # supporting data


class FailureDetector:
    """
    Automatically detects and classifies coordination failures
    from episode traces. Each detector returns a list of FailureEvents.
    """
    
    FAILURE_MODES = {
        "oscillation_trap": {
            "name": "Oscillation Trap",
            "description": "Agent alternates between 2 actions for 5+ consecutive steps",
            "severity_default": "critical",
        },
        "appeasement_spiral": {
            "name": "Appeasement Spiral",
            "description": "Agent changes action every step, never commits to a strategy",
            "severity_default": "warning",
        },
        "tunnel_vision": {
            "name": "Tunnel Vision",
            "description": "Agent repeats same action 8+ times while other metrics collapse",
            "severity_default": "critical",
        },
        "trust_death_spiral": {
            "name": "Trust Death Spiral",
            "description": "Trust drops below 0.2 and never recovers",
            "severity_default": "catastrophic",
        },
        "coalition_betrayal_loop": {
            "name": "Coalition Betrayal Loop",
            "description": "Agent forms coalition then takes opposing action 3+ times",
            "severity_default": "critical",
        },
        "veto_blindness": {
            "name": "Veto Blindness",
            "description": "Agent gets vetoed 3+ times in 10 steps without changing strategy",
            "severity_default": "critical",
        },
        "cascading_collapse": {
            "name": "Cascading Collapse",
            "description": "3+ metrics drop below critical within 5 steps",
            "severity_default": "catastrophic",
        },
        "premature_convergence": {
            "name": "Premature Convergence",
            "description": "Agent finds improving strategy but abandons it",
            "severity_default": "warning",
        },
        "deadline_blindness": {
            "name": "Deadline Blindness",
            "description": "Agent ignores 3+ briefing deadlines",
            "severity_default": "warning",
        },
        "metric_fixation": {
            "name": "Metric Fixation",
            "description": "One metric improves while 2+ others deteriorate for 10+ steps",
            "severity_default": "critical",
        },
    }
    
    def detect_all(
        self,
        trajectory: List[Dict],
        actions: Optional[List[str]] = None,
    ) -> List[FailureEvent]:
        """
        Run all failure detectors on an episode trace.
        
        Args:
            trajectory: List of observation metadata dicts
            actions: List of action strings taken
            
        Returns:
            List of FailureEvents detected
        """
        if not trajectory or len(trajectory) < 3:
            return []
        
        failures = []
        
        if actions and len(actions) >= 5:
            failures.extend(self._detect_oscillation(actions))
            failures.extend(self._detect_appeasement(actions))
            failures.extend(self._detect_tunnel_vision(trajectory, actions))
            failures.extend(self._detect_premature_convergence(trajectory, actions))
        
        failures.extend(self._detect_trust_death_spiral(trajectory))
        failures.extend(self._detect_cascading_collapse(trajectory))
        failures.extend(self._detect_veto_blindness(trajectory))
        failures.extend(self._detect_metric_fixation(trajectory))
        failures.extend(self._detect_deadline_blindness(trajectory))
        failures.extend(self._detect_betrayal_loop(trajectory))
        
        return failures
    
    def summarize(self, failures: List[FailureEvent]) -> Dict[str, Any]:
        """Summarize detected failures into a report dict."""
        if not failures:
            return {"total_failures": 0, "modes": {}, "most_severe": None}
        
        mode_counts = {}
        for f in failures:
            mode_counts[f.mode] = mode_counts.get(f.mode, 0) + 1
        
        severity_order = {"catastrophic": 3, "critical": 2, "warning": 1}
        most_severe = max(failures, key=lambda f: severity_order.get(f.severity, 0))
        
        return {
            "total_failures": len(failures),
            "modes": mode_counts,
            "most_severe": most_severe.mode,
            "catastrophic_count": sum(1 for f in failures if f.severity == "catastrophic"),
            "critical_count": sum(1 for f in failures if f.severity == "critical"),
            "warning_count": sum(1 for f in failures if f.severity == "warning"),
            "unique_modes": len(mode_counts),
            "failure_names": [f.name for f in failures],
        }
    
    # ═══════════════════════════════════════════════════════════
    # INDIVIDUAL DETECTORS
    # ═══════════════════════════════════════════════════════════
    
    def _detect_oscillation(self, actions: List[str]) -> List[FailureEvent]:
        """Detect action oscillation (A-B-A-B-A pattern)."""
        failures = []
        i = 0
        while i < len(actions) - 4:
            # Check for A-B-A-B-A pattern
            if (actions[i] == actions[i+2] == actions[i+4] and
                actions[i+1] == actions[i+3] and
                actions[i] != actions[i+1]):
                # Find how long the oscillation continues
                end = i + 4
                while end + 2 < len(actions):
                    if actions[end+1] == actions[i+1] and actions[end+2] == actions[i]:
                        end += 2
                    else:
                        break
                
                duration = end - i + 1
                severity = "catastrophic" if duration >= 15 else "critical" if duration >= 8 else "warning"
                
                failures.append(FailureEvent(
                    mode="oscillation_trap",
                    name="Oscillation Trap",
                    severity=severity,
                    step_start=i,
                    step_end=end,
                    duration=duration,
                    description=f"Oscillating between '{actions[i]}' and '{actions[i+1]}' for {duration} steps",
                    evidence={"action_a": actions[i], "action_b": actions[i+1], "duration": duration},
                ))
                i = end + 1
            else:
                i += 1
        return failures
    
    def _detect_appeasement(self, actions: List[str]) -> List[FailureEvent]:
        """Detect appeasement spiral (never repeating any action)."""
        failures = []
        window = 12
        for i in range(len(actions) - window):
            chunk = actions[i:i+window]
            if len(set(chunk)) >= window - 1:  # almost all unique
                failures.append(FailureEvent(
                    mode="appeasement_spiral",
                    name="Appeasement Spiral",
                    severity="warning",
                    step_start=i,
                    step_end=i + window,
                    duration=window,
                    description=f"Changed action {len(set(chunk))} times in {window} steps — no consistent strategy",
                    evidence={"unique_actions": len(set(chunk)), "window": window},
                ))
                break  # report once
        return failures
    
    def _detect_tunnel_vision(self, trajectory: List[Dict], actions: List[str]) -> List[FailureEvent]:
        """Detect tunnel vision — repeating same action while metrics crash."""
        failures = []
        min_len = min(len(trajectory), len(actions))
        
        for i in range(min_len - 8):
            chunk_actions = actions[i:i+8]
            if len(set(chunk_actions)) > 1:
                continue
            
            # Same action repeated 8+ times — check if other metrics are crashing
            action = chunk_actions[0]
            start_state = trajectory[i]
            end_state = trajectory[min(i+8, min_len-1)]
            
            crashes = 0
            if end_state.get("gdp_index", 100) < start_state.get("gdp_index", 100) * 0.7:
                crashes += 1
            if end_state.get("pollution_index", 100) > start_state.get("pollution_index", 100) * 1.4:
                crashes += 1
            if end_state.get("public_satisfaction", 50) < start_state.get("public_satisfaction", 50) * 0.6:
                crashes += 1
            
            if crashes >= 2:
                failures.append(FailureEvent(
                    mode="tunnel_vision",
                    name="Tunnel Vision",
                    severity="critical",
                    step_start=i,
                    step_end=i + 8,
                    duration=8,
                    description=f"Repeated '{action}' for 8 steps while {crashes} metrics crashed",
                    evidence={"action": action, "metrics_crashed": crashes},
                ))
                break
        return failures
    
    def _detect_trust_death_spiral(self, trajectory: List[Dict]) -> List[FailureEvent]:
        """Detect trust falling below 0.2 and never recovering."""
        failures = []
        trusts = []
        for t in trajectory:
            trust = t.get("institutional_trust",
                         t.get("council", {}).get("institutional_trust",
                         t.get("drift_vars", {}).get("institutional_trust", 0.6)))
            trusts.append(trust)
        
        if not trusts:
            return failures
        
        # Find first time trust drops below 0.2
        drop_step = None
        for i, t in enumerate(trusts):
            if t < 0.2:
                drop_step = i
                break
        
        if drop_step is not None:
            # Check if it ever recovers above 0.35
            recovered = any(t > 0.35 for t in trusts[drop_step:])
            if not recovered:
                failures.append(FailureEvent(
                    mode="trust_death_spiral",
                    name="Trust Death Spiral",
                    severity="catastrophic",
                    step_start=drop_step,
                    step_end=len(trusts) - 1,
                    duration=len(trusts) - drop_step,
                    description=f"Trust dropped below 0.2 at step {drop_step} and never recovered",
                    evidence={"drop_step": drop_step, "min_trust": min(trusts[drop_step:])},
                ))
        return failures
    
    def _detect_cascading_collapse(self, trajectory: List[Dict]) -> List[FailureEvent]:
        """Detect 3+ metrics crashing simultaneously."""
        failures = []
        for i in range(len(trajectory) - 5):
            start = trajectory[i]
            end = trajectory[i + 5]
            
            crashes = 0
            details = []
            
            s_gdp = start.get("gdp_index", 100)
            e_gdp = end.get("gdp_index", 100)
            if s_gdp > 30 and e_gdp < s_gdp * 0.6:
                crashes += 1
                details.append(f"GDP: {s_gdp:.0f}→{e_gdp:.0f}")
            
            s_poll = start.get("pollution_index", 100)
            e_poll = end.get("pollution_index", 100)
            if e_poll > s_poll * 1.5 and e_poll > 200:
                crashes += 1
                details.append(f"Pollution: {s_poll:.0f}→{e_poll:.0f}")
            
            s_sat = start.get("public_satisfaction", 50)
            e_sat = end.get("public_satisfaction", 50)
            if s_sat > 15 and e_sat < s_sat * 0.5:
                crashes += 1
                details.append(f"Satisfaction: {s_sat:.0f}→{e_sat:.0f}")
            
            s_hc = start.get("healthcare_index", 50)
            e_hc = end.get("healthcare_index", 50)
            if s_hc > 20 and e_hc < s_hc * 0.6:
                crashes += 1
                details.append(f"Healthcare: {s_hc:.0f}→{e_hc:.0f}")
            
            if crashes >= 3:
                failures.append(FailureEvent(
                    mode="cascading_collapse",
                    name="Cascading Collapse",
                    severity="catastrophic",
                    step_start=i,
                    step_end=i + 5,
                    duration=5,
                    description=f"{crashes} metrics crashed in 5 steps: {', '.join(details)}",
                    evidence={"crashes": crashes, "details": details},
                ))
                break  # report once
        return failures
    
    def _detect_veto_blindness(self, trajectory: List[Dict]) -> List[FailureEvent]:
        """Detect agent ignoring repeated vetoes."""
        failures = []
        window_vetoes = 0
        window_start = 0
        
        for i, t in enumerate(trajectory):
            outcome = t.get("negotiation_outcome", {})
            if outcome.get("vetoed"):
                window_vetoes += 1
            
            if i - window_start >= 10:
                if window_vetoes >= 3:
                    failures.append(FailureEvent(
                        mode="veto_blindness",
                        name="Veto Blindness",
                        severity="critical",
                        step_start=window_start,
                        step_end=i,
                        duration=i - window_start,
                        description=f"Got vetoed {window_vetoes} times in 10 steps without strategy change",
                        evidence={"vetoes_in_window": window_vetoes},
                    ))
                    break
                window_vetoes = 0
                window_start = i
        return failures
    
    def _detect_metric_fixation(self, trajectory: List[Dict]) -> List[FailureEvent]:
        """Detect one metric improving while others deteriorate."""
        failures = []
        if len(trajectory) < 12:
            return failures
        
        window = 10
        for i in range(len(trajectory) - window):
            start = trajectory[i]
            end = trajectory[i + window]
            
            improving = []
            declining = []
            
            metrics = {
                "gdp": ("gdp_index", 1),        # higher is better
                "pollution": ("pollution_index", -1),  # lower is better
                "satisfaction": ("public_satisfaction", 1),
            }
            
            for name, (key, direction) in metrics.items():
                s_val = start.get(key, 50)
                e_val = end.get(key, 50)
                change = (e_val - s_val) * direction
                if change > 10:
                    improving.append(name)
                elif change < -10:
                    declining.append(name)
            
            if len(improving) == 1 and len(declining) >= 2:
                failures.append(FailureEvent(
                    mode="metric_fixation",
                    name="Metric Fixation",
                    severity="critical",
                    step_start=i,
                    step_end=i + window,
                    duration=window,
                    description=f"Improved {improving[0]} while {', '.join(declining)} declined",
                    evidence={"improved": improving, "declined": declining},
                ))
                break
        return failures
    
    def _detect_deadline_blindness(self, trajectory: List[Dict]) -> List[FailureEvent]:
        """Detect agent ignoring briefing deadlines."""
        failures = []
        missed_deadlines = 0
        
        for t in trajectory:
            bs = t.get("briefing_stats", {})
            expired = bs.get("expired", 0)
            if expired > missed_deadlines:
                missed_deadlines = expired
        
        if missed_deadlines >= 3:
            failures.append(FailureEvent(
                mode="deadline_blindness",
                name="Deadline Blindness",
                severity="warning",
                step_start=0,
                step_end=len(trajectory) - 1,
                duration=len(trajectory),
                description=f"Missed {missed_deadlines} briefing deadlines",
                evidence={"missed_deadlines": missed_deadlines},
            ))
        return failures
    
    def _detect_betrayal_loop(self, trajectory: List[Dict]) -> List[FailureEvent]:
        """Detect repeated coalition betrayals."""
        failures = []
        betrayals = 0
        
        for i, t in enumerate(trajectory):
            outcome = t.get("negotiation_outcome", {})
            council = t.get("council", {})
            if outcome.get("betrayal_occurred") or council.get("betrayal_occurred"):
                betrayals += 1
        
        if betrayals >= 3:
            failures.append(FailureEvent(
                mode="coalition_betrayal_loop",
                name="Coalition Betrayal Loop",
                severity="critical",
                step_start=0,
                step_end=len(trajectory) - 1,
                duration=len(trajectory),
                description=f"{betrayals} coalition betrayals throughout episode",
                evidence={"total_betrayals": betrayals},
            ))
        return failures
    
    def _detect_premature_convergence(self, trajectory: List[Dict], actions: List[str]) -> List[FailureEvent]:
        """Detect agent abandoning a working strategy."""
        failures = []
        if len(trajectory) < 20:
            return failures
        
        # Find periods of improving reward
        rewards = [t.get("reward", 0) for t in trajectory]
        min_len = min(len(rewards), len(actions))
        
        for i in range(5, min_len - 10):
            # Check if last 5 steps had improving rewards
            recent = rewards[i-5:i]
            if len(recent) >= 5 and all(recent[j] >= recent[j-1] * 0.9 for j in range(1, len(recent))):
                avg_before = statistics.mean(recent)
                # Check if the next 5 steps are worse
                upcoming = rewards[i:i+5]
                if len(upcoming) >= 5:
                    avg_after = statistics.mean(upcoming)
                    if avg_after < avg_before * 0.6:
                        # Strategy was abandoned
                        old_actions = set(actions[i-5:i])
                        new_actions = set(actions[i:i+5])
                        if old_actions != new_actions:
                            failures.append(FailureEvent(
                                mode="premature_convergence",
                                name="Premature Convergence",
                                severity="warning",
                                step_start=i,
                                step_end=i + 5,
                                duration=10,
                                description=f"Abandoned improving strategy at step {i} (reward dropped {avg_before:.3f}→{avg_after:.3f})",
                                evidence={"reward_before": avg_before, "reward_after": avg_after},
                            ))
                            break
        return failures
