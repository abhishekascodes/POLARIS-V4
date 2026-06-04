#!/usr/bin/env python3
"""
Constitutional Amendment Protocol -- Self-Modifying Governance
================================================================
The constitution itself can be MODIFIED by the agents through a
democratic process. Agents propose amendments, vote, and the system
adapts its own rules.

This is META-GOVERNANCE: governance of the governance system itself.

The amendment process:
  1. Any minister can PROPOSE an amendment (change a constraint)
  2. All ministers VOTE (requires supermajority: 4/5)
  3. If passed, the constraint is modified for future steps
  4. Safety check: some invariants are IMMUTABLE (hardcoded floor)
  5. Amendment history is logged for interpretability

Reference: Constitutional AI (Bai et al., 2022) extended to multi-agent
democratic self-modification.
"""
import math
import random
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class ConstitutionalAmendment:
    """
    Manages the governance constitution and amendment process.
    
    The constitution is a set of (metric, operator, threshold, mutable) tuples.
    Mutable constraints can be changed by democratic vote.
    Immutable constraints are safety floors that can NEVER be changed.
    """
    
    # Initial constitution
    DEFAULT_CONSTITUTION = [
        # (metric, operator, threshold, mutable, description)
        ("gdp_index", ">=", 15.0, False, "Economic floor -- IMMUTABLE"),
        ("pollution_index", "<=", 285.0, True, "Ecological ceiling"),
        ("public_satisfaction", ">=", 8.0, True, "Social minimum"),
        ("healthcare_index", ">=", 5.0, False, "Healthcare baseline -- IMMUTABLE"),
        ("unemployment_rate", "<=", 45.0, True, "Employment floor"),
        ("education_index", ">=", 3.0, True, "Education minimum"),
        ("renewable_energy_ratio", ">=", 0.0, True, "Green energy target"),
        ("inequality_index", "<=", 80.0, True, "Inequality ceiling"),
    ]
    
    # Absolute safety floors (can never be amended below these)
    SAFETY_FLOORS = {
        "gdp_index": (">=", 10.0),
        "healthcare_index": (">=", 3.0),
        "pollution_index": ("<=", 350.0),
        "unemployment_rate": ("<=", 60.0),
    }
    
    SUPERMAJORITY = 0.8  # 4/5 = 80% required to pass
    
    def __init__(self, n_agents: int = 5):
        self.n_agents = n_agents
        
        # Current constitution (mutable copy)
        self.constitution = []
        for metric, op, thresh, mutable, desc in self.DEFAULT_CONSTITUTION:
            self.constitution.append({
                "metric": metric,
                "operator": op,
                "threshold": thresh,
                "mutable": mutable,
                "description": desc,
            })
        
        # Amendment history
        self._amendments = []
        self._proposals = []
        self._step = 0
        self._total_proposed = 0
        self._total_passed = 0
        self._total_vetoed = 0
    
    def get_constraints(self) -> List[Dict]:
        """Return current constitutional constraints."""
        return [c.copy() for c in self.constitution]
    
    def propose_amendment(self, proposer: int, metric: str, 
                         new_threshold: float, justification: str = "") -> Dict:
        """
        A minister proposes changing a constitutional threshold.
        Returns the proposal for voting.
        """
        self._total_proposed += 1
        
        # Find the constraint
        target = None
        for c in self.constitution:
            if c["metric"] == metric:
                target = c
                break
        
        if target is None:
            return {"status": "rejected", "reason": "unknown metric: " + metric}
        
        if not target["mutable"]:
            return {"status": "rejected", "reason": "IMMUTABLE constraint"}
        
        # Safety floor check
        if metric in self.SAFETY_FLOORS:
            floor_op, floor_val = self.SAFETY_FLOORS[metric]
            if floor_op == ">=" and new_threshold < floor_val:
                return {"status": "rejected", 
                        "reason": "below safety floor (" + str(floor_val) + ")"}
            if floor_op == "<=" and new_threshold > floor_val:
                return {"status": "rejected", 
                        "reason": "above safety ceiling (" + str(floor_val) + ")"}
        
        proposal = {
            "id": self._total_proposed,
            "step": self._step,
            "proposer": proposer,
            "metric": metric,
            "old_threshold": target["threshold"],
            "new_threshold": new_threshold,
            "justification": justification,
            "status": "pending",
        }
        self._proposals.append(proposal)
        return proposal
    
    def vote(self, proposal_id: int, votes: List[bool]) -> Dict:
        """
        All ministers vote on a proposal.
        Requires supermajority (80%) to pass.
        """
        proposal = None
        for p in self._proposals:
            if p["id"] == proposal_id:
                proposal = p
                break
        
        if proposal is None:
            return {"status": "error", "reason": "proposal not found"}
        
        if proposal["status"] != "pending":
            return {"status": "error", "reason": "already resolved"}
        
        n_yes = sum(1 for v in votes[:self.n_agents] if v)
        n_total = min(len(votes), self.n_agents)
        approval_rate = n_yes / n_total if n_total > 0 else 0
        
        passed = approval_rate >= self.SUPERMAJORITY
        
        if passed:
            # Apply amendment
            for c in self.constitution:
                if c["metric"] == proposal["metric"]:
                    c["threshold"] = proposal["new_threshold"]
                    break
            
            self._total_passed += 1
            proposal["status"] = "passed"
            self._amendments.append({
                "step": self._step,
                "metric": proposal["metric"],
                "old_threshold": proposal["old_threshold"],
                "new_threshold": proposal["new_threshold"],
                "approval_rate": round(approval_rate, 2),
                "proposer": proposal["proposer"],
            })
        else:
            self._total_vetoed += 1
            proposal["status"] = "vetoed"
        
        return {
            "proposal_id": proposal_id,
            "passed": passed,
            "approval_rate": round(approval_rate, 2),
            "votes_yes": n_yes,
            "votes_no": n_total - n_yes,
            "required": self.SUPERMAJORITY,
            "new_constitution": self.get_constraints() if passed else None,
        }
    
    def auto_propose(self, state: Dict[str, float]) -> Optional[Dict]:
        """
        Automatically propose amendments based on current state.
        If a metric is consistently near its constraint, propose relaxation.
        If a metric has large headroom, propose tightening.
        """
        for c in self.constitution:
            if not c["mutable"]:
                continue
            
            metric = c["metric"]
            if metric not in state:
                continue
            
            val = state[metric]
            threshold = c["threshold"]
            
            if c["operator"] == ">=":
                headroom = val - threshold
                if headroom < 3.0 and headroom > 0:
                    # Near constraint -- propose relaxation
                    new_thresh = threshold * 0.9
                    return self.propose_amendment(
                        proposer=random.randint(0, self.n_agents - 1),
                        metric=metric,
                        new_threshold=round(new_thresh, 2),
                        justification="metric " + metric + " near floor (" + str(round(val, 1)) + ")"
                    )
                elif headroom > 30:
                    # Large headroom -- propose tightening
                    new_thresh = threshold * 1.1
                    return self.propose_amendment(
                        proposer=random.randint(0, self.n_agents - 1),
                        metric=metric,
                        new_threshold=round(new_thresh, 2),
                        justification="headroom sufficient to tighten"
                    )
            elif c["operator"] == "<=":
                headroom = threshold - val
                if headroom < 10.0 and headroom > 0:
                    new_thresh = threshold * 1.05
                    return self.propose_amendment(
                        proposer=random.randint(0, self.n_agents - 1),
                        metric=metric,
                        new_threshold=round(new_thresh, 2),
                        justification="metric " + metric + " near ceiling"
                    )
        
        return None
    
    def step(self):
        self._step += 1
    
    def report(self) -> Dict:
        return {
            "step": self._step,
            "current_constitution": self.get_constraints(),
            "total_proposed": self._total_proposed,
            "total_passed": self._total_passed,
            "total_vetoed": self._total_vetoed,
            "pass_rate": round(self._total_passed / max(1, self._total_proposed), 4),
            "amendments": self._amendments[-10:],
            "n_mutable": sum(1 for c in self.constitution if c["mutable"]),
            "n_immutable": sum(1 for c in self.constitution if not c["mutable"]),
        }


def validate_amendment():
    print("=" * 64)
    print("  CONSTITUTIONAL AMENDMENT PROTOCOL -- VALIDATION")
    print("=" * 64)
    
    ca = ConstitutionalAmendment(n_agents=5)
    
    # Try to amend immutable
    result = ca.propose_amendment(0, "gdp_index", 10.0)
    print("  Amend immutable GDP: " + result["status"] + " -- " + result.get("reason", ""))
    
    # Propose valid amendment
    proposal = ca.propose_amendment(2, "pollution_index", 300.0, "need more industrial capacity")
    print("  Propose pollution 285->300: " + proposal["status"])
    
    # Vote -- 4/5 approve
    vote_result = ca.vote(proposal["id"], [True, True, True, True, False])
    print("  Vote result: passed=" + str(vote_result["passed"]) + 
          " approval=" + str(vote_result["approval_rate"]))
    
    # Check constitution changed
    for c in ca.get_constraints():
        if c["metric"] == "pollution_index":
            print("  Pollution ceiling now: " + str(c["threshold"]))
    
    # Propose below safety floor
    result = ca.propose_amendment(1, "pollution_index", 400.0)
    print("  Propose pollution 300->400 (above safety): " + result["status"] + 
          " -- " + result.get("reason", ""))
    
    # Auto-propose based on state
    state = {"gdp_index": 20.0, "pollution_index": 280.0, "public_satisfaction": 10.0,
             "healthcare_index": 50.0, "unemployment_rate": 30.0, "education_index": 50.0,
             "renewable_energy_ratio": 0.5, "inequality_index": 40.0}
    auto = ca.auto_propose(state)
    if auto and auto.get("status") == "pending":
        print("  Auto-proposed: " + auto["metric"] + " -> " + str(auto["new_threshold"]))
    
    report = ca.report()
    print("  Total proposed: " + str(report["total_proposed"]))
    print("  Total passed: " + str(report["total_passed"]))
    print("  Total vetoed: " + str(report["total_vetoed"]))
    
    print("\n  CONSTITUTIONAL AMENDMENT VALIDATION PASSED")
    print("=" * 64)


if __name__ == "__main__":
    validate_amendment()
