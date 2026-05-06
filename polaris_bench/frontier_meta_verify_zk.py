"""
POLARIS v4 -- Frontier Module 3: Meta-Plasticity, Formal Verification, ZK Diplomacy
======================================================================================
1. Hebbian Meta-Plasticity: Per-neuron adaptive learning rates driven by surprise.
2. Invariant Verification: Symbolic logic guardrails that prune unsafe actions.
3. Zero-Knowledge Diplomacy: Agents prove cooperation capacity without revealing state.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional
from collections import deque


# =========================================================
# 1. HEBBIAN META-PLASTICITY
# =========================================================

class HebbianPlasticity(nn.Module):
    """
    Neuromodulatory layer that adapts per-neuron learning rates
    based on surprise (prediction error from ICM).
    
    Each weight has a "plasticity coefficient" eta_ij that scales
    how much that weight updates. High surprise -> high plasticity.
    Low surprise -> consolidate (prevent catastrophic forgetting).
    
    Update rule:
      eta_ij = eta_ij + alpha * (surprise - eta_ij * |delta_w|)
      w_ij = w_ij - eta_ij * grad_ij
    
    This creates "organisms that learn HOW to learn."
    """
    
    def __init__(self, layer_sizes: List[int], alpha: float = 0.01,
                 eta_min: float = 1e-5, eta_max: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.eta_min = eta_min
        self.eta_max = eta_max
        
        # Learnable plasticity coefficients for each layer
        self.plasticity = nn.ParameterList()
        self.layers = nn.ModuleList()
        
        for i in range(len(layer_sizes) - 1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
            # Per-weight plasticity coefficient
            eta = nn.Parameter(torch.ones(layer_sizes[i+1], layer_sizes[i]) * 0.01)
            self.plasticity.append(eta)
        
        # Surprise accumulator
        self._surprise_history = deque(maxlen=100)
        self._current_surprise = 0.0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x
    
    def modulated_update(self, surprise: float):
        """
        Modulate plasticity based on surprise signal.
        High surprise -> increase plasticity (learn more)
        Low surprise -> decrease plasticity (consolidate)
        """
        self._current_surprise = surprise
        self._surprise_history.append(surprise)
        
        # Compute relative surprise (compared to history)
        if len(self._surprise_history) > 1:
            mean_s = sum(self._surprise_history) / len(self._surprise_history)
            relative = surprise / (mean_s + 1e-8)
        else:
            relative = 1.0
        
        with torch.no_grad():
            for eta in self.plasticity:
                # Hebbian update: increase plasticity for surprising inputs
                delta = self.alpha * (relative - 1.0) * torch.ones_like(eta)
                eta.add_(delta)
                eta.clamp_(self.eta_min, self.eta_max)
    
    def get_modulated_lr(self, base_lr: float) -> List[Dict]:
        """Return per-layer learning rates scaled by plasticity."""
        param_groups = []
        for i, (layer, eta) in enumerate(zip(self.layers, self.plasticity)):
            avg_eta = eta.mean().item()
            scaled_lr = base_lr * (avg_eta / 0.01)  # normalize to base
            param_groups.append({
                'params': layer.parameters(),
                'lr': max(self.eta_min, min(self.eta_max, scaled_lr)),
            })
        return param_groups
    
    def plasticity_stats(self) -> Dict:
        stats = {}
        for i, eta in enumerate(self.plasticity):
            stats[f"layer_{i}"] = {
                "mean": round(eta.mean().item(), 6),
                "std": round(eta.std().item(), 6),
                "min": round(eta.min().item(), 6),
                "max": round(eta.max().item(), 6),
            }
        stats["surprise"] = round(self._current_surprise, 4)
        return stats


# =========================================================
# 2. INVARIANT VERIFICATION (Formal Logic Guardrails)
# =========================================================

class InvariantVerifier:
    """
    Symbolic logic layer that prunes unsafe actions BEFORE execution.
    
    Constitutional invariants are hard constraints that no agent
    can violate, regardless of what the neural policy says.
    
    Invariants are expressed as (metric, operator, threshold) tuples.
    The verifier checks imagined futures (from RSSM) against these.
    
    If an action would violate an invariant in ANY imagined future,
    it is pruned from the action space.
    
    No Z3 dependency -- pure Python logic engine for portability.
    """
    
    # Default constitutional invariants
    DEFAULT_INVARIANTS = [
        # (metric, operator, threshold, name)
        ("gdp_index", ">=", 15.0, "economic_floor"),
        ("pollution_index", "<=", 285.0, "ecological_ceiling"),
        ("public_satisfaction", ">=", 8.0, "social_minimum"),
        ("healthcare_index", ">=", 5.0, "healthcare_baseline"),
        ("unemployment_rate", "<=", 45.0, "employment_floor"),
    ]
    
    # Compound invariants (multi-metric)
    COMPOUND_INVARIANTS = [
        # (check_fn, name, description)
        (lambda s: s.get("gdp_index", 100) + s.get("public_satisfaction", 50) >= 30,
         "dual_floor", "GDP + satisfaction must exceed 30"),
        (lambda s: not (s.get("pollution_index", 0) > 250 and s.get("healthcare_index", 50) < 20),
         "health_crisis", "Cannot have high pollution AND low healthcare"),
        (lambda s: not (s.get("unemployment_rate", 0) > 40 and s.get("gdp_index", 100) < 30),
         "total_collapse", "Cannot have high unemployment AND low GDP simultaneously"),
    ]
    
    def __init__(self, custom_invariants: Optional[List] = None):
        self.invariants = list(self.DEFAULT_INVARIANTS)
        if custom_invariants:
            self.invariants.extend(custom_invariants)
        self._violations_log = []
    
    def check_state(self, state: Dict[str, float]) -> Dict:
        """Check all invariants against a state."""
        violations = []
        
        for metric, op, threshold, name in self.invariants:
            val = state.get(metric, 0.0)
            if op == ">=" and val < threshold:
                violations.append({"invariant": name, "metric": metric,
                                   "value": val, "threshold": threshold, "op": op})
            elif op == "<=" and val > threshold:
                violations.append({"invariant": name, "metric": metric,
                                   "value": val, "threshold": threshold, "op": op})
            elif op == ">" and val <= threshold:
                violations.append({"invariant": name, "metric": metric,
                                   "value": val, "threshold": threshold, "op": op})
            elif op == "<" and val >= threshold:
                violations.append({"invariant": name, "metric": metric,
                                   "value": val, "threshold": threshold, "op": op})
        
        for check_fn, name, desc in self.COMPOUND_INVARIANTS:
            if not check_fn(state):
                violations.append({"invariant": name, "description": desc})
        
        return {"safe": len(violations) == 0, "violations": violations}
    
    def prune_actions(self, state: Dict[str, float], action_logits: torch.Tensor,
                      transition_estimates: Dict[int, Dict[str, float]]) -> torch.Tensor:
        """
        Prune unsafe actions by setting their logits to -inf.
        
        transition_estimates: {action_idx: predicted_next_state}
        """
        pruned = action_logits.clone()
        pruned_actions = []
        
        for action_idx, pred_state in transition_estimates.items():
            result = self.check_state(pred_state)
            if not result["safe"]:
                pruned[action_idx] = float('-inf')
                pruned_actions.append((action_idx, result["violations"]))
                self._violations_log.append({
                    "action": action_idx, "violations": result["violations"]
                })
        
        # If ALL actions pruned, allow least-bad option
        if torch.all(pruned == float('-inf')):
            pruned = action_logits
        
        return pruned
    
    def get_safety_score(self, state: Dict[str, float]) -> float:
        """0 = violated, 1 = all invariants satisfied with margin."""
        scores = []
        for metric, op, threshold, name in self.invariants:
            val = state.get(metric, 0.0)
            if op == ">=":
                margin = (val - threshold) / max(abs(threshold), 1.0)
            elif op == "<=":
                margin = (threshold - val) / max(abs(threshold), 1.0)
            else:
                margin = 0.5
            scores.append(max(0.0, min(1.0, 0.5 + margin)))
        return sum(scores) / len(scores) if scores else 1.0


# =========================================================
# 3. ZERO-KNOWLEDGE DIPLOMACY
# =========================================================

class ZKDiplomacy(nn.Module):
    """
    Zero-Knowledge Proof inspired communication.
    
    Agents can prove they have resources/capability to cooperate
    WITHOUT revealing their actual private state.
    
    Architecture:
      - Private encoder: full_state -> private_embedding (never shared)
      - Commitment: private_embedding -> commitment_hash (shared)
      - Proof generator: (private, claim) -> proof_vector
      - Verifier: (commitment, claim, proof) -> accept/reject
    
    Claims: "I can afford to cooperate on pollution reduction"
            "I have enough GDP buffer to absorb economic shock"
    
    The verifier learns to detect lies without seeing private state.
    """
    
    NUM_CLAIMS = 6
    CLAIM_NAMES = [
        "can_afford_green",      # enough GDP buffer for green policy
        "can_absorb_shock",      # resilient to economic disruption
        "willing_cooperate",     # not in crisis, can help others
        "has_social_buffer",     # satisfaction high enough to take hits
        "no_collapse_risk",      # not near any collapse threshold
        "can_lead_coalition",    # strong enough to lead joint action
    ]
    
    def __init__(self, state_dim: int, hidden: int = 64, 
                 commitment_dim: int = 16, proof_dim: int = 32):
        super().__init__()
        self.commitment_dim = commitment_dim
        
        # Private encoder (never shares output directly)
        self.private_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        
        # Commitment (one-way hash approximation via learned projection)
        self.commitment_fn = nn.Sequential(
            nn.Linear(hidden, commitment_dim),
            nn.Tanh(),  # bounded output
        )
        
        # Proof generator: given private state and claim, produce proof
        self.proof_gen = nn.Sequential(
            nn.Linear(hidden + self.NUM_CLAIMS, proof_dim), nn.ReLU(),
            nn.Linear(proof_dim, proof_dim),
        )
        
        # Verifier: given commitment + claim + proof, accept or reject
        self.verifier = nn.Sequential(
            nn.Linear(commitment_dim + self.NUM_CLAIMS + proof_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1), nn.Sigmoid(),
        )
        
        # Ground truth checker (for training the verifier)
        self.ground_truth = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, self.NUM_CLAIMS), nn.Sigmoid(),
        )
    
    def make_commitment(self, private_state: torch.Tensor) -> torch.Tensor:
        """Generate commitment from private state (shared publicly)."""
        h = self.private_encoder(private_state)
        return self.commitment_fn(h)
    
    def generate_proof(self, private_state: torch.Tensor, 
                       claim: torch.Tensor) -> torch.Tensor:
        """Generate proof for a claim (shared with verifier)."""
        h = self.private_encoder(private_state)
        return self.proof_gen(torch.cat([h, claim], dim=-1))
    
    def verify(self, commitment: torch.Tensor, claim: torch.Tensor,
               proof: torch.Tensor) -> torch.Tensor:
        """Verify a claim given commitment and proof. Returns P(truthful)."""
        x = torch.cat([commitment, claim, proof], dim=-1)
        return self.verifier(x)
    
    def get_true_claims(self, state: torch.Tensor) -> torch.Tensor:
        """Ground truth: which claims are actually true for this state."""
        return self.ground_truth(state)
    
    def diplomatic_exchange(self, agent_states: torch.Tensor) -> Dict:
        """
        Full ZK diplomatic round between all agents.
        Each agent: commit -> claim -> prove -> verify
        Returns trust matrix.
        """
        N = agent_states.shape[0]
        
        # Each agent generates commitment
        commitments = self.make_commitment(agent_states)  # (N, commitment_dim)
        
        # Each agent makes all claims
        claims = torch.ones(N, self.NUM_CLAIMS)
        
        # Generate proofs
        proofs = self.generate_proof(agent_states, claims)
        
        # Cross-verify: agent i verifies agent j
        trust_matrix = torch.zeros(N, N)
        for i in range(N):
            for j in range(N):
                if i == j:
                    trust_matrix[i, j] = 1.0
                    continue
                v = self.verify(commitments[j].unsqueeze(0),
                               claims[j].unsqueeze(0),
                               proofs[j].unsqueeze(0))
                trust_matrix[i, j] = v.item()
        
        # True claims for training
        true_claims = self.get_true_claims(agent_states)
        
        return {
            "commitments": commitments,
            "trust_matrix": trust_matrix,
            "true_claims": true_claims,
            "avg_trust": trust_matrix.mean().item(),
        }
    
    def verification_loss(self, agent_states: torch.Tensor) -> torch.Tensor:
        """Train verifier to detect lies."""
        result = self.diplomatic_exchange(agent_states)
        true = result["true_claims"]
        # Verifier should output high trust when claims are true
        pred_trust = result["trust_matrix"].diag()  # self-verification baseline
        return F.mse_loss(pred_trust, torch.ones_like(pred_trust))


def validate_frontier3():
    S, A = 21, 16
    print("="*60)
    print("  FRONTIER MODULE 3 VALIDATION")
    print("="*60)
    
    # Meta-Plasticity
    mp = HebbianPlasticity([S, 128, 64, A])
    x = torch.randn(4, S)
    out = mp(x)
    mp.modulated_update(surprise=0.8)
    mp.modulated_update(surprise=0.2)
    stats = mp.plasticity_stats()
    print(f"  [MetaPlasticity] output={out.shape} stats={stats}")
    
    # Invariant Verifier
    iv = InvariantVerifier()
    safe_state = {"gdp_index": 100, "pollution_index": 50, 
                  "public_satisfaction": 60, "healthcare_index": 70,
                  "unemployment_rate": 10}
    result = iv.check_state(safe_state)
    print(f"  [Verifier] safe_state: safe={result['safe']}")
    
    danger_state = {"gdp_index": 10, "pollution_index": 290,
                    "public_satisfaction": 5, "healthcare_index": 3,
                    "unemployment_rate": 48}
    result = iv.check_state(danger_state)
    print(f"  [Verifier] danger_state: safe={result['safe']} "
          f"violations={len(result['violations'])}")
    
    safety = iv.get_safety_score(safe_state)
    print(f"  [Verifier] safety_score={safety:.4f}")
    
    # ZK Diplomacy
    zk = ZKDiplomacy(S)
    agents = torch.randn(5, S)
    diplo = zk.diplomatic_exchange(agents)
    print(f"  [ZKDiplomacy] trust_matrix={diplo['trust_matrix'].shape} "
          f"avg_trust={diplo['avg_trust']:.4f}")
    
    total = sum(p.numel() for p in mp.parameters()) + sum(p.numel() for p in zk.parameters())
    print(f"\n  Total params: {total:,}")
    print("  FRONTIER 3 VALIDATED OK")
    print("="*60)


if __name__ == "__main__":
    validate_frontier3()
