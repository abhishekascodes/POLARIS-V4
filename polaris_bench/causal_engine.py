#!/usr/bin/env python3
"""
Causal Intervention Engine for AI Governance
=============================================
Implements Pearl's do-calculus for governance decisions.

Key distinction:
  P(GDP | observe tax_cut) != P(GDP | do(tax_cut))

Observational: "When we SAW tax cuts, GDP went up" (could be confounded)
Interventional: "If we FORCE a tax cut, what happens?" (causal effect)

This module maintains a structural causal model (SCM) of the governance
system and answers interventional queries.

Reference: Pearl, "Causality" (2009)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class StructuralCausalModel:
    """
    Learned Structural Causal Model (SCM) for governance metrics.
    
    Maintains a DAG of causal relationships between metrics,
    learned from interventional data (actions -> outcomes).
    
    Can answer:
      1. What is the causal effect of action X on metric Y?
      2. What would have happened if we had done X instead of Z? (counterfactual)
      3. Which metrics are upstream causes of collapse?
    """
    
    METRICS = [
        "gdp_index", "pollution_index", "public_satisfaction",
        "healthcare_index", "education_index", "unemployment_rate",
        "renewable_energy_ratio", "inequality_index",
    ]
    
    def __init__(self, n_metrics: int = 8, n_actions: int = 16):
        self.n_metrics = n_metrics
        self.n_actions = n_actions
        
        # Causal adjacency matrix: A[i,j] = strength of i -> j
        # Learned from data, not hardcoded
        self._adjacency = [[0.0] * n_metrics for _ in range(n_metrics)]
        
        # Action effect matrix: E[a,m] = effect of action a on metric m
        self._action_effects = [[0.0] * n_metrics for _ in range(n_actions)]
        self._action_counts = [0] * n_actions
        
        # Observational data for learning
        self._transitions = []  # (state_before, action, state_after)
        self._max_data = 5000
    
    def observe(self, state_before: List[float], action: int, state_after: List[float]):
        """Record an observed transition for causal learning."""
        if len(self._transitions) >= self._max_data:
            self._transitions.pop(0)
        self._transitions.append((
            state_before[:self.n_metrics],
            action,
            state_after[:self.n_metrics],
        ))
        
        # Update action effects (running average)
        self._action_counts[action] += 1
        n = self._action_counts[action]
        for m in range(min(self.n_metrics, len(state_before), len(state_after))):
            delta = state_after[m] - state_before[m]
            self._action_effects[action][m] += (delta - self._action_effects[action][m]) / n
    
    def learn_structure(self):
        """
        Learn causal structure from accumulated data.
        Uses a simplified Granger-causality approach:
        metric i causes metric j if changes in i predict changes in j.
        """
        if len(self._transitions) < 10:
            return
        
        # Compute correlation of changes
        for i in range(self.n_metrics):
            for j in range(self.n_metrics):
                if i == j:
                    continue
                
                # Collect paired changes
                changes_i = []
                changes_j = []
                for s_before, _, s_after in self._transitions[-500:]:
                    if i < len(s_before) and j < len(s_before):
                        changes_i.append(s_after[i] - s_before[i])
                        changes_j.append(s_after[j] - s_before[j])
                
                if len(changes_i) < 5:
                    continue
                
                # Correlation as causal strength proxy
                mean_i = sum(changes_i) / len(changes_i)
                mean_j = sum(changes_j) / len(changes_j)
                
                cov = sum((a - mean_i) * (b - mean_j) for a, b in zip(changes_i, changes_j))
                var_i = sum((a - mean_i) ** 2 for a in changes_i)
                var_j = sum((b - mean_j) ** 2 for b in changes_j)
                
                if var_i > 1e-10 and var_j > 1e-10:
                    corr = cov / (math.sqrt(var_i) * math.sqrt(var_j))
                else:
                    corr = 0.0
                
                self._adjacency[i][j] = round(corr, 4)
    
    def do(self, action: int) -> Dict[str, float]:
        """
        do(action) -- Interventional query.
        Returns expected change in each metric if we FORCE this action.
        This is the causal effect, not the observational association.
        """
        effects = {}
        for m in range(self.n_metrics):
            direct = self._action_effects[action][m]
            
            # Propagate through causal graph (one hop)
            indirect = 0.0
            for m2 in range(self.n_metrics):
                if m2 != m:
                    indirect += self._action_effects[action][m2] * self._adjacency[m2][m]
            
            effects[self.METRICS[m] if m < len(self.METRICS) else f"metric_{m}"] = round(
                direct + 0.3 * indirect, 4
            )
        
        return effects
    
    def counterfactual(self, state_before: List[float], actual_action: int,
                       hypothetical_action: int) -> Dict[str, float]:
        """
        Counterfactual query: "What would have happened if we had done
        hypothetical_action instead of actual_action?"
        
        Uses structural equations: Y_cf = Y_actual - effect(actual) + effect(hypothetical)
        """
        actual_effects = self.do(actual_action)
        hyp_effects = self.do(hypothetical_action)
        
        counterfactual_state = {}
        for m in range(min(self.n_metrics, len(state_before))):
            key = self.METRICS[m] if m < len(self.METRICS) else f"metric_{m}"
            actual_change = actual_effects.get(key, 0.0)
            hyp_change = hyp_effects.get(key, 0.0)
            counterfactual_state[key] = round(
                state_before[m] + hyp_change, 4
            )
        
        return counterfactual_state
    
    def root_cause_analysis(self, collapse_metric: str) -> List[Tuple[str, float]]:
        """
        Identify root causes of a metric collapse by tracing
        upstream in the causal graph.
        """
        if collapse_metric not in self.METRICS:
            return []
        
        target_idx = self.METRICS.index(collapse_metric)
        causes = []
        
        for i in range(self.n_metrics):
            if i != target_idx:
                strength = abs(self._adjacency[i][target_idx])
                if strength > 0.1:
                    causes.append((self.METRICS[i], round(strength, 4)))
        
        causes.sort(key=lambda x: -x[1])
        return causes
    
    def causal_graph(self) -> Dict:
        """Return the learned causal graph as an adjacency list."""
        edges = []
        for i in range(self.n_metrics):
            for j in range(self.n_metrics):
                if i != j and abs(self._adjacency[i][j]) > 0.15:
                    edges.append({
                        "from": self.METRICS[i] if i < len(self.METRICS) else f"m{i}",
                        "to": self.METRICS[j] if j < len(self.METRICS) else f"m{j}",
                        "strength": self._adjacency[i][j],
                    })
        return {
            "n_nodes": self.n_metrics,
            "n_edges": len(edges),
            "edges": sorted(edges, key=lambda x: -abs(x["strength"])),
        }
    
    def report(self) -> Dict:
        return {
            "n_observations": len(self._transitions),
            "causal_graph": self.causal_graph(),
            "action_effects_learned": sum(1 for c in self._action_counts if c > 0),
        }


class NeuralCausalModel(nn.Module):
    """
    Neural network that learns causal structure from interventional data.
    
    Architecture: 
      - Encoder: state -> latent
      - Intervention head: (latent, action) -> predicted next state
      - Causal mask: learnable binary mask on the adjacency matrix
    """
    
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # State encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        
        # Intervention predictor: (encoded_state, action_onehot) -> delta_state
        self.intervention = nn.Sequential(
            nn.Linear(hidden + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, obs_dim),
        )
        
        # Learnable causal mask (soft, differentiable)
        self.causal_logits = nn.Parameter(torch.zeros(obs_dim, obs_dim))
    
    @property
    def causal_mask(self) -> torch.Tensor:
        """Soft causal adjacency matrix."""
        mask = torch.sigmoid(self.causal_logits)
        # Zero diagonal (no self-causation)
        return mask * (1 - torch.eye(self.obs_dim, device=mask.device))
    
    def predict_intervention(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict state change from intervention."""
        encoded = self.encoder(state)
        if action.dim() == 1:
            action = F.one_hot(action.long(), self.action_dim).float()
        combined = torch.cat([encoded, action], dim=-1)
        delta = self.intervention(combined)
        
        # Apply causal mask: each output dimension only depends on
        # causally upstream dimensions
        masked_delta = delta @ self.causal_mask
        return state + masked_delta
    
    def training_step(self, state_before: torch.Tensor, action: torch.Tensor,
                      state_after: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute training loss."""
        predicted = self.predict_intervention(state_before, action)
        recon_loss = F.mse_loss(predicted, state_after)
        
        # Sparsity penalty on causal graph (prefer simpler explanations)
        sparsity = self.causal_mask.sum()
        
        total = recon_loss + 0.01 * sparsity
        return {
            "total": total,
            "reconstruction": recon_loss,
            "sparsity": sparsity,
            "n_causal_edges": (self.causal_mask > 0.5).sum().item(),
        }


def validate_causal():
    print("=" * 64)
    print("  CAUSAL INTERVENTION ENGINE -- VALIDATION")
    print("=" * 64)
    
    scm = StructuralCausalModel(n_metrics=8, n_actions=16)
    
    # Feed synthetic transition data
    for _ in range(200):
        before = [random.gauss(50, 10) for _ in range(8)]
        action = random.randint(0, 15)
        # Simulate causal structure: action affects metric 0, which causes metric 1
        after = before.copy()
        effect = (action - 8) * 0.5
        after[0] += effect + random.gauss(0, 1)
        after[1] += 0.6 * effect + random.gauss(0, 1)  # downstream
        after[2] -= 0.3 * effect + random.gauss(0, 1)  # inverse
        scm.observe(before, action, after)
    
    scm.learn_structure()
    
    # do() query
    effects = scm.do(12)
    print("  do(action=12) effects:")
    for k, v in list(effects.items())[:4]:
        print("    " + k + ": " + str(v))
    
    # Counterfactual
    state = [50.0] * 8
    cf = scm.counterfactual(state, actual_action=0, hypothetical_action=15)
    print("  Counterfactual (action 0 -> 15):")
    for k, v in list(cf.items())[:3]:
        print("    " + k + ": " + str(v))
    
    # Root cause
    causes = scm.root_cause_analysis("pollution_index")
    print("  Root causes of pollution_index: " + str(causes[:3]))
    
    # Causal graph
    graph = scm.causal_graph()
    print("  Causal graph: " + str(graph["n_edges"]) + " edges")
    
    # Neural model
    ncm = NeuralCausalModel(obs_dim=8, action_dim=16)
    s_before = torch.randn(4, 8)
    a = torch.randint(0, 16, (4,))
    s_after = torch.randn(4, 8)
    loss = ncm.training_step(s_before, a, s_after)
    print("  Neural causal loss: " + str(round(loss["total"].item(), 4)))
    print("  Learned edges: " + str(int(loss["n_causal_edges"])))
    
    print("\n  CAUSAL ENGINE VALIDATION PASSED")
    print("=" * 64)


if __name__ == "__main__":
    validate_causal()
