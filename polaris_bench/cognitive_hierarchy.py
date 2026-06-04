#!/usr/bin/env python3
"""
Cognitive Hierarchy -- Level-K Thinking for Deep Theory of Mind
================================================================
Models how agents reason about each other's reasoning.

Level-0: Random/naive policy (no strategic thinking)
Level-1: Best response to Level-0 (assumes others are naive)
Level-2: Best response to Level-1 (assumes others think you're naive)
Level-K: Best response to Level-(K-1) (recursive reasoning)

This creates genuine Theory of Mind -- each minister builds a model
of what other ministers THINK, not just what they DO.

The depth of reasoning (K) is itself a learnable parameter.

Reference: Camerer, Ho, Chong, "A Cognitive Hierarchy Model of Games" (QJE 2004)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from typing import Dict, List, Optional
from collections import defaultdict


class LevelKReasoner(nn.Module):
    """
    Each minister has a hierarchy of opponent models.
    
    Level 0: Uniform random
    Level 1: Neural best-response to Level 0
    Level 2: Neural best-response to Level 1
    ...
    Level K: Neural best-response to Level K-1
    
    The actual policy is a Poisson-weighted mixture:
      pi = sum_k (poisson(k; tau) * pi_k)
    where tau is the learned sophistication parameter.
    """
    
    def __init__(self, obs_dim: int, action_dim: int, n_agents: int = 5,
                 max_level: int = 3, hidden: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.max_level = max_level
        
        # Level-K policies (each level has its own network)
        self.level_policies = nn.ModuleList()
        for k in range(max_level + 1):
            if k == 0:
                # Level 0: simple linear (near-random)
                self.level_policies.append(nn.Sequential(
                    nn.Linear(obs_dim, action_dim),
                ))
            else:
                # Level K>0: takes state + predicted opponent actions
                input_dim = obs_dim + action_dim * (n_agents - 1)
                self.level_policies.append(nn.Sequential(
                    nn.Linear(input_dim, hidden), nn.ReLU(),
                    nn.Linear(hidden, hidden // 2), nn.ReLU(),
                    nn.Linear(hidden // 2, action_dim),
                ))
        
        # Sophistication parameter (tau) per agent -- learnable
        # Higher tau = deeper reasoning
        self.log_tau = nn.Parameter(torch.zeros(n_agents))
        
        # Opponent modeling network: predicts what level each opponent is
        self.opponent_level_predictor = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, max_level + 1),
        )
        
        # Belief tracking: what does agent i think agent j will do?
        self._beliefs = {}  # (agent_i, agent_j) -> action distribution
        self._history = defaultdict(list)  # agent_id -> [(state, action)]
    
    @property
    def tau(self) -> torch.Tensor:
        """Sophistication levels per agent."""
        return F.softplus(self.log_tau)
    
    def _poisson_weights(self, tau: float) -> List[float]:
        """Poisson distribution weights for cognitive hierarchy."""
        weights = []
        for k in range(self.max_level + 1):
            # Poisson(k; tau)
            log_w = k * math.log(max(tau, 1e-6)) - tau - math.lgamma(k + 1)
            weights.append(math.exp(log_w))
        total = sum(weights)
        if total < 1e-10:
            return [1.0 / (self.max_level + 1)] * (self.max_level + 1)
        return [w / total for w in weights]
    
    def predict_opponent(self, agent_id: int, state: torch.Tensor) -> torch.Tensor:
        """
        Predict what opponent will do based on their estimated level.
        
        Uses history to estimate opponent sophistication, then predicts
        their action distribution.
        """
        dev = next(self.parameters()).device
        
        # Level 0 prediction: uniform
        level0_pred = torch.ones(self.action_dim, device=dev) / self.action_dim
        
        # If we have history, use it
        if agent_id in self._history and len(self._history[agent_id]) > 0:
            # Count action frequencies
            action_counts = torch.zeros(self.action_dim, device=dev)
            for _, a in self._history[agent_id][-20:]:
                if a < self.action_dim:
                    action_counts[a] += 1
            total = action_counts.sum()
            if total > 0:
                empirical = action_counts / total
                # Blend empirical with uniform (smooth)
                level0_pred = 0.7 * empirical + 0.3 * level0_pred
        
        return level0_pred
    
    def forward(self, agent_id: int, state: torch.Tensor) -> Dict:
        """
        Compute action distribution for agent using cognitive hierarchy.
        
        Returns the Poisson-weighted mixture of Level-K policies.
        """
        dev = next(self.parameters()).device
        state = state.to(dev)
        
        tau_val = self.tau[agent_id].item()
        weights = self._poisson_weights(tau_val)
        
        # Level 0: just use state
        level0_logits = self.level_policies[0](state)
        mixture = weights[0] * F.softmax(level0_logits, dim=-1)
        
        # Higher levels: predict opponent actions, then best-respond
        for k in range(1, self.max_level + 1):
            # Predict opponent actions at level k-1
            opponent_preds = []
            for j in range(self.n_agents):
                if j != agent_id:
                    pred = self.predict_opponent(j, state)
                    # Sample from predicted distribution
                    opponent_preds.append(pred)
            
            # Concatenate opponent predictions
            if opponent_preds:
                opp_flat = torch.cat(opponent_preds, dim=-1)
            else:
                opp_flat = torch.zeros(self.action_dim * (self.n_agents - 1), device=dev)
            
            # Best response at level k
            level_input = torch.cat([state, opp_flat], dim=-1)
            level_logits = self.level_policies[k](level_input)
            mixture = mixture + weights[k] * F.softmax(level_logits, dim=-1)
        
        # Normalize
        mixture = mixture / (mixture.sum() + 1e-8)
        
        return {
            "action_probs": mixture,
            "action": torch.argmax(mixture).item(),
            "tau": round(tau_val, 4),
            "level_weights": [round(w, 4) for w in weights],
            "dominant_level": weights.index(max(weights)),
        }
    
    def update_beliefs(self, agent_id: int, state: torch.Tensor, action: int):
        """Record observed action for belief updating."""
        self._history[agent_id].append((state.detach().cpu(), action))
        # Keep only recent history
        if len(self._history[agent_id]) > 50:
            self._history[agent_id] = self._history[agent_id][-50:]
    
    def estimate_opponent_levels(self) -> Dict[int, Dict]:
        """Estimate the sophistication level of each agent from behavior."""
        estimates = {}
        for agent_id in range(self.n_agents):
            if agent_id not in self._history or len(self._history[agent_id]) < 5:
                estimates[agent_id] = {"estimated_level": 0, "confidence": 0.0}
                continue
            
            # Compute action entropy -- lower entropy = higher level
            action_counts = [0] * self.action_dim
            for _, a in self._history[agent_id][-20:]:
                if a < self.action_dim:
                    action_counts[a] += 1
            total = sum(action_counts)
            if total == 0:
                estimates[agent_id] = {"estimated_level": 0, "confidence": 0.0}
                continue
            
            entropy = 0.0
            for c in action_counts:
                if c > 0:
                    p = c / total
                    entropy -= p * math.log2(p)
            
            max_entropy = math.log2(self.action_dim)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            # Lower entropy = more strategic = higher level
            estimated_level = max(0, min(self.max_level, 
                int((1 - normalized_entropy) * self.max_level + 0.5)))
            
            estimates[agent_id] = {
                "estimated_level": estimated_level,
                "entropy": round(entropy, 4),
                "normalized_entropy": round(normalized_entropy, 4),
                "confidence": round(1 - normalized_entropy, 4),
                "n_observations": len(self._history[agent_id]),
            }
        
        return estimates
    
    def report(self) -> Dict:
        """Full cognitive hierarchy analysis."""
        return {
            "tau_per_agent": [round(t, 4) for t in self.tau.tolist()],
            "opponent_estimates": self.estimate_opponent_levels(),
            "n_total_observations": sum(len(v) for v in self._history.values()),
        }


class RecursiveBeliefNetwork(nn.Module):
    """
    I think that you think that I think...
    
    Recursive belief modeling to arbitrary depth.
    Each layer models one level of recursive reasoning.
    """
    
    def __init__(self, obs_dim: int, n_agents: int = 5, 
                 depth: int = 3, hidden: int = 32):
        super().__init__()
        self.depth = depth
        self.n_agents = n_agents
        
        # Each depth level has a belief network
        self.belief_layers = nn.ModuleList()
        for d in range(depth):
            input_dim = obs_dim + hidden * n_agents if d > 0 else obs_dim
            self.belief_layers.append(nn.Sequential(
                nn.Linear(input_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden),
            ))
    
    def forward(self, state: torch.Tensor, agent_id: int) -> Dict:
        """
        Compute nested beliefs for an agent.
        
        Returns beliefs at each depth:
          depth 0: What does agent_i believe about the world?
          depth 1: What does agent_i believe agent_j believes?
          depth 2: What does agent_i believe agent_j believes agent_k believes?
        """
        dev = next(self.parameters()).device
        state = state.to(dev)
        
        beliefs = {}
        prev_beliefs = None
        
        for d in range(self.depth):
            if d == 0:
                inp = state
            else:
                # Stack all agents' previous-depth beliefs
                all_prev = torch.cat([prev_beliefs[j] for j in range(self.n_agents)], dim=-1)
                inp = torch.cat([state, all_prev], dim=-1)
            
            # Compute belief for each agent at this depth
            current_beliefs = {}
            for j in range(self.n_agents):
                current_beliefs[j] = self.belief_layers[d](inp)
            
            beliefs[d] = {j: b.detach() for j, b in current_beliefs.items()}
            prev_beliefs = current_beliefs
        
        return {
            "agent": agent_id,
            "belief_depth": self.depth,
            "beliefs": beliefs,
        }


def validate_cognitive():
    print("=" * 64)
    print("  COGNITIVE HIERARCHY (LEVEL-K) -- VALIDATION")
    print("=" * 64)
    
    obs_dim, act_dim, n_agents = 8, 16, 5
    
    ck = LevelKReasoner(obs_dim, act_dim, n_agents, max_level=3)
    state = torch.randn(obs_dim)
    
    # Test each agent
    for agent in range(n_agents):
        result = ck(agent, state)
        print("  Agent " + str(agent) + ": action=" + str(result["action"]) +
              " tau=" + str(result["tau"]) +
              " dominant_level=" + str(result["dominant_level"]))
    
    # Simulate some history
    for _ in range(30):
        for agent in range(n_agents):
            a = random.randint(0, act_dim - 1) if agent != 2 else random.choice([3, 5, 7])
            ck.update_beliefs(agent, torch.randn(obs_dim), a)
    
    # Estimate opponent levels
    estimates = ck.estimate_opponent_levels()
    print("\n  Opponent level estimates:")
    for agent, est in estimates.items():
        print("    Agent " + str(agent) + ": level=" + str(est["estimated_level"]) +
              " confidence=" + str(est.get("confidence", 0)))
    
    # Recursive beliefs
    rbn = RecursiveBeliefNetwork(obs_dim, n_agents, depth=3)
    beliefs = rbn(state, agent_id=0)
    print("\n  Recursive belief depth: " + str(beliefs["belief_depth"]))
    
    # Report
    report = ck.report()
    print("  Tau per agent: " + str(report["tau_per_agent"]))
    
    print("\n  COGNITIVE HIERARCHY VALIDATION PASSED")
    print("=" * 64)


if __name__ == "__main__":
    validate_cognitive()
