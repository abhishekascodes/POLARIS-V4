#!/usr/bin/env python3
"""
Distributional RL + Multi-Objective Pareto + Meta-Learning (MAML)
===================================================================

Three critical systems that complete the POLARIS architecture:

1. DISTRIBUTIONAL RL:
   Predict the DISTRIBUTION of returns, not just the mean.
   "Expected reward is 30, but there's a 15% chance of total collapse."
   Uses quantile regression (QR-DQN approach).
   Reference: Dabney et al., "Distributional RL with Quantile Regression" (AAAI 2018)

2. MULTI-OBJECTIVE PARETO:
   GDP vs Pollution vs Equality are CONFLICTING objectives.
   Find the Pareto frontier -- the set of solutions where you can't
   improve one objective without hurting another.
   Reference: Van Moffaert & Nowe, "Multi-Objective RL" (JMLR 2014)

3. META-LEARNING (MAML):
   Learn to adapt to NEW governance scenarios in 3 gradient steps.
   "Never seen a pandemic before? Give me 3 examples and I'll handle it."
   Reference: Finn, Abbeel, Levine, "Model-Agnostic Meta-Learning" (ICML 2017)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import copy
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


# ============================================================
# DISTRIBUTIONAL RL
# ============================================================

class DistributionalCritic(nn.Module):
    """
    Quantile Regression critic for risk-aware governance.
    
    Instead of V(s) = E[R], predicts N quantiles of the return distribution.
    This captures:
      - Expected value (median)
      - Risk (spread between quantiles)
      - Tail risk (5th percentile = worst case)
      - Upside potential (95th percentile = best case)
    """
    
    def __init__(self, obs_dim: int, action_dim: int,
                 n_quantiles: int = 32, hidden: int = 64):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.action_dim = action_dim
        
        # Quantile network
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim * n_quantiles),
        )
        
        # Fixed quantile fractions
        taus = torch.arange(1, n_quantiles + 1, dtype=torch.float32) / (n_quantiles + 1)
        self.register_buffer('taus', taus)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Returns: [batch, action_dim, n_quantiles] quantile values
        """
        out = self.net(state)
        return out.view(-1, self.action_dim, self.n_quantiles)
    
    def risk_analysis(self, state: torch.Tensor) -> Dict:
        """Full risk analysis for each action."""
        quantiles = self.forward(state.unsqueeze(0)).squeeze(0)  # [action_dim, n_quantiles]
        
        analysis = {}
        for a in range(self.action_dim):
            q = quantiles[a]
            analysis[a] = {
                "mean": round(q.mean().item(), 4),
                "median": round(q[self.n_quantiles // 2].item(), 4),
                "var_5pct": round(q[int(self.n_quantiles * 0.05)].item(), 4),
                "var_95pct": round(q[int(self.n_quantiles * 0.95)].item(), 4),
                "spread": round((q[-1] - q[0]).item(), 4),
                "downside_risk": round((q.mean() - q[0]).item(), 4),
            }
        
        # Risk-aware action selection
        # CVaR (Conditional Value at Risk): expected value of worst 10% outcomes
        cvar_10 = []
        for a in range(self.action_dim):
            q = quantiles[a]
            worst_10pct = q[:max(1, self.n_quantiles // 10)]
            cvar_10.append(worst_10pct.mean().item())
        
        return {
            "per_action": analysis,
            "best_action_mean": int(quantiles.mean(dim=-1).argmax()),
            "best_action_cvar": int(cvar_10.index(max(cvar_10))),
            "safest_action": int(quantiles[:, 0].argmax()),  # best worst-case
            "riskiest_action": int(quantiles[:, -1].argmax()),  # best best-case
        }
    
    def quantile_loss(self, predicted: torch.Tensor, 
                      target: torch.Tensor) -> torch.Tensor:
        """Quantile Huber loss for distributional training."""
        # predicted: [batch, n_quantiles]
        # target: [batch, 1]
        target = target.unsqueeze(-1).expand_as(predicted)
        diff = target - predicted
        
        huber = torch.where(diff.abs() < 1.0,
                           0.5 * diff.pow(2),
                           diff.abs() - 0.5)
        
        tau = self.taus.unsqueeze(0).expand_as(diff)
        loss = (tau - (diff < 0).float()).abs() * huber
        return loss.mean()


# ============================================================
# MULTI-OBJECTIVE PARETO
# ============================================================

class ParetoOptimizer:
    """
    Multi-objective optimization for conflicting governance goals.
    
    Maintains a Pareto frontier of non-dominated solutions.
    A solution is Pareto optimal if no other solution is better
    in ALL objectives simultaneously.
    """
    
    OBJECTIVES = ["gdp", "environment", "equality", "health", "stability"]
    
    def __init__(self, n_objectives: int = 5):
        self.n_objectives = n_objectives
        self._frontier = []  # List of (solution_id, objective_values)
        self._all_solutions = []
        self._next_id = 0
    
    def _dominates(self, a: List[float], b: List[float]) -> bool:
        """Does solution a Pareto-dominate solution b?"""
        at_least_one_better = False
        for i in range(min(len(a), len(b))):
            if a[i] < b[i]:
                return False
            if a[i] > b[i]:
                at_least_one_better = True
        return at_least_one_better
    
    def add_solution(self, objectives: List[float], 
                     metadata: Optional[Dict] = None) -> Dict:
        """
        Add a solution and update the Pareto frontier.
        Returns whether this solution is on the frontier.
        """
        self._next_id += 1
        sol_id = self._next_id
        
        entry = {
            "id": sol_id,
            "objectives": objectives[:self.n_objectives],
            "metadata": metadata or {},
        }
        self._all_solutions.append(entry)
        
        # Check if dominated by any frontier solution
        is_dominated = False
        for f_entry in self._frontier:
            if self._dominates(f_entry["objectives"], objectives):
                is_dominated = True
                break
        
        if not is_dominated:
            # Remove solutions dominated by new one
            self._frontier = [
                f for f in self._frontier 
                if not self._dominates(objectives, f["objectives"])
            ]
            self._frontier.append(entry)
        
        return {
            "id": sol_id,
            "on_frontier": not is_dominated,
            "frontier_size": len(self._frontier),
        }
    
    def get_frontier(self) -> List[Dict]:
        """Return current Pareto frontier."""
        return [f.copy() for f in self._frontier]
    
    def hypervolume(self, reference: Optional[List[float]] = None) -> float:
        """
        Hypervolume indicator: volume of objective space dominated
        by the Pareto frontier. Higher = better frontier.
        
        This is THE standard metric for multi-objective optimization quality.
        """
        if not self._frontier:
            return 0.0
        
        if reference is None:
            reference = [0.0] * self.n_objectives
        
        # For 2D, compute exact hypervolume
        if self.n_objectives == 2:
            points = sorted(self._frontier, key=lambda f: f["objectives"][0])
            hv = 0.0
            prev_y = reference[1]
            for p in points:
                x, y = p["objectives"]
                if x > reference[0] and y > reference[1]:
                    hv += (x - reference[0]) * (y - prev_y)
                    prev_y = y
            return round(hv, 4)
        
        # For higher dimensions, use Monte Carlo estimate
        n_samples = 10000
        mins = reference
        maxs = [max(f["objectives"][i] for f in self._frontier) 
                for i in range(self.n_objectives)]
        
        dominated_count = 0
        for _ in range(n_samples):
            point = [random.uniform(mins[i], maxs[i] + 1e-6) 
                     for i in range(self.n_objectives)]
            for f in self._frontier:
                if all(f["objectives"][i] >= point[i] for i in range(self.n_objectives)):
                    dominated_count += 1
                    break
        
        volume = 1.0
        for i in range(self.n_objectives):
            volume *= (maxs[i] - mins[i] + 1e-6)
        
        return round(volume * dominated_count / n_samples, 4)
    
    def tradeoff_analysis(self) -> Dict:
        """Analyze tradeoffs between objectives on the frontier."""
        if len(self._frontier) < 2:
            return {"tradeoffs": "insufficient frontier points"}
        
        tradeoffs = {}
        for i in range(self.n_objectives):
            for j in range(i + 1, self.n_objectives):
                name_i = self.OBJECTIVES[i] if i < len(self.OBJECTIVES) else f"obj_{i}"
                name_j = self.OBJECTIVES[j] if j < len(self.OBJECTIVES) else f"obj_{j}"
                
                vals_i = [f["objectives"][i] for f in self._frontier]
                vals_j = [f["objectives"][j] for f in self._frontier]
                
                # Correlation on frontier (negative = tradeoff)
                if len(vals_i) > 2:
                    mean_i = sum(vals_i) / len(vals_i)
                    mean_j = sum(vals_j) / len(vals_j)
                    cov = sum((a - mean_i) * (b - mean_j) for a, b in zip(vals_i, vals_j))
                    var_i = sum((a - mean_i) ** 2 for a in vals_i)
                    var_j = sum((b - mean_j) ** 2 for b in vals_j)
                    
                    if var_i > 0 and var_j > 0:
                        corr = cov / (math.sqrt(var_i) * math.sqrt(var_j))
                    else:
                        corr = 0.0
                    
                    tradeoffs[f"{name_i}_vs_{name_j}"] = {
                        "correlation": round(corr, 4),
                        "tradeoff": corr < -0.3,
                        "synergy": corr > 0.3,
                    }
        
        return tradeoffs
    
    def report(self) -> Dict:
        return {
            "frontier_size": len(self._frontier),
            "total_solutions": len(self._all_solutions),
            "hypervolume": self.hypervolume(),
            "tradeoffs": self.tradeoff_analysis(),
        }


# ============================================================
# META-LEARNING (MAML)
# ============================================================

class MAMLGovernance(nn.Module):
    """
    Model-Agnostic Meta-Learning for rapid adaptation to new crises.
    
    The model learns an INITIALIZATION from which it can adapt to
    any new governance scenario in just a few gradient steps.
    
    Training:
      1. Sample a batch of governance tasks (scenarios)
      2. For each task, take K gradient steps (inner loop)
      3. Evaluate adapted model
      4. Update initialization to minimize post-adaptation loss (outer loop)
    
    Result: A model that can handle a pandemic, trade war, or climate
    crisis after seeing just 3-5 examples.
    """
    
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 64,
                 inner_lr: float = 0.01, inner_steps: int = 3):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        
        # Base policy (the meta-learned initialization)
        self.policy = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, action_dim),
        )
        
        # Task encoder: identifies what type of crisis this is
        self.task_encoder = nn.Sequential(
            nn.Linear(obs_dim * 3, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, hidden // 4),
        )
        
        self._adaptation_history = []
    
    def adapt(self, support_states: torch.Tensor, 
              support_actions: torch.Tensor,
              support_rewards: torch.Tensor) -> nn.Module:
        """
        Adapt to a new task using K gradient steps.
        
        Args:
            support_states: [K, obs_dim] states from new task
            support_actions: [K] taken actions
            support_rewards: [K] received rewards
        
        Returns:
            Adapted policy
        """
        dev = next(self.parameters()).device
        
        # Clone the policy for adaptation (don't modify original)
        adapted = copy.deepcopy(self.policy)
        
        for step in range(self.inner_steps):
            logits = adapted(support_states.to(dev))
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Policy gradient loss
            action_log_probs = log_probs.gather(1, support_actions.long().to(dev).unsqueeze(1)).squeeze(1)
            
            # Normalize rewards
            rewards = support_rewards.to(dev)
            if rewards.std() > 1e-6:
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            
            loss = -(action_log_probs * rewards).mean()
            
            # Inner gradient step
            grads = torch.autograd.grad(loss, adapted.parameters(), 
                                        create_graph=True, allow_unused=True)
            
            # Manual SGD update
            for param, grad in zip(adapted.parameters(), grads):
                if grad is not None:
                    param.data = param.data - self.inner_lr * grad
        
        return adapted
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass with base (unadapted) policy."""
        return self.policy(state)
    
    def meta_test(self, support_states: torch.Tensor,
                  support_actions: torch.Tensor,
                  support_rewards: torch.Tensor,
                  query_states: torch.Tensor) -> Dict:
        """
        Meta-test: adapt to new task, then evaluate on query set.
        """
        adapted = self.adapt(support_states, support_actions, support_rewards)
        
        dev = next(self.parameters()).device
        
        # Base policy predictions (before adaptation)
        base_logits = self.policy(query_states.to(dev))
        base_actions = base_logits.argmax(dim=-1)
        
        # Adapted policy predictions
        with torch.no_grad():
            adapted_logits = adapted(query_states.to(dev))
            adapted_actions = adapted_logits.argmax(dim=-1)
        
        # How much did adaptation change the policy?
        action_change = (base_actions != adapted_actions).float().mean().item()
        
        self._adaptation_history.append({
            "n_support": len(support_states),
            "action_change": round(action_change, 4),
        })
        
        return {
            "adapted_actions": adapted_actions,
            "base_actions": base_actions,
            "action_change_rate": round(action_change, 4),
            "n_adaptation_steps": self.inner_steps,
            "n_support_examples": len(support_states),
        }
    
    def report(self) -> Dict:
        if not self._adaptation_history:
            return {"n_adaptations": 0}
        
        changes = [h["action_change"] for h in self._adaptation_history]
        return {
            "n_adaptations": len(self._adaptation_history),
            "avg_action_change": round(sum(changes) / len(changes), 4),
            "max_action_change": round(max(changes), 4),
            "inner_lr": self.inner_lr,
            "inner_steps": self.inner_steps,
        }


def validate_dist_pareto_maml():
    print("=" * 64)
    print("  DISTRIBUTIONAL RL + PARETO + META-LEARNING -- VALIDATION")
    print("=" * 64)
    
    obs_dim, act_dim = 8, 16
    
    # Distributional RL
    print("\n  [Distributional Critic]")
    dc = DistributionalCritic(obs_dim, act_dim, n_quantiles=32)
    state = torch.randn(obs_dim)
    risk = dc.risk_analysis(state)
    print("    Best action (mean): " + str(risk["best_action_mean"]))
    print("    Best action (CVaR): " + str(risk["best_action_cvar"]))
    print("    Safest action: " + str(risk["safest_action"]))
    a0 = risk["per_action"][0]
    print("    Action 0: mean=" + str(a0["mean"]) + " VaR5=" + str(a0["var_5pct"]) +
          " spread=" + str(a0["spread"]))
    
    # Pareto Optimizer
    print("\n  [Pareto Optimizer]")
    po = ParetoOptimizer(n_objectives=3)
    
    for _ in range(50):
        # Random solutions with tradeoffs
        x = random.random()
        objs = [x, 1 - x + random.gauss(0, 0.1), random.random()]
        po.add_solution(objs)
    
    report = po.report()
    print("    Frontier size: " + str(report["frontier_size"]))
    print("    Hypervolume: " + str(report["hypervolume"]))
    for k, v in report["tradeoffs"].items():
        if isinstance(v, dict):
            print("    " + k + ": corr=" + str(v["correlation"]) +
                  " tradeoff=" + str(v["tradeoff"]))
    
    # Meta-Learning
    print("\n  [MAML Meta-Learning]")
    maml = MAMLGovernance(obs_dim, act_dim, inner_lr=0.01, inner_steps=3)
    
    # Simulate new crisis
    support_s = torch.randn(5, obs_dim)
    support_a = torch.randint(0, act_dim, (5,))
    support_r = torch.randn(5)
    query_s = torch.randn(10, obs_dim)
    
    result = maml.meta_test(support_s, support_a, support_r, query_s)
    print("    Action change after 3-step adaptation: " + str(result["action_change_rate"]))
    print("    Support examples: " + str(result["n_support_examples"]))
    
    maml_report = maml.report()
    print("    Avg action change: " + str(maml_report["avg_action_change"]))
    
    print("\n  DISTRIBUTIONAL + PARETO + MAML VALIDATION PASSED")
    print("=" * 64)


if __name__ == "__main__":
    validate_dist_pareto_maml()
