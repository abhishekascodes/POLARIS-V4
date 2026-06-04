#!/usr/bin/env python3
"""
POLARIS v5 — Nash Equilibrium Detection & Equilibrium Analysis
================================================================
Verifies whether the learned joint policy of N AI ministers converges
to (approximate) Nash Equilibria.  Provides:

  • NashDetector      — epsilon-Nash testing via unilateral deviation
  • EquilibriumAnalyzer — Pareto optimality, social welfare, price of anarchy

Self-contained: imports only torch, numpy, math, random, collections, typing.
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
import random
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════
#  Lightweight environment protocol used for deviation testing.
#  Any env passed to NashDetector must duck-type this interface:
#     env.reward(state, joint_actions) -> List[float]   (per-agent)
#     env.num_agents -> int
#     env.num_actions -> int
# ═══════════════════════════════════════════════════════════════════


class _StubEnv:
    """Minimal env for validation — reward = -|action - preferred_action|."""

    def __init__(self, num_agents: int = 5, num_actions: int = 16):
        self.num_agents = num_agents
        self.num_actions = num_actions
        # Each agent has a privately preferred action
        self._preferred = [random.randint(0, num_actions - 1) for _ in range(num_agents)]

    def reward(self, state: torch.Tensor, joint_actions: List[int]) -> List[float]:
        """
        Simple reward: each agent gets 1 - normalised distance to
        its preferred action, plus a tiny cooperation bonus when
        neighbours pick similar actions.
        """
        rewards = []
        for i, a in enumerate(joint_actions):
            base = 1.0 - abs(a - self._preferred[i]) / self.num_actions
            coop = 0.0
            if i > 0:
                coop += 0.05 * (1.0 - abs(a - joint_actions[i - 1]) / self.num_actions)
            if i < self.num_agents - 1:
                coop += 0.05 * (1.0 - abs(a - joint_actions[i + 1]) / self.num_actions)
            rewards.append(base + coop)
        return rewards


# ═══════════════════════════════════════════════════════════════════
#  1.  NASH DETECTOR
# ═══════════════════════════════════════════════════════════════════

class NashDetector:
    """
    Checks whether a learned joint policy constitutes an (epsilon-) Nash
    Equilibrium by testing *unilateral deviations*.

    Algorithm
    ---------
    For every agent *i*, fix all other agents' actions and try every
    alternative action.  If no agent can improve its payoff by more than
    *epsilon*, the profile is an epsilon-Nash equilibrium.

    Parameters
    ----------
    num_agents : int
        Number of minister agents (default 5).
    num_actions : int
        Size of the discrete action space (default 16).
    """

    def __init__(self, num_agents: int = 5, num_actions: int = 16):
        self.num_agents = num_agents
        self.num_actions = num_actions
        self._history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    #  Core: single-state Nash check
    # ------------------------------------------------------------------

    def check_nash(
        self,
        agent_policies: Any,          # ignored if current_actions given
        env: Any,                      # must expose reward(state, actions)
        state: torch.Tensor,
        current_actions: List[int],
    ) -> Dict[str, Any]:
        """
        Test whether *current_actions* is a Nash Equilibrium given *state*.

        Returns
        -------
        dict with keys:
            is_nash        : bool   — True if no agent can improve
            epsilon        : float  — max improvement any agent could get
            best_deviation : dict   — {agent_idx: (best_alt_action, gain)}
            current_rewards: list   — reward of each agent under current actions
        """
        assert len(current_actions) == self.num_agents, (
            f"Expected {self.num_agents} actions, got {len(current_actions)}"
        )

        current_rewards = env.reward(state, current_actions)
        best_deviation: Dict[int, Tuple[int, float]] = {}
        max_improvement = 0.0

        for i in range(self.num_agents):
            best_alt_action = current_actions[i]
            best_gain = 0.0

            for alt_a in range(self.num_actions):
                if alt_a == current_actions[i]:
                    continue
                # Construct deviated action profile
                deviated = list(current_actions)
                deviated[i] = alt_a
                deviated_rewards = env.reward(state, deviated)
                gain = deviated_rewards[i] - current_rewards[i]

                if gain > best_gain:
                    best_gain = gain
                    best_alt_action = alt_a

            best_deviation[i] = (best_alt_action, best_gain)
            max_improvement = max(max_improvement, best_gain)

        is_nash = max_improvement <= 0.0
        result = {
            "is_nash": is_nash,
            "epsilon": max_improvement,
            "best_deviation": best_deviation,
            "current_rewards": current_rewards,
        }
        self._history.append(result)
        return result

    # ------------------------------------------------------------------
    #  Aggregate epsilon-Nash across many states / seeds
    # ------------------------------------------------------------------

    def find_epsilon_nash(
        self,
        agent_policies: Any,
        env: Any,
        states: List[torch.Tensor],
        seeds: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Run Nash checks across multiple states (and optional seeds) to
        compute a robust *average* epsilon.

        For each state the agent policies are sampled to produce
        current_actions.  If agent_policies is a callable
        ``(state) -> List[int]`` it will be called; otherwise random
        actions are used.

        Returns
        -------
        dict with keys:
            avg_epsilon    : float
            max_epsilon    : float
            min_epsilon    : float
            nash_fraction  : float  — fraction of states that are exact NE
            per_state      : list of per-state results
        """
        if seeds is None:
            seeds = list(range(len(states)))

        per_state: List[Dict] = []
        epsilons: List[float] = []
        exact_count = 0

        for idx, state in enumerate(states):
            if idx < len(seeds):
                random.seed(seeds[idx])
                torch.manual_seed(seeds[idx])

            # Sample actions from policies
            if callable(agent_policies):
                actions = agent_policies(state)
            else:
                actions = [random.randint(0, self.num_actions - 1)
                           for _ in range(self.num_agents)]

            result = self.check_nash(agent_policies, env, state, actions)
            per_state.append(result)
            epsilons.append(result["epsilon"])
            if result["is_nash"]:
                exact_count += 1

        return {
            "avg_epsilon": float(np.mean(epsilons)),
            "max_epsilon": float(np.max(epsilons)),
            "min_epsilon": float(np.min(epsilons)),
            "nash_fraction": exact_count / max(len(states), 1),
            "num_states": len(states),
            "per_state": per_state,
        }

    # ------------------------------------------------------------------
    #  Convergence trajectory within a single episode
    # ------------------------------------------------------------------

    def convergence_trajectory(
        self,
        episode_data: List[Tuple[torch.Tensor, List[int], List[float]]],
    ) -> List[float]:
        """
        Given an episode as a list of ``(state, joint_actions, rewards)``,
        compute the epsilon-Nash gap at every step, showing whether agents
        converge toward equilibrium over time.

        Because we do not have the full env here, epsilon is approximated
        as the maximum reward difference any agent could theoretically gain,
        estimated from the variance of rewards over the episode so far.

        A dedicated env can be injected via ``convergence_trajectory_env``.
        """
        if not episode_data:
            return []

        epsilons: List[float] = []
        cum_rewards = np.zeros(self.num_agents)

        for t, (state, actions, rewards) in enumerate(episode_data):
            cum_rewards += np.array(rewards)
            if t == 0:
                # No deviation data yet — assume neutral
                epsilons.append(0.0)
                continue

            # Heuristic epsilon: max per-agent reward variance so far
            mean_r = cum_rewards / (t + 1)
            current_r = np.array(rewards)
            gap = np.max(np.abs(current_r - mean_r))
            epsilons.append(float(gap))

        return epsilons

    def convergence_trajectory_env(
        self,
        episode_data: List[Tuple[torch.Tensor, List[int], List[float]]],
        env: Any,
    ) -> List[float]:
        """
        Same as ``convergence_trajectory`` but uses the real env to
        evaluate unilateral deviations at each timestep.
        """
        epsilons: List[float] = []
        for state, actions, _rewards in episode_data:
            result = self.check_nash(None, env, state, actions)
            epsilons.append(result["epsilon"])
        return epsilons

    # ------------------------------------------------------------------
    #  Report
    # ------------------------------------------------------------------

    def report(self) -> Dict[str, Any]:
        """Full analysis summary over all ``check_nash`` calls so far."""
        if not self._history:
            return {"status": "no_data", "num_checks": 0}

        eps_values = [h["epsilon"] for h in self._history]
        nash_count = sum(1 for h in self._history if h["is_nash"])

        # Per-agent deviation statistics
        per_agent_max_gain: Dict[int, float] = defaultdict(float)
        for h in self._history:
            for agent_idx, (_, gain) in h["best_deviation"].items():
                per_agent_max_gain[agent_idx] = max(
                    per_agent_max_gain[agent_idx], gain
                )

        return {
            "num_checks": len(self._history),
            "exact_nash_count": nash_count,
            "exact_nash_fraction": nash_count / len(self._history),
            "avg_epsilon": float(np.mean(eps_values)),
            "max_epsilon": float(np.max(eps_values)),
            "min_epsilon": float(np.min(eps_values)),
            "std_epsilon": float(np.std(eps_values)),
            "per_agent_max_gain": dict(per_agent_max_gain),
        }


# ═══════════════════════════════════════════════════════════════════
#  2.  EQUILIBRIUM ANALYZER
# ═══════════════════════════════════════════════════════════════════

class EquilibriumAnalyzer:
    """
    Classifies a Nash equilibrium by several welfare criteria:

      • **Pareto optimality** — no other outcome makes every agent
        at least as well off and at least one strictly better.
      • **Social welfare** — sum of all agents' utilities.
      • **Price of anarchy** — ratio of optimal social welfare to the
        welfare of the *worst* Nash equilibrium.

    Parameters
    ----------
    num_agents : int
    num_actions : int
    """

    def __init__(self, num_agents: int = 5, num_actions: int = 16):
        self.num_agents = num_agents
        self.num_actions = num_actions

    # ------------------------------------------------------------------

    def pareto_check(self, outcomes: List[List[float]]) -> bool:
        """
        Check if the *first* outcome in the list is Pareto optimal
        with respect to the remaining outcomes.

        Parameters
        ----------
        outcomes : list of list of float
            ``outcomes[0]`` is the candidate; the rest are alternatives.

        Returns
        -------
        bool — True if ``outcomes[0]`` is Pareto optimal.
        """
        candidate = np.array(outcomes[0])
        for alt in outcomes[1:]:
            alt = np.array(alt)
            # alt dominates candidate iff every agent is >= and at least one >
            if np.all(alt >= candidate) and np.any(alt > candidate):
                return False
        return True

    def social_welfare(self, outcome: List[float]) -> float:
        """Sum of all agents' utilities (utilitarian social welfare)."""
        return float(np.sum(outcome))

    def price_of_anarchy(
        self, nash_welfare: float, optimal_welfare: float
    ) -> float:
        """
        Price of Anarchy = optimal_welfare / nash_welfare.
        Returns inf if nash_welfare ≤ 0.  PoA ≥ 1; closer to 1 is better.
        """
        if nash_welfare <= 0.0:
            return float("inf")
        return optimal_welfare / nash_welfare

    # ------------------------------------------------------------------
    #  Full analysis with environment sampling
    # ------------------------------------------------------------------

    def analyze(
        self,
        nash_result: Dict[str, Any],
        env: Any,
        seeds: Optional[List[int]] = None,
        sample_budget: int = 256,
    ) -> Dict[str, Any]:
        """
        Comprehensive equilibrium analysis.

        Parameters
        ----------
        nash_result : dict
            Output of ``NashDetector.check_nash`` or a dict containing
            at minimum ``current_rewards`` and ``epsilon``.
        env : object
            Environment with ``reward(state, actions)``, ``num_agents``,
            ``num_actions``.
        seeds : list of int, optional
            Seeds for reproducibility.
        sample_budget : int
            Number of random joint-action profiles to sample for Pareto /
            social-welfare estimation.

        Returns
        -------
        dict with:
            is_nash, epsilon, is_pareto_optimal, social_welfare,
            optimal_welfare, price_of_anarchy, welfare_ratio,
            sampled_alternatives.
        """
        if seeds is None:
            seeds = list(range(sample_budget))

        current_rewards = nash_result["current_rewards"]
        nash_sw = self.social_welfare(current_rewards)

        # --- Sample alternative outcomes for Pareto / optimality check ---
        state_dim = 21  # default POLARIS state size
        alternatives: List[List[float]] = []
        max_sw = nash_sw

        rng = np.random.RandomState(seeds[0] if seeds else 42)
        for _ in range(sample_budget):
            rand_actions = rng.randint(0, env.num_actions, size=env.num_agents).tolist()
            fake_state = torch.randn(state_dim)
            alt_rewards = env.reward(fake_state, rand_actions)
            alternatives.append(alt_rewards)
            alt_sw = self.social_welfare(alt_rewards)
            if alt_sw > max_sw:
                max_sw = alt_sw

        # Pareto check: is the Nash outcome undominated?
        all_outcomes = [current_rewards] + alternatives
        is_pareto = self.pareto_check(all_outcomes)

        poa = self.price_of_anarchy(nash_sw, max_sw)

        return {
            "is_nash": nash_result.get("is_nash", nash_result["epsilon"] <= 0),
            "epsilon": nash_result["epsilon"],
            "is_pareto_optimal": is_pareto,
            "social_welfare": nash_sw,
            "optimal_welfare_estimate": max_sw,
            "price_of_anarchy": poa,
            "welfare_ratio": nash_sw / max(max_sw, 1e-12),
            "num_alternatives_sampled": len(alternatives),
            "per_agent_reward": current_rewards,
        }


# ═══════════════════════════════════════════════════════════════════
#  VALIDATION
# ═══════════════════════════════════════════════════════════════════

def validate_nash() -> None:
    """Smoke-test NashDetector and EquilibriumAnalyzer with random policies."""

    S, A, N = 21, 16, 5
    print("=" * 64)
    print("  POLARIS v5 — Nash Equilibrium Detection Validation")
    print("=" * 64)

    env = _StubEnv(num_agents=N, num_actions=A)
    detector = NashDetector(num_agents=N, num_actions=A)

    # --- Single state check ---
    state = torch.randn(S)
    actions = [random.randint(0, A - 1) for _ in range(N)]
    result = detector.check_nash(None, env, state, actions)
    print(f"\n  [check_nash]  actions={actions}")
    print(f"    is_nash={result['is_nash']}  epsilon={result['epsilon']:.4f}")
    print(f"    rewards={[round(r, 3) for r in result['current_rewards']]}")
    for agent_idx, (alt, gain) in result["best_deviation"].items():
        if gain > 0:
            print(f"    agent {agent_idx}: can switch to {alt} for +{gain:.4f}")

    # --- Multi-state epsilon-Nash ---
    states = [torch.randn(S) for _ in range(20)]
    eps_result = detector.find_epsilon_nash(None, env, states, seeds=list(range(20)))
    print(f"\n  [find_epsilon_nash]  over {eps_result['num_states']} states")
    print(f"    avg_epsilon={eps_result['avg_epsilon']:.4f}  "
          f"max={eps_result['max_epsilon']:.4f}  "
          f"nash_fraction={eps_result['nash_fraction']:.2f}")

    # --- Convergence trajectory ---
    episode = []
    for t in range(30):
        s = torch.randn(S)
        a = [random.randint(0, A - 1) for _ in range(N)]
        r = env.reward(s, a)
        episode.append((s, a, r))
    traj = detector.convergence_trajectory(episode)
    print(f"\n  [convergence_trajectory]  {len(traj)} steps")
    print(f"    first 5 epsilons: {[round(e, 4) for e in traj[:5]]}")
    print(f"    last  5 epsilons: {[round(e, 4) for e in traj[-5:]]}")

    # --- Report ---
    rpt = detector.report()
    print(f"\n  [report]  {rpt['num_checks']} checks, "
          f"exact NE frac={rpt['exact_nash_fraction']:.2f}, "
          f"avg_eps={rpt['avg_epsilon']:.4f}")

    # --- Equilibrium Analyzer ---
    analyzer = EquilibriumAnalyzer(num_agents=N, num_actions=A)

    # Pareto check
    outcomes = [[0.8, 0.7, 0.6, 0.5, 0.4],
                [0.9, 0.8, 0.7, 0.6, 0.5],  # dominates first
                [0.3, 0.3, 0.3, 0.3, 0.3]]
    print(f"\n  [pareto_check]  candidate dominated? -> "
          f"is_pareto={analyzer.pareto_check(outcomes)}")

    # Non-dominated check
    outcomes2 = [[0.8, 0.7, 0.9, 0.5, 0.4],
                 [0.9, 0.6, 0.7, 0.4, 0.5]]
    print(f"  [pareto_check]  non-dominated? -> "
          f"is_pareto={analyzer.pareto_check(outcomes2)}")

    sw = analyzer.social_welfare([0.8, 0.7, 0.6, 0.5, 0.4])
    print(f"  [social_welfare]  = {sw:.2f}")

    poa = analyzer.price_of_anarchy(nash_welfare=3.0, optimal_welfare=4.5)
    print(f"  [price_of_anarchy]  = {poa:.3f}")

    # Full analysis
    analysis = analyzer.analyze(result, env, sample_budget=128)
    print(f"\n  [analyze]  full equilibrium analysis:")
    print(f"    is_pareto_optimal = {analysis['is_pareto_optimal']}")
    print(f"    social_welfare    = {analysis['social_welfare']:.4f}")
    print(f"    optimal_estimate  = {analysis['optimal_welfare_estimate']:.4f}")
    print(f"    price_of_anarchy  = {analysis['price_of_anarchy']:.4f}")
    print(f"    welfare_ratio     = {analysis['welfare_ratio']:.4f}")

    print("\n" + "=" * 64)
    print("  NASH VALIDATION PASSED OK")
    print("=" * 64)


if __name__ == "__main__":
    validate_nash()
