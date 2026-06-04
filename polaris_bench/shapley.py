#!/usr/bin/env python3
"""
POLARIS v5 — Shapley Value Credit Assignment
==============================================
Exact Shapley value computation for N=5 agent governance.

With N=5 there are 2^5 = 32 coalitions and 5! = 120 permutations,
which is entirely tractable for exact computation.

Modules:
  1. ShapleyCredit  — Exact Shapley values for a single step
  2. ShapleyTracker — Running Shapley attribution across an episode

Self-contained.  Imports: torch, math, itertools, collections, random, statistics, typing.
"""

import torch
import math
import itertools
import random
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple
import statistics as stats_lib

# ─── Global constants ────────────────────────────────────────
N_AGENTS: int = 5
STATE_DIM: int = 21
ACTION_DIM: int = 16

# Indices into state vector (matching env conventions)
_GDP_IDX = 0
_POLLUTION_IDX = 1
_SATISFACTION_IDX = 2
_HEALTHCARE_IDX = 3
_EDUCATION_IDX = 4
_UNEMPLOYMENT_IDX = 5

# The "no-op" action used when an agent is NOT in the active coalition
NO_ACTION: int = 0


# ═══════════════════════════════════════════════════════════════
# LIGHTWEIGHT ENVIRONMENT FOR SELF-CONTAINED TESTING
# ═══════════════════════════════════════════════════════════════

class _MiniEnv:
    """
    Minimal deterministic governance environment.
    Given a 21-d state and a list of N action indices, produces the
    next state and a scalar reward.
    """

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM,
                 n_agents: int = N_AGENTS):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.state = torch.zeros(state_dim)

    def reset(self, seed: int = 0) -> torch.Tensor:
        rng = random.Random(seed)
        self.state = torch.tensor(
            [100.0, 100.0, 60.0, 50.0, 50.0, 8.0]
            + [rng.uniform(-1, 1) for _ in range(self.state_dim - 6)],
            dtype=torch.float32,
        )
        return self.state.clone()

    def simulate(self, state: torch.Tensor, actions: List[int]) -> Tuple[torch.Tensor, float]:
        """
        Pure function: given state + joint actions, return (next_state, reward).
        Does NOT mutate self.state — safe for counterfactual evaluation.
        """
        delta = torch.zeros(self.state_dim)
        for a in actions:
            torch.manual_seed(a * 137 + 7)
            delta += 0.25 * torch.randn(self.state_dim)
        next_state = state + delta
        reward = (
            0.01 * next_state[_GDP_IDX].item()
            - 0.005 * next_state[_POLLUTION_IDX].item()
            + 0.01 * next_state[_SATISFACTION_IDX].item()
            - 0.02 * next_state[_UNEMPLOYMENT_IDX].item()
        )
        return next_state, reward


# ═══════════════════════════════════════════════════════════════
# 1. SHAPLEY CREDIT
# ═══════════════════════════════════════════════════════════════

class ShapleyCredit:
    """
    Exact Shapley value computation for N agents.

    For each outcome we enumerate all 2^N coalitions and compute each
    agent's marginal contribution using the classic formula:

        phi_i = (1 / N!) * sum_{pi} [ v(S_pi^i + {i}) - v(S_pi^i) ]

    where S_pi^i is the set of agents preceding i in permutation pi.

    For N=5 this means 120 permutations x 5 agents = 600 marginal
    contribution evaluations (with caching of coalition values, only
    32 unique coalition evaluations are needed).

    Parameters
    ----------
    n_agents : int
        Number of agents (default 5).
    no_action : int
        Action index used for inactive agents in a coalition.
    """

    def __init__(self, n_agents: int = N_AGENTS, no_action: int = NO_ACTION):
        self.n_agents = n_agents
        self.no_action = no_action
        # Pre-compute all permutations
        self._perms: List[Tuple[int, ...]] = list(itertools.permutations(range(n_agents)))

    def _coalition_key(self, coalition: frozenset) -> frozenset:
        return coalition

    def _evaluate_coalition(
        self,
        coalition: frozenset,
        env_step_fn: Callable[[List[int]], float],
        agent_actions: List[int],
    ) -> float:
        """
        Evaluate the coalition value v(S).

        Active agents (in *coalition*) use their chosen action; inactive
        agents use ``no_action``.
        """
        joint = [
            agent_actions[i] if i in coalition else self.no_action
            for i in range(self.n_agents)
        ]
        return env_step_fn(joint)

    def compute(
        self,
        env_step_fn: Callable[[List[int]], float],
        agent_actions: List[int],
    ) -> Dict[int, float]:
        """
        Compute exact Shapley values for a single step.

        Parameters
        ----------
        env_step_fn : callable
            ``f(actions: List[int]) -> float``  — returns the reward/value
            for a joint action vector.  Inactive agents should have action
            ``self.no_action``.
        agent_actions : list[int]
            The actual action each agent chose (length N).

        Returns
        -------
        dict  {agent_id: shapley_value}
        """
        assert len(agent_actions) == self.n_agents

        # Cache coalition values to avoid redundant evaluations
        cache: Dict[frozenset, float] = {}

        def v(coalition: frozenset) -> float:
            if coalition not in cache:
                cache[coalition] = self._evaluate_coalition(
                    coalition, env_step_fn, agent_actions
                )
            return cache[coalition]

        shapley: Dict[int, float] = {i: 0.0 for i in range(self.n_agents)}
        n_fact = math.factorial(self.n_agents)

        for perm in self._perms:
            predecessors: frozenset = frozenset()
            for agent_i in perm:
                v_without = v(predecessors)
                v_with = v(predecessors | frozenset([agent_i]))
                shapley[agent_i] += (v_with - v_without) / n_fact
                predecessors = predecessors | frozenset([agent_i])

        return shapley

    def compute_from_env(
        self,
        env: "_MiniEnv",
        agent_actions: List[int],
        state: torch.Tensor,
        seed: int = 0,
    ) -> Dict[int, float]:
        """
        Convenience wrapper that constructs env_step_fn from a _MiniEnv.

        Parameters
        ----------
        env : _MiniEnv
        agent_actions : list[int]
        state : Tensor — current state
        seed : int

        Returns
        -------
        dict  {agent_id: shapley_value}
        """
        def step_fn(actions: List[int]) -> float:
            _, reward = env.simulate(state, actions)
            return reward

        return self.compute(step_fn, agent_actions)

    def attribute_collapse(
        self,
        agent_shapley_values: Dict[int, float],
    ) -> int:
        """
        Identify the agent most responsible for a collapse (lowest Shapley value).

        A negative Shapley value indicates the agent's participation *reduced*
        total reward — i.e. the agent caused harm.

        Parameters
        ----------
        agent_shapley_values : dict {agent_id: shapley_value}

        Returns
        -------
        int — agent_id of the most harmful agent
        """
        return min(agent_shapley_values, key=lambda k: agent_shapley_values[k])

    def summary(
        self,
        agent_shapley_values: Dict[int, float],
    ) -> Dict:
        """
        Variance decomposition and attribution summary.

        Returns
        -------
        dict with keys:
            values          — raw Shapley dict
            total_value     — sum of all Shapley values (= v(grand coalition) - v(empty))
            top_contributor — agent with highest Shapley value
            blame_agent     — agent with lowest Shapley value
            variance        — variance across agents
            relative_share  — each agent's share of total (normalised)
        """
        vals = list(agent_shapley_values.values())
        total = sum(vals)
        variance = stats_lib.variance(vals) if len(vals) > 1 else 0.0
        top = max(agent_shapley_values, key=lambda k: agent_shapley_values[k])
        blame = min(agent_shapley_values, key=lambda k: agent_shapley_values[k])

        relative: Dict[int, float] = {}
        for k, v in agent_shapley_values.items():
            relative[k] = v / total if abs(total) > 1e-9 else 1.0 / len(agent_shapley_values)

        return {
            "values": dict(agent_shapley_values),
            "total_value": total,
            "top_contributor": top,
            "blame_agent": blame,
            "variance": variance,
            "relative_share": relative,
        }


# ═══════════════════════════════════════════════════════════════
# 2. SHAPLEY TRACKER
# ═══════════════════════════════════════════════════════════════

class ShapleyTracker:
    """
    Tracks Shapley values across an episode, maintaining running
    averages and identifying persistent contributors / free-riders.

    Parameters
    ----------
    n_agents : int
        Number of agents to track.
    """

    def __init__(self, n_agents: int = N_AGENTS):
        self.n_agents = n_agents
        self._history: List[Dict[int, float]] = []
        self._running_sum: Dict[int, float] = {i: 0.0 for i in range(n_agents)}
        self._running_count: int = 0

    def step(self, values: Dict[int, float]) -> None:
        """
        Record one step of Shapley values.

        Parameters
        ----------
        values : dict {agent_id: shapley_value}
        """
        self._history.append(dict(values))
        self._running_count += 1
        for i in range(self.n_agents):
            self._running_sum[i] += values.get(i, 0.0)

    def running_average(self) -> Dict[int, float]:
        """Running average Shapley value per agent."""
        if self._running_count == 0:
            return {i: 0.0 for i in range(self.n_agents)}
        return {i: self._running_sum[i] / self._running_count for i in range(self.n_agents)}

    def report(self) -> Dict:
        """
        Full episode report.

        Returns
        -------
        dict with keys:
            n_steps             — number of steps tracked
            running_average     — per-agent running average
            per_step            — full history
            top_contributor     — agent with highest running average
            free_rider          — agent with lowest running average
            agent_std           — per-agent standard deviation across steps
            contribution_rank   — agents sorted best -> worst
        """
        avg = self.running_average()

        # Per-agent std across steps
        agent_std: Dict[int, float] = {}
        for i in range(self.n_agents):
            vals = [h.get(i, 0.0) for h in self._history]
            agent_std[i] = stats_lib.stdev(vals) if len(vals) > 1 else 0.0

        sorted_agents = sorted(avg, key=lambda k: avg[k], reverse=True)
        top = sorted_agents[0] if sorted_agents else 0
        rider = sorted_agents[-1] if sorted_agents else 0

        return {
            "n_steps": self._running_count,
            "running_average": avg,
            "per_step": list(self._history),
            "top_contributor": top,
            "free_rider": rider,
            "agent_std": agent_std,
            "contribution_rank": sorted_agents,
        }

    def reset(self) -> None:
        """Clear all tracked data."""
        self._history.clear()
        self._running_sum = {i: 0.0 for i in range(self.n_agents)}
        self._running_count = 0


# ═══════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════

def validate_shapley() -> None:
    """Smoke-test ShapleyCredit and ShapleyTracker."""
    print("=" * 64)
    print("  POLARIS v5 — SHAPLEY MODULE VALIDATION")
    print("=" * 64)

    # ── ShapleyCredit basic properties ────────────────────────
    sc = ShapleyCredit(n_agents=N_AGENTS)

    # Dummy step function: reward = sum of actions (each agent contributes its action index)
    def dummy_step(actions: List[int]) -> float:
        return float(sum(actions))

    actions = [3, 1, 4, 1, 5]
    shapley_vals = sc.compute(dummy_step, actions)

    print("  [ShapleyCredit]  dummy_step = sum(actions)")
    print(f"    actions = {actions}")
    for i in range(N_AGENTS):
        print(f"    agent {i}: shapley = {shapley_vals[i]:+.4f}  (action = {actions[i]})")

    # Efficiency check: sum of Shapley values == v(grand) - v(empty)
    v_grand = dummy_step(actions)
    v_empty = dummy_step([NO_ACTION] * N_AGENTS)
    shapley_sum = sum(shapley_vals.values())
    expected = v_grand - v_empty
    print(f"    sum(shapley) = {shapley_sum:.4f}  expected = {expected:.4f}  "
          f"diff = {abs(shapley_sum - expected):.6f}")
    assert abs(shapley_sum - expected) < 1e-6, "Shapley efficiency violated!"
    print("    Efficiency check PASSED")
    print()

    # ── ShapleyCredit with _MiniEnv ───────────────────────────
    env = _MiniEnv()
    state = env.reset(seed=42)
    actions_env = [2, 5, 0, 11, 7]
    sv_env = sc.compute_from_env(env, actions_env, state, seed=42)
    print("  [ShapleyCredit + _MiniEnv]")
    for i in range(N_AGENTS):
        print(f"    agent {i}: shapley = {sv_env[i]:+.6f}")

    blame = sc.attribute_collapse(sv_env)
    summary = sc.summary(sv_env)
    print(f"    top_contributor = agent {summary['top_contributor']}")
    print(f"    blame_agent     = agent {summary['blame_agent']}")
    print(f"    variance        = {summary['variance']:.6f}")
    print()

    # ── ShapleyTracker across an episode ──────────────────────
    tracker = ShapleyTracker()
    env2 = _MiniEnv()
    state2 = env2.reset(seed=99)

    for step in range(20):
        acts = [random.randint(0, ACTION_DIM - 1) for _ in range(N_AGENTS)]
        sv = sc.compute_from_env(env2, acts, state2, seed=99 + step)
        tracker.step(sv)
        state2, _ = env2.simulate(state2, acts)

    report = tracker.report()
    print("  [ShapleyTracker]  20-step episode")
    print(f"    top_contributor  = agent {report['top_contributor']}")
    print(f"    free_rider       = agent {report['free_rider']}")
    print(f"    contribution_rank = {report['contribution_rank']}")
    avg = report["running_average"]
    for i in range(N_AGENTS):
        print(f"    agent {i}: avg_shapley = {avg[i]:+.6f}  std = {report['agent_std'][i]:.6f}")
    print()

    # ── Symmetry test ─────────────────────────────────────────
    # If two agents take the same action in a symmetric game, their
    # Shapley values should be equal.
    sym_actions = [3, 3, 3, 3, 3]  # all identical
    sv_sym = sc.compute(dummy_step, sym_actions)
    vals_set = set(round(v, 8) for v in sv_sym.values())
    assert len(vals_set) == 1, f"Symmetry violated: {sv_sym}"
    print("  [Symmetry test]  all agents identical -> equal Shapley values PASSED")

    # ── Null player test ──────────────────────────────────────
    # An agent with action=0 (no_action) should have Shapley value 0
    # in the dummy_step game since no_action = 0 contributes 0.
    null_actions = [5, 0, 3, 0, 7]
    sv_null = sc.compute(dummy_step, null_actions)
    assert abs(sv_null[1]) < 1e-6, f"Null player test failed for agent 1: {sv_null[1]}"
    assert abs(sv_null[3]) < 1e-6, f"Null player test failed for agent 3: {sv_null[3]}"
    print("  [Null player test]  agents with no_action -> shapley=0 PASSED")
    print()

    print("=" * 64)
    print("  ALL SHAPLEY TESTS PASSED")
    print("=" * 64)


if __name__ == "__main__":
    validate_shapley()
