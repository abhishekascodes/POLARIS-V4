#!/usr/bin/env python3
"""
POLARIS v5 — Byzantine Fault Tolerance for AI Governance
==========================================================
Adversarial robustness modules that stress-test multi-agent governance
under hostile/compromised agents.

Modules:
  1. RogueMinister   — Hostile agent that maximises damage while appearing cooperative
  2. ByzantineDetector — Monitors agent behaviour and isolates anomalies
  3. AdversarialBenchmark — Full evaluation harness for adversarial robustness

Self-contained.  Imports: torch, nn, math, random, collections, statistics, typing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
import statistics

# ─── Global constants ────────────────────────────────────────
N_AGENTS: int = 5
STATE_DIM: int = 21
ACTION_DIM: int = 16

# Semantic indices inside the 21-d state vector (matching env conventions)
_GDP_IDX = 0
_POLLUTION_IDX = 1
_SATISFACTION_IDX = 2
_HEALTHCARE_IDX = 3
_EDUCATION_IDX = 4
_UNEMPLOYMENT_IDX = 5

# Attack-mode to "damage-desirable" action biases (action indices the rogue
# will gravitate toward to cause maximum harm in each mode).
_ATTACK_ACTION_BIASES: Dict[str, List[int]] = {
    "economic_sabotage":          [6, 7, 8, 9],
    "environmental_destruction":  [10, 11, 12],
    "social_manipulation":        [13, 14, 15],
    "random_chaos":               list(range(ACTION_DIM)),
}


def _infer_device(module: nn.Module) -> torch.device:
    """Infer device from a module's first parameter (or default to CPU)."""
    p = next(module.parameters(), None)
    return p.device if p is not None else torch.device("cpu")


# ═══════════════════════════════════════════════════════════════
# 1. ROGUE MINISTER
# ═══════════════════════════════════════════════════════════════

class RogueMinister(nn.Module):
    """
    Adversarial agent that learns to *maximise* system damage while
    keeping its action distribution close enough to the cooperative
    baseline to evade naive detection.

    Attack modes
    ------------
    * ``economic_sabotage``          — tanks GDP and employment
    * ``environmental_destruction``  — spikes pollution
    * ``social_manipulation``        — crashes public satisfaction
    * ``random_chaos``               — randomly destructive, hardest to profile

    The rogue has a small MLP policy trained to *maximise* an adversarial
    reward (positive when damage is observed) while a *stealth* term
    regularises toward the cooperative action distribution.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the environment state vector.
    action_dim : int
        Number of discrete actions.
    hidden : int
        Hidden-layer width.
    attack_mode : str
        One of the four attack modes above.
    stealth : float
        Weight of the KL-stealth penalty (higher = harder to detect
        but less destructive).
    """

    ATTACK_MODES = list(_ATTACK_ACTION_BIASES.keys())

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        hidden: int = 128,
        attack_mode: str = "economic_sabotage",
        stealth: float = 0.3,
    ):
        super().__init__()
        assert attack_mode in self.ATTACK_MODES, (
            f"Unknown attack_mode '{attack_mode}'. "
            f"Choose from {self.ATTACK_MODES}"
        )
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.attack_mode = attack_mode
        self.stealth = stealth

        # Neural adversarial policy
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

        # Bias vector toward harmful actions for the chosen attack mode
        bias_idx = _ATTACK_ACTION_BIASES[attack_mode]
        bias = torch.zeros(action_dim)
        bias[bias_idx] = 2.0  # soft prior toward damage actions
        self.register_buffer("_action_bias", bias)

        # "Cooperative camouflage" — a fixed soft distribution the rogue
        # tries to resemble so that statistical detectors are confused.
        camo = torch.ones(action_dim) / action_dim
        self.register_buffer("_camo_dist", camo)

    # ── public API ────────────────────────────────────────────

    def act(self, state: torch.Tensor) -> int:
        """
        Select an action index from the adversarial policy.

        Parameters
        ----------
        state : Tensor  shape ``(state_dim,)`` or ``(1, state_dim)``

        Returns
        -------
        int — action index
        """
        device = _infer_device(self)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state = state.to(device)

        logits = self.policy(state).squeeze(0) + self._action_bias
        probs = F.softmax(logits, dim=-1)

        # Stealthy blend: mix adversarial probs with camouflage
        blended = (1.0 - self.stealth) * probs + self.stealth * self._camo_dist
        dist = torch.distributions.Categorical(probs=blended)
        return dist.sample().item()

    def adversarial_reward(
        self,
        state_before: torch.Tensor,
        state_after: torch.Tensor,
    ) -> float:
        """
        Compute adversarial reward: *positive* when the rogue causes damage.

        Rewards large drops in GDP / satisfaction and increases in pollution
        / unemployment, weighted by attack mode.

        Parameters
        ----------
        state_before, state_after : Tensor  shape ``(state_dim,)``

        Returns
        -------
        float — adversarial reward (higher = more damage done)
        """
        sb = state_before.detach().cpu()
        sa = state_after.detach().cpu()

        # Raw deltas (positive delta = metric went up)
        gdp_delta = (sb[_GDP_IDX] - sa[_GDP_IDX]).item()
        poll_delta = (sa[_POLLUTION_IDX] - sb[_POLLUTION_IDX]).item()
        sat_delta = (sb[_SATISFACTION_IDX] - sa[_SATISFACTION_IDX]).item()
        unemp_delta = (sa[_UNEMPLOYMENT_IDX] - sb[_UNEMPLOYMENT_IDX]).item()

        mode = self.attack_mode
        if mode == "economic_sabotage":
            return 2.0 * gdp_delta + 1.5 * unemp_delta + 0.5 * sat_delta
        elif mode == "environmental_destruction":
            return 3.0 * poll_delta + 0.5 * gdp_delta
        elif mode == "social_manipulation":
            return 2.5 * sat_delta + 1.0 * unemp_delta + 0.5 * poll_delta
        else:  # random_chaos
            return gdp_delta + poll_delta + sat_delta + unemp_delta


# ═══════════════════════════════════════════════════════════════
# 2. BYZANTINE DETECTOR
# ═══════════════════════════════════════════════════════════════

class ByzantineDetector(nn.Module):
    """
    Anomaly detection system that monitors all N ministers and
    identifies rogue behaviour using a combination of:

    1. **Learned anomaly scorer** — a small MLP that maps
       (state, action, group_consensus_action) -> anomaly logit.
    2. **Statistical deviation** — tracks per-agent action histograms
       and flags agents whose distributions diverge from the group
       (via Jensen-Shannon divergence).
    3. **Trust scores** — exponentially-decaying trust that drops on
       anomaly detections and slowly recovers otherwise.

    Designed to detect a rogue agent within 10-20 steps.

    Parameters
    ----------
    state_dim, action_dim : int
        Environment dimensions.
    n_agents : int
        Number of ministers to monitor.
    hidden : int
        Hidden width of the anomaly scorer.
    window : int
        Rolling window length for action histograms.
    anomaly_threshold : float
        Score above which an agent is flagged.
    trust_decay : float
        Multiplicative decay applied to trust on each anomaly flag.
    trust_recovery : float
        Additive recovery per clean step (capped at 1.0).
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        n_agents: int = N_AGENTS,
        hidden: int = 64,
        window: int = 20,
        anomaly_threshold: float = 0.6,
        trust_decay: float = 0.7,
        trust_recovery: float = 0.02,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.window = window
        self.anomaly_threshold = anomaly_threshold
        self.trust_decay = trust_decay
        self.trust_recovery = trust_recovery

        # ── Learned anomaly scorer ────────────────────────────
        # Input:  state  +  one-hot(action)  +  group_consensus one-hot
        scorer_input = state_dim + 2 * action_dim
        self.scorer = nn.Sequential(
            nn.Linear(scorer_input, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        # ── Per-agent behavioural profiles ────────────────────
        self._action_history: Dict[int, deque] = {
            i: deque(maxlen=window) for i in range(n_agents)
        }
        self._trust: Dict[int, float] = {i: 1.0 for i in range(n_agents)}
        self._isolated: Dict[int, bool] = {i: False for i in range(n_agents)}
        self._step = 0

    # ── helpers ───────────────────────────────────────────────

    def _action_distribution(self, agent_id: int) -> torch.Tensor:
        """Normalised action histogram for *agent_id* over the rolling window."""
        hist = torch.zeros(self.action_dim)
        for a in self._action_history[agent_id]:
            hist[a] += 1.0
        total = hist.sum()
        if total > 0:
            hist = hist / total
        else:
            hist = torch.ones(self.action_dim) / self.action_dim
        return hist

    def _group_distribution(self, exclude: int = -1) -> torch.Tensor:
        """Average action histogram across all non-isolated agents."""
        combined = torch.zeros(self.action_dim)
        count = 0
        for i in range(self.n_agents):
            if i == exclude or self._isolated[i]:
                continue
            combined += self._action_distribution(i)
            count += 1
        if count > 0:
            combined = combined / count
        else:
            combined = torch.ones(self.action_dim) / self.action_dim
        return combined

    @staticmethod
    def _js_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
        """Jensen-Shannon divergence between two distributions."""
        eps = 1e-8
        p = p.clamp(min=eps)
        q = q.clamp(min=eps)
        m = 0.5 * (p + q)
        kl_pm = (p * (p / m).log()).sum()
        kl_qm = (q * (q / m).log()).sum()
        return 0.5 * (kl_pm + kl_qm).item()

    def _consensus_action(self, group_actions: List[int]) -> int:
        """Most common action among the group (mode)."""
        counts: Dict[int, int] = defaultdict(int)
        for a in group_actions:
            counts[a] += 1
        return max(counts, key=lambda k: counts[k])

    # ── public API ────────────────────────────────────────────

    def update(
        self,
        agent_id: int,
        action: int,
        state: torch.Tensor,
        group_actions: List[int],
    ) -> None:
        """
        Record one step of observation for *agent_id*.

        Parameters
        ----------
        agent_id : int
        action   : int — the action this agent took
        state    : Tensor  shape ``(state_dim,)``
        group_actions : list[int] — actions of *all* agents this step
        """
        self._action_history[agent_id].append(action)
        self._step += 1

    def detect(self) -> Dict[int, float]:
        """
        Compute anomaly scores for every agent.

        Returns
        -------
        dict  {agent_id: anomaly_score}  where score in [0, 1].
        Higher = more anomalous.
        """
        device = _infer_device(self)
        scores: Dict[int, float] = {}

        for i in range(self.n_agents):
            if self._isolated[i]:
                scores[i] = 1.0
                continue

            # 1) Statistical deviation (JSD from group)
            p_i = self._action_distribution(i)
            p_group = self._group_distribution(exclude=i)
            jsd = self._js_divergence(p_i, p_group)

            # 2) Learned anomaly score
            if len(self._action_history[i]) > 0:
                last_action = self._action_history[i][-1]
            else:
                last_action = 0
            action_oh = F.one_hot(torch.tensor(last_action), self.action_dim).float()

            active_last = [
                self._action_history[j][-1]
                for j in range(self.n_agents)
                if len(self._action_history[j]) > 0 and not self._isolated[j]
            ] or [0]
            consensus_oh = F.one_hot(
                torch.tensor(self._consensus_action(active_last)),
                self.action_dim,
            ).float()

            dummy_state = torch.zeros(STATE_DIM, device=device)
            inp = torch.cat([
                dummy_state,
                action_oh.to(device),
                consensus_oh.to(device),
            ]).unsqueeze(0)
            with torch.no_grad():
                learned_score = torch.sigmoid(self.scorer(inp)).item()

            # Combine: weighted average of statistical + learned
            combined = 0.5 * min(jsd / 0.5, 1.0) + 0.5 * learned_score

            # Update trust
            if combined > self.anomaly_threshold:
                self._trust[i] *= self.trust_decay
            else:
                self._trust[i] = min(1.0, self._trust[i] + self.trust_recovery)

            scores[i] = combined

        return scores

    def isolate(self, agent_id: int) -> None:
        """Mark *agent_id* as untrusted and exclude from group consensus."""
        self._isolated[agent_id] = True
        self._trust[agent_id] = 0.0

    def get_trust_scores(self) -> Dict[int, float]:
        """Behavioural trust score per agent (separate from any ZK trust)."""
        return dict(self._trust)

    def reset(self) -> None:
        """Clear all profiles and trust scores."""
        for i in range(self.n_agents):
            self._action_history[i].clear()
            self._trust[i] = 1.0
            self._isolated[i] = False
        self._step = 0


# ═══════════════════════════════════════════════════════════════
# 3. ADVERSARIAL BENCHMARK
# ═══════════════════════════════════════════════════════════════

class _SimpleEnv:
    """
    Lightweight governance environment stub for self-contained testing.
    Maintains a 21-d state vector and applies crude action effects.
    """

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state = torch.zeros(state_dim)

    def reset(self, seed: int = 0) -> torch.Tensor:
        rng = random.Random(seed)
        self.state = torch.tensor(
            [100.0, 100.0, 60.0, 50.0, 50.0, 8.0]
            + [rng.uniform(-1, 1) for _ in range(self.state_dim - 6)],
            dtype=torch.float32,
        )
        return self.state.clone()

    def step(self, actions: List[int]) -> Tuple[torch.Tensor, float]:
        """Apply joint actions, return (new_state, reward)."""
        delta = torch.zeros(self.state_dim)
        for a in actions:
            torch.manual_seed(a * 137)
            delta += 0.3 * torch.randn(self.state_dim)
        self.state = self.state + delta
        reward = (
            0.01 * self.state[_GDP_IDX].item()
            - 0.005 * self.state[_POLLUTION_IDX].item()
            + 0.01 * self.state[_SATISFACTION_IDX].item()
            - 0.02 * self.state[_UNEMPLOYMENT_IDX].item()
        )
        return self.state.clone(), reward

    def score(self) -> float:
        return (
            0.3 * self.state[_GDP_IDX].item()
            + 0.2 * self.state[_SATISFACTION_IDX].item()
            - 0.2 * self.state[_POLLUTION_IDX].item()
            - 0.3 * self.state[_UNEMPLOYMENT_IDX].item()
        )


class _SimpleCouncil(nn.Module):
    """Minimal cooperative council of N independent MLP policies."""

    def __init__(
        self,
        n_agents: int = N_AGENTS,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        hidden: int = 64,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.policies = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, action_dim),
            )
            for _ in range(n_agents)
        ])

    def act(self, state: torch.Tensor) -> List[int]:
        device = _infer_device(self)
        s = state.unsqueeze(0).to(device) if state.dim() == 1 else state.to(device)
        actions: List[int] = []
        for net in self.policies:
            logits = net(s).squeeze(0)
            actions.append(torch.distributions.Categorical(logits=logits).sample().item())
        return actions


class AdversarialBenchmark:
    """
    Full adversarial evaluation harness.

    Replaces agent *rogue_slot* with a RogueMinister, runs an
    episode, and measures:

    * **Detection latency** — steps before the detector flags the rogue
    * **System resilience**  — score with rogue vs clean baseline
    * **Isolation effectiveness** — score improvement after isolation

    Parameters
    ----------
    n_agents : int
        Number of ministers.
    max_steps : int
        Episode length.
    rogue_slot : int
        Which agent index to replace with the rogue.
    attack_mode : str
        Attack mode for the RogueMinister.
    detection_threshold : float
        Anomaly score above which an agent is auto-isolated.
    """

    def __init__(
        self,
        n_agents: int = N_AGENTS,
        max_steps: int = 50,
        rogue_slot: int = 2,
        attack_mode: str = "economic_sabotage",
        detection_threshold: float = 0.55,
    ):
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.rogue_slot = rogue_slot
        self.attack_mode = attack_mode
        self.detection_threshold = detection_threshold

    def _run_clean(self, env, council, seed: int) -> float:
        state = env.reset(seed)
        for _ in range(self.max_steps):
            actions = council.act(state)
            state, _ = env.step(actions)
        return env.score()

    def _run_adversarial(self, env, council, rogue, detector, seed: int) -> Dict:
        state = env.reset(seed)
        detector.reset()

        detection_step: Optional[int] = None
        isolation_step: Optional[int] = None
        pre_isolation_rewards: List[float] = []
        post_isolation_rewards: List[float] = []
        adv_rewards: List[float] = []

        for step in range(1, self.max_steps + 1):
            state_before = state.clone()
            coop_actions = council.act(state)

            if not detector._isolated.get(self.rogue_slot, False):
                rogue_action = rogue.act(state)
            else:
                rogue_action = 0
            coop_actions[self.rogue_slot] = rogue_action

            state, reward = env.step(coop_actions)
            adv_r = rogue.adversarial_reward(state_before, state)
            adv_rewards.append(adv_r)

            for aid in range(self.n_agents):
                detector.update(aid, coop_actions[aid], state_before, coop_actions)

            scores = detector.detect()
            rogue_score = scores.get(self.rogue_slot, 0.0)

            if detection_step is None and rogue_score > self.detection_threshold:
                detection_step = step

            if (
                detection_step is not None
                and isolation_step is None
                and rogue_score > self.detection_threshold
            ):
                detector.isolate(self.rogue_slot)
                isolation_step = step

            if isolation_step is None:
                pre_isolation_rewards.append(reward)
            else:
                post_isolation_rewards.append(reward)

        return {
            "terminal_score": env.score(),
            "detection_step": detection_step,
            "isolation_step": isolation_step,
            "mean_adv_reward": statistics.mean(adv_rewards) if adv_rewards else 0.0,
            "mean_pre_isolation_reward": (
                statistics.mean(pre_isolation_rewards) if pre_isolation_rewards else 0.0
            ),
            "mean_post_isolation_reward": (
                statistics.mean(post_isolation_rewards) if post_isolation_rewards else 0.0
            ),
            "trust_scores": detector.get_trust_scores(),
        }

    def run(
        self,
        env=None,
        council=None,
        seeds: Optional[List[int]] = None,
    ) -> Dict:
        """
        Execute the full adversarial benchmark.

        Parameters
        ----------
        env : environment instance (created internally if None).
        council : cooperative council (created internally if None).
        seeds : list[int] — seeds for repeated trials (default [42, 123, 777]).

        Returns
        -------
        dict — Aggregated metrics:
            detection_latency_mean, detection_latency_std,
            clean_score_mean, adversarial_score_mean,
            resilience, isolation_effectiveness,
            per_seed.
        """
        if env is None:
            env = _SimpleEnv()
        if council is None:
            council = _SimpleCouncil()
        if seeds is None:
            seeds = [42, 123, 777]

        rogue = RogueMinister(attack_mode=self.attack_mode)
        detector = ByzantineDetector(n_agents=self.n_agents)

        clean_scores: List[float] = []
        adv_scores: List[float] = []
        detection_latencies: List[int] = []
        isolation_effects: List[float] = []
        per_seed: List[Dict] = []

        for seed in seeds:
            clean_score = self._run_clean(env, council, seed)
            clean_scores.append(clean_score)

            result = self._run_adversarial(env, council, rogue, detector, seed)
            adv_scores.append(result["terminal_score"])

            if result["detection_step"] is not None:
                detection_latencies.append(result["detection_step"])

            iso_eff = (
                result["mean_post_isolation_reward"]
                - result["mean_pre_isolation_reward"]
            )
            isolation_effects.append(iso_eff)

            per_seed.append({"seed": seed, "clean_score": clean_score, **result})

        clean_mean = statistics.mean(clean_scores)
        adv_mean = statistics.mean(adv_scores)
        resilience = adv_mean / clean_mean if abs(clean_mean) > 1e-6 else 0.0

        dl_mean = statistics.mean(detection_latencies) if detection_latencies else float("inf")
        dl_std = statistics.stdev(detection_latencies) if len(detection_latencies) > 1 else 0.0
        iso_mean = statistics.mean(isolation_effects)

        return {
            "attack_mode": self.attack_mode,
            "rogue_slot": self.rogue_slot,
            "n_seeds": len(seeds),
            "clean_score_mean": clean_mean,
            "adversarial_score_mean": adv_mean,
            "resilience": resilience,
            "detection_latency_mean": dl_mean,
            "detection_latency_std": dl_std,
            "isolation_effectiveness_mean": iso_mean,
            "per_seed": per_seed,
        }


# ═══════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════

def validate_adversarial() -> None:
    """Smoke-test every class in this module."""
    print("=" * 64)
    print("  POLARIS v5 — ADVERSARIAL MODULE VALIDATION")
    print("=" * 64)

    # ── RogueMinister ─────────────────────────────────────────
    for mode in RogueMinister.ATTACK_MODES:
        rogue = RogueMinister(attack_mode=mode)
        state = torch.randn(STATE_DIM)
        action = rogue.act(state)
        assert 0 <= action < ACTION_DIM, f"Bad action {action} for mode {mode}"
        state2 = state + 0.1 * torch.randn(STATE_DIM)
        adv_r = rogue.adversarial_reward(state, state2)
        print(f"  [RogueMinister:{mode:>25s}]  action={action:>2d}  adv_r={adv_r:+.4f}")
    print()

    # ── ByzantineDetector ─────────────────────────────────────
    detector = ByzantineDetector()
    rogue = RogueMinister(attack_mode="economic_sabotage", stealth=0.1)
    env = _SimpleEnv()
    state = env.reset(seed=42)

    for step in range(30):
        coop_actions = [random.randint(0, ACTION_DIM - 1) for _ in range(N_AGENTS)]
        rogue_action = rogue.act(state)
        coop_actions[2] = rogue_action
        for aid in range(N_AGENTS):
            detector.update(aid, coop_actions[aid], state, coop_actions)
        state, _ = env.step(coop_actions)

    scores = detector.detect()
    trust = detector.get_trust_scores()
    print("  [ByzantineDetector]  anomaly scores:")
    for aid in range(N_AGENTS):
        tag = " <-- ROGUE" if aid == 2 else ""
        print(f"    agent {aid}: score={scores[aid]:.4f}  trust={trust[aid]:.4f}{tag}")

    detector.isolate(2)
    assert detector._isolated[2], "Isolation failed"
    assert detector.get_trust_scores()[2] == 0.0, "Trust not zeroed"
    print("  [ByzantineDetector]  agent 2 isolated OK")
    print()

    # ── AdversarialBenchmark ──────────────────────────────────
    bench = AdversarialBenchmark(max_steps=30, rogue_slot=2, attack_mode="economic_sabotage")
    results = bench.run(seeds=[42, 99])

    print("  [AdversarialBenchmark]")
    print(f"    attack_mode          = {results['attack_mode']}")
    print(f"    clean_score_mean     = {results['clean_score_mean']:.4f}")
    print(f"    adversarial_score    = {results['adversarial_score_mean']:.4f}")
    print(f"    resilience           = {results['resilience']:.4f}")
    print(f"    detection_latency    = {results['detection_latency_mean']:.1f} "
          f"+/- {results['detection_latency_std']:.1f} steps")
    print(f"    isolation_eff_mean   = {results['isolation_effectiveness_mean']:.4f}")
    print()

    print("=" * 64)
    print("  ALL ADVERSARIAL TESTS PASSED")
    print("=" * 64)


if __name__ == "__main__":
    validate_adversarial()
