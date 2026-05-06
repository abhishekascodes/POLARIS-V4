"""
AI Policy Engine — Multi-Agent Council (v1 — Nuclear Upgrade)

Implements 5 minister agents with:
  - Dynamic utility vectors that evolve based on state trends,
    recent outcomes, hidden trust, and personal role bias.
  - Weighted voting based on influence + utility alignment.
  - Vetoes and betrayals emerge from utility conflicts under chaos/volatility.
  - Influence updated via per-agent credit assignment.
  - Meta-coordination: propose and vote on global policy packages.

This module acts as an action-selection layer: given the environment state,
it returns a recommended action (or meta-action) representing the council's
collective decision, along with full negotiation metadata.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .config import CORE_ACTIONS, META_ACTIONS, VALID_ACTIONS


# ─────────────────────────────────────────────────────────────
# Minister Role Definitions
# ─────────────────────────────────────────────────────────────

@dataclass
class MinisterRole:
    name: str
    # Which objective dimensions this minister cares about (weights sum to 1)
    # Dimensions: [gdp, environment, social, health, industry]
    objective_weights: List[float]
    # Preferred action categories (for proposal biases)
    preferred_actions: List[str]
    # Disliked action categories (utility penalty)
    disliked_actions: List[str]

MINISTER_ROLES = [
    MinisterRole(
        name="Finance",
        objective_weights=[0.45, 0.10, 0.20, 0.10, 0.15],
        preferred_actions=["stimulate_economy", "decrease_tax", "reduce_interest_rates",
                           "invest_in_education", "invest_in_transport"],
        disliked_actions=["implement_carbon_tax", "restrict_polluting_industries",
                          "enforce_emission_limits"],
    ),
    MinisterRole(
        name="Environment",
        objective_weights=[0.10, 0.45, 0.20, 0.10, 0.15],
        preferred_actions=["subsidize_renewables", "enforce_emission_limits",
                           "incentivize_clean_tech", "implement_carbon_tax",
                           "restrict_polluting_industries"],
        disliked_actions=["expand_industry", "stimulate_economy", "decrease_tax"],
    ),
    MinisterRole(
        name="Health",
        objective_weights=[0.10, 0.20, 0.30, 0.35, 0.05],
        preferred_actions=["invest_in_healthcare", "increase_welfare", "invest_in_education",
                           "restrict_polluting_industries"],
        disliked_actions=["expand_industry", "stimulate_economy"],
    ),
    MinisterRole(
        name="Industry",
        objective_weights=[0.35, 0.10, 0.15, 0.05, 0.35],
        preferred_actions=["expand_industry", "stimulate_economy", "reduce_interest_rates",
                           "decrease_tax", "invest_in_transport"],
        disliked_actions=["restrict_polluting_industries", "implement_carbon_tax",
                          "enforce_emission_limits", "increase_tax"],
    ),
    MinisterRole(
        name="Social",
        objective_weights=[0.10, 0.15, 0.40, 0.20, 0.15],
        preferred_actions=["increase_welfare", "invest_in_education", "invest_in_healthcare",
                           "upgrade_energy_grid", "invest_in_transport"],
        disliked_actions=["decrease_tax", "expand_industry"],
    ),
    # ── Extended roles for 8-12 minister scaling scenarios ──
    MinisterRole(
        name="Energy",
        objective_weights=[0.15, 0.30, 0.10, 0.10, 0.35],
        preferred_actions=["upgrade_energy_grid", "subsidize_renewables",
                           "incentivize_clean_tech", "invest_in_transport"],
        disliked_actions=["expand_industry", "decrease_tax"],
    ),
    MinisterRole(
        name="Defense",
        objective_weights=[0.25, 0.05, 0.15, 0.05, 0.50],
        preferred_actions=["expand_industry", "increase_tax", "invest_in_transport",
                           "upgrade_energy_grid"],
        disliked_actions=["decrease_tax", "increase_welfare", "subsidize_renewables"],
    ),
    MinisterRole(
        name="Education",
        objective_weights=[0.10, 0.10, 0.35, 0.30, 0.15],
        preferred_actions=["invest_in_education", "invest_in_healthcare",
                           "increase_welfare", "incentivize_clean_tech"],
        disliked_actions=["expand_industry", "decrease_tax"],
    ),
    MinisterRole(
        name="Trade",
        objective_weights=[0.40, 0.10, 0.15, 0.05, 0.30],
        preferred_actions=["stimulate_economy", "decrease_tax", "reduce_interest_rates",
                           "expand_industry", "invest_in_transport"],
        disliked_actions=["increase_tax", "implement_carbon_tax",
                          "restrict_polluting_industries"],
    ),
    MinisterRole(
        name="Labor",
        objective_weights=[0.15, 0.10, 0.45, 0.15, 0.15],
        preferred_actions=["increase_welfare", "invest_in_education",
                           "invest_in_healthcare", "stimulate_economy"],
        disliked_actions=["decrease_tax", "restrict_polluting_industries"],
    ),
    MinisterRole(
        name="Technology",
        objective_weights=[0.20, 0.25, 0.10, 0.10, 0.35],
        preferred_actions=["incentivize_clean_tech", "upgrade_energy_grid",
                           "invest_in_education", "subsidize_renewables"],
        disliked_actions=["no_action", "increase_tax"],
    ),
    MinisterRole(
        name="Agriculture",
        objective_weights=[0.20, 0.35, 0.25, 0.10, 0.10],
        preferred_actions=["subsidize_renewables", "increase_welfare",
                           "restrict_polluting_industries", "invest_in_transport"],
        disliked_actions=["expand_industry", "stimulate_economy"],
    ),
]


# ─────────────────────────────────────────────────────────────
# Minister Agent
# ─────────────────────────────────────────────────────────────

@dataclass
class Minister:
    """A single minister with dynamic utility vector and influence score."""

    role: MinisterRole
    influence: float = 0.2          # [0, 1], starts equal
    # Dynamic utility vector: [gdp_val, env_val, social_val, health_val, industry_val]
    utility_vector: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5, 0.5, 0.5])
    # Coalition membership: ID or None
    coalition_id: Optional[int] = None
    # Veto count this episode
    veto_count: int = 0
    # Personal trust scalar (how much others trust this minister)
    personal_trust: float = 0.5
    # Credit accumulated this episode
    total_credit: float = 0.0

    def propose_action(
        self,
        state: Dict[str, float],
        available_actions: List[str],
        rng: random.Random,
        utility_volatility: float = 1.0,
    ) -> str:
        """Propose an action based on current utility vector and state."""
        scores = {}
        for action in available_actions:
            base_score = self._score_action(action, state)
            # Apply volatility noise
            noise = rng.gauss(0, 0.05 * utility_volatility)
            scores[action] = base_score + noise
        # Softmax-weighted selection
        return self._softmax_select(scores, rng, temperature=0.3)

    def _score_action(self, action: str, state: Dict[str, float]) -> float:
        """Score an action based on this minister's utility vector."""
        score = 0.0
        role = self.role
        uv = self.utility_vector  # [gdp, env, social, health, industry]

        # Role preference bonus
        if action in role.preferred_actions:
            score += 0.3 * self.personal_trust
        if action in role.disliked_actions:
            score -= 0.2

        # Utility-weighted state urgency
        gdp = state.get("gdp_index", 100)
        poll = state.get("pollution_index", 100)
        sat = state.get("public_satisfaction", 50)
        hc = state.get("healthcare_index", 50)
        ind = state.get("industrial_output", 60)

        # How urgently does each dimension need attention?
        gdp_urgency = max(0, (80 - gdp) / 80)          # low gdp = high urgency
        env_urgency = max(0, (poll - 100) / 200)         # high poll = high urgency
        soc_urgency = max(0, (50 - sat) / 50)            # low sat = high urgency
        hc_urgency = max(0, (50 - hc) / 50)              # low hc = high urgency
        ind_urgency = max(0, (60 - ind) / 60)            # low ind = high urgency

        urgency_vector = [gdp_urgency, env_urgency, soc_urgency, hc_urgency, ind_urgency]

        # Weighted urgency contribution
        for i, w in enumerate(role.objective_weights):
            score += w * uv[i] * urgency_vector[i] * 2.0

        return score

    def _softmax_select(
        self, scores: Dict[str, float], rng: random.Random, temperature: float = 0.3
    ) -> str:
        """Select action via softmax sampling."""
        actions = list(scores.keys())
        if not actions:
            return "no_action"
        vals = [scores[a] / temperature for a in actions]
        max_v = max(vals)
        exp_v = [math.exp(min(v - max_v, 80)) for v in vals]
        total = sum(exp_v)
        probs = [e / total for e in exp_v]
        r = rng.random()
        cum = 0.0
        for a, p in zip(actions, probs):
            cum += p
            if r < cum:
                return a
        return actions[-1]

    def update_utility_vector(
        self,
        state: Dict[str, float],
        prev_state: Optional[Dict[str, float]],
        institutional_trust: float,
    ) -> None:
        """Evolve utility vector based on state trends and trust."""
        if prev_state is None:
            return

        # Compute dimension trends
        gdp_trend = (state.get("gdp_index", 100) - prev_state.get("gdp_index", 100)) / 100.0
        env_trend = -(state.get("pollution_index", 100) - prev_state.get("pollution_index", 100)) / 100.0
        soc_trend = (state.get("public_satisfaction", 50) - prev_state.get("public_satisfaction", 50)) / 50.0
        hc_trend = (state.get("healthcare_index", 50) - prev_state.get("healthcare_index", 50)) / 50.0
        ind_trend = (state.get("industrial_output", 60) - prev_state.get("industrial_output", 60)) / 60.0

        trends = [gdp_trend, env_trend, soc_trend, hc_trend, ind_trend]

        # Utility evolves toward dimensions that are declining (crisis response)
        trust_factor = 0.5 + 0.5 * institutional_trust
        for i, trend in enumerate(trends):
            # Minister cares more about declining dimensions
            urgency_pull = -trend * 0.1 * trust_factor
            # Role bias keeps vector anchored near role weights
            role_anchor = (self.role.objective_weights[i] - self.utility_vector[i]) * 0.05
            self.utility_vector[i] = max(0.1, min(1.0,
                self.utility_vector[i] + urgency_pull + role_anchor
            ))

    def update_influence(self, contribution: float) -> None:
        """Update influence via credit assignment."""
        # Exponential moving average
        self.influence = max(0.05, min(1.0, 0.9 * self.influence + 0.1 * contribution))
        self.total_credit += contribution

    def will_veto(
        self,
        proposed_action: str,
        state: Dict[str, float],
        chaos_level: float,
        utility_volatility: float,
        rng: random.Random,
    ) -> bool:
        """Veto emerges from utility conflict under high chaos — no hard-coded triggers."""
        action_score = self._score_action(proposed_action, state)
        # Low utility alignment + high chaos -> veto probability
        veto_prob = (1.0 - max(0.0, action_score)) * chaos_level * utility_volatility * 0.3
        veto_prob = max(0.0, min(0.5, veto_prob))
        vetoed = rng.random() < veto_prob
        if vetoed:
            self.veto_count += 1
        return vetoed


# ─────────────────────────────────────────────────────────────
# Coalition
# ─────────────────────────────────────────────────────────────

@dataclass
class Coalition:
    """A coalition of ministers aligned on a policy direction."""
    coalition_id: int
    members: List[str]       # minister names
    action: str              # agreed action
    strength: float          # combined influence
    formed_at_step: int = 0
    survived_steps: int = 0


# ─────────────────────────────────────────────────────────────
# Multi-Agent Council
# ─────────────────────────────────────────────────────────────

class MultiAgentCouncil:
    """
    Orchestrates 5 ministers with emergent negotiation, coalition formation,
    vetoes, and credit assignment.

    The council acts as an action-selector layer: given state, returns
    the council's recommended action plus full negotiation metadata.
    """

    COALITION_THRESHOLD = 0.35  # combined influence needed for coalition

    def __init__(self) -> None:
        self._ministers: List[Minister] = []
        self._coalitions: List[Coalition] = []
        self._active_coalition: Optional[Coalition] = None
        self._coalition_counter: int = 0
        self._step: int = 0
        self._rng: random.Random = random.Random(42)
        self._chaos_level: float = 0.5
        self._institutional_trust: float = 0.6
        self._prev_state: Optional[Dict[str, float]] = None
        self._num_ministers: int = 5
        # Action history (last 8 joint actions)
        self._action_history: List[str] = []

    def reset(
        self,
        seed: int = 42,
        num_ministers: int = 5,
        chaos_level: float = 0.5,
        institutional_trust: float = 0.6,
    ) -> None:
        """Reinitialise council for a new episode."""
        self._rng = random.Random(seed + 99991)
        self._num_ministers = min(num_ministers, len(MINISTER_ROLES))  # supports up to 12
        self._chaos_level = chaos_level
        self._institutional_trust = institutional_trust
        self._step = 0
        self._active_coalition = None
        self._coalitions = []
        self._coalition_counter = 0
        self._prev_state = None
        self._action_history = []

        # Initialise minister agents (use first N roles)
        self._ministers = []
        for i in range(self._num_ministers):
            role = MINISTER_ROLES[i]
            m = Minister(role=role)
            # Slightly randomise initial influence
            m.influence = 0.2 + self._rng.uniform(-0.02, 0.02)
            m.personal_trust = 0.5 + self._rng.uniform(-0.1, 0.1)
            self._ministers.append(m)

    def step(
        self,
        state: Dict[str, float],
        forced_action: Optional[str] = None,
        utility_volatility: float = 1.0,
        chaos_level: Optional[float] = None,
    ) -> Dict:
        """
        Run one council negotiation step.

        Args:
            state: Current world state.
            forced_action: If provided, use this action (bypass vote).
            utility_volatility: From event engine.
            chaos_level: Override chaos level.

        Returns:
            Dict with keys:
              action, votes, vetoes, coalition_formed, coalition_strength,
              proposals, credit_deltas, influence_vector,
              action_history, alignment_score
        """
        if chaos_level is not None:
            self._chaos_level = chaos_level

        # Update utility vectors
        for m in self._ministers:
            m.update_utility_vector(state, self._prev_state, self._institutional_trust)

        if forced_action is not None:
            chosen_action = forced_action
            votes = {m.role.name: forced_action for m in self._ministers}
            vetoes = []
        else:
            chosen_action, votes, vetoes = self._negotiate(state, utility_volatility)

        # Check for meta-action triggers
        if chosen_action == "force_emergency_coalition_vote":
            # Override to form emergency coalition
            chosen_action, votes, vetoes = self._emergency_coalition_vote(state, utility_volatility)

        # Update action history
        self._action_history.append(chosen_action)
        if len(self._action_history) > 8:
            self._action_history.pop(0)

        # Credit assignment
        credit_deltas = self._assign_credit(chosen_action, state, votes)
        for m in self._ministers:
            credit = credit_deltas.get(m.role.name, 0.0)
            m.update_influence(credit)

        # Update coalition survival
        if self._active_coalition:
            if chosen_action == self._active_coalition.action:
                self._active_coalition.survived_steps += 1
            else:
                self._active_coalition = None  # coalition dissolved

        # Compute alignment score
        alignment = self._compute_alignment_score(state, credit_deltas)

        # Update coalition status vector [strength per minister slot]
        coalition_status = self._get_coalition_status()

        # Advance step
        self._prev_state = dict(state)
        self._step += 1

        # Detect betrayal: a minister who was in the previous coalition now vetoes
        betrayal_occurred = False
        if vetoes and self._active_coalition:
            for v_name in vetoes:
                if v_name in self._active_coalition.members:
                    betrayal_occurred = True
                    break

        return {
            "action": chosen_action,
            "votes": votes,
            "vetoes": vetoes,
            "coalition_formed": self._active_coalition is not None,
            "coalition_strength": self._active_coalition.strength if self._active_coalition else 0.0,
            "proposals": {m.role.name: m.propose_action(state, CORE_ACTIONS, self._rng, utility_volatility)
                          for m in self._ministers},
            "credit_deltas": credit_deltas,
            "influence_vector": [round(m.influence, 4) for m in self._ministers],
            "action_history": list(self._action_history),
            "alignment_score": round(alignment, 2),
            "coalition_status": coalition_status,
            "institutional_trust": round(self._institutional_trust, 4),
            "veto_counts": {m.role.name: m.veto_count for m in self._ministers},
            "betrayal_occurred": betrayal_occurred,
        }

    # -----------------------------------------------------------------
    # Negotiation
    # -----------------------------------------------------------------

    def _negotiate(
        self,
        state: Dict[str, float],
        utility_volatility: float,
    ) -> Tuple[str, Dict[str, str], List[str]]:
        """
        Run weighted voting with possible vetoes.

        Returns: (chosen_action, votes_dict, vetoes_list)
        """
        # Each minister proposes
        proposals = {}
        for m in self._ministers:
            proposals[m.role.name] = m.propose_action(
                state, CORE_ACTIONS, self._rng, utility_volatility
            )

        # Weighted tally
        action_weights: Dict[str, float] = {}
        for m in self._ministers:
            action = proposals[m.role.name]
            action_weights[action] = action_weights.get(action, 0.0) + m.influence

        # Find top action
        top_action = max(action_weights, key=lambda a: action_weights[a])
        top_weight = action_weights[top_action]
        total_influence = sum(m.influence for m in self._ministers)

        # Check vetoes
        vetoes = []
        for m in self._ministers:
            if proposals[m.role.name] != top_action:
                if m.will_veto(top_action, state, self._chaos_level, utility_volatility, self._rng):
                    vetoes.append(m.role.name)
                    # Veto reduces effective weight
                    top_weight -= m.influence * 0.7

        # Coalition check
        if top_weight / max(total_influence, 0.01) >= self.COALITION_THRESHOLD:
            # Coalition forms
            coalition_members = [m.role.name for m in self._ministers
                                  if proposals[m.role.name] == top_action]
            if not (self._active_coalition and
                    self._active_coalition.action == top_action and
                    set(self._active_coalition.members) == set(coalition_members)):
                self._coalition_counter += 1
                self._active_coalition = Coalition(
                    coalition_id=self._coalition_counter,
                    members=coalition_members,
                    action=top_action,
                    strength=top_weight / max(total_influence, 0.01),
                    formed_at_step=self._step,
                )
                self._coalitions.append(self._active_coalition)
        else:
            # No coalition consensus — pick highest individual proposal if vetoes block
            if vetoes:
                # Fallback: second-best action
                sorted_actions = sorted(action_weights, key=lambda a: action_weights[a], reverse=True)
                top_action = sorted_actions[1] if len(sorted_actions) > 1 else sorted_actions[0]
            self._active_coalition = None

        return top_action, proposals, vetoes

    def _emergency_coalition_vote(
        self, state: Dict[str, float], utility_volatility: float
    ) -> Tuple[str, Dict[str, str], List[str]]:
        """Force immediate coalition on the single highest-urgency action."""
        # Find the action with highest average minister utility
        action_scores: Dict[str, float] = {}
        for action in CORE_ACTIONS:
            score = sum(m._score_action(action, state) for m in self._ministers)
            action_scores[action] = score / max(len(self._ministers), 1)

        emergency_action = max(action_scores, key=lambda a: action_scores[a])
        votes = {m.role.name: emergency_action for m in self._ministers}

        # Form mandatory coalition
        total_influence = sum(m.influence for m in self._ministers)
        self._coalition_counter += 1
        self._active_coalition = Coalition(
            coalition_id=self._coalition_counter,
            members=[m.role.name for m in self._ministers],
            action=emergency_action,
            strength=min(1.0, total_influence),
            formed_at_step=self._step,
        )
        self._coalitions.append(self._active_coalition)

        return emergency_action, votes, []

    # -----------------------------------------------------------------
    # Global Policy Package
    # -----------------------------------------------------------------

    def propose_policy_package(
        self, state: Dict[str, float], n_actions: int = 2
    ) -> List[str]:
        """
        Bundle the top N highest-consensus actions into a policy package.
        Returns list of 2–3 action names.
        """
        action_scores: Dict[str, float] = {}
        for action in CORE_ACTIONS:
            score = sum(m._score_action(action, state) for m in self._ministers)
            action_scores[action] = score / max(len(self._ministers), 1)

        sorted_actions = sorted(action_scores, key=lambda a: action_scores[a], reverse=True)
        return sorted_actions[:min(n_actions, 3)]

    # -----------------------------------------------------------------
    # Credit Assignment
    # -----------------------------------------------------------------

    def _assign_credit(
        self,
        action: str,
        state: Dict[str, float],
        votes: Dict[str, str],
    ) -> Dict[str, float]:
        """
        Assign credit to each minister based on:
        - Whether they voted for the chosen action
        - Quality of the action relative to their utility vector
        """
        credits: Dict[str, float] = {}
        for m in self._ministers:
            voted_for = votes.get(m.role.name, "") == action
            action_quality = max(0.0, m._score_action(action, state))
            # Credit = alignment * quality
            base = action_quality * (1.0 if voted_for else 0.5)
            # Normalise to [0, 1]
            credits[m.role.name] = round(max(0.0, min(1.0, base)), 4)
        return credits

    # -----------------------------------------------------------------
    # Alignment Score
    # -----------------------------------------------------------------

    def _compute_alignment_score(
        self,
        state: Dict[str, float],
        credit_deltas: Dict[str, float],
    ) -> float:
        """
        Alignment Score (0–100): balance between individual utilities and global objectives.

        High alignment = ministers' individual gains are correlated with global welfare.
        """
        # Global welfare proxy: average of key metrics normalised
        gdp_norm = state.get("gdp_index", 100) / 200.0
        env_norm = 1.0 - state.get("pollution_index", 100) / 300.0
        soc_norm = state.get("public_satisfaction", 50) / 100.0
        global_welfare = (gdp_norm + env_norm + soc_norm) / 3.0

        # Average credit (individual utility gain proxy)
        avg_credit = sum(credit_deltas.values()) / max(len(credit_deltas), 1)

        # Alignment = how well individual credit tracks global welfare
        alignment = 1.0 - abs(avg_credit - global_welfare)
        return max(0.0, min(1.0, alignment)) * 100.0

    # -----------------------------------------------------------------
    # Status Accessors
    # -----------------------------------------------------------------

    def _get_coalition_status(self) -> List[float]:
        """Return coalition strength per minister slot (0 if not in coalition)."""
        status = []
        for m in self._ministers:
            if self._active_coalition and m.role.name in self._active_coalition.members:
                status.append(round(self._active_coalition.strength, 4))
            else:
                status.append(0.0)
        # Pad to 5 if fewer than 5 ministers
        while len(status) < 5:
            status.append(0.0)
        return status

    def get_influence_vector(self) -> List[float]:
        """Return [0..1] influence for all 5 minister slots."""
        vec = [round(m.influence, 4) for m in self._ministers]
        while len(vec) < 5:
            vec.append(0.0)
        return vec

    def get_action_history_encoded(self) -> List[float]:
        """Return last 8 actions encoded as float IDs (normalised)."""
        all_actions = VALID_ACTIONS
        n = len(all_actions)
        encoded = []
        for action in self._action_history[-8:]:
            idx = all_actions.index(action) if action in all_actions else 0
            encoded.append(idx / max(n - 1, 1))
        # Pad to 8
        while len(encoded) < 8:
            encoded.append(0.0)
        return encoded

    def get_coalition_survival_ratio(self) -> float:
        """Ratio of steps where an active coalition was in place."""
        if not self._coalitions:
            return 0.0
        total_survived = sum(c.survived_steps for c in self._coalitions)
        return total_survived / max(self._step, 1)

    def update_institutional_trust(self, delta: float) -> None:
        """Update institutional trust (called by environment on special actions)."""
        self._institutional_trust = max(0.0, min(1.0, self._institutional_trust + delta))

    def get_episode_summary(self) -> Dict:
        """Return episode-level council summary for JSONL export."""
        return {
            "coalitions_formed": len(self._coalitions),
            "total_vetoes": sum(m.veto_count for m in self._ministers),
            "veto_by_minister": {m.role.name: m.veto_count for m in self._ministers},
            "avg_influence_stability": round(sum(m.influence for m in self._ministers) / max(len(self._ministers), 1), 4),
            "influence_vector": self.get_influence_vector(),
            "coalition_survival_ratio": round(self.get_coalition_survival_ratio(), 4),
            "institutional_trust_final": round(self._institutional_trust, 4),
        }
