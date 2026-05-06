"""
POLARIS v3 — Multi-Agent Negotiation Protocol

Implements the 3-phase negotiation that makes the LLM interact with
minister agents each step:

  Phase 1: PROPOSE  — Ministers generate proposals (returned to agent)
  Phase 2: NEGOTIATE — Agent reads proposals, outputs structured response
  Phase 3: RESOLVE  — Council votes, vetoes, coalition outcomes computed

The observation includes the full negotiation context so the training
agent learns theory-of-mind reasoning about other agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .llm_minister import LLMMinisterEngine, MinisterProposal


@dataclass
class NegotiationContext:
    """Full negotiation context included in the agent's observation."""
    minister_proposals: List[Dict]
    active_coalitions: List[str]
    institutional_trust: float
    recent_betrayals: List[str]
    step: int
    diplomatic_briefing: str = ""

    def to_dict(self) -> Dict:
        return {
            "minister_proposals": self.minister_proposals,
            "active_coalitions": self.active_coalitions,
            "institutional_trust": round(self.institutional_trust, 3),
            "recent_betrayals": self.recent_betrayals,
            "step": self.step,
            "diplomatic_briefing": self.diplomatic_briefing,
        }

    def to_narrative(self) -> str:
        """Format as natural language for LLM consumption."""
        lines = [f"=== COUNCIL NEGOTIATION — Step {self.step} ===\n"]
        for p in self.minister_proposals:
            veto = " ⚠️ VETO THREAT" if p.get("veto_threat") else ""
            lines.append(
                f"{p.get('emoji', '👤')} {p['minister']} ({p['role']}):{veto}\n"
                f"  Proposal: {p['proposed_action'].replace('_', ' ')}\n"
                f"  Argument: {p['argument']}\n"
                f"  Coalition: {p['coalition_offer']}\n"
                f"  Trust: {p['trust_level']:.0%}\n"
                f"  Intel: {p.get('hidden_agenda_hint', '')}\n"
            )
        if self.recent_betrayals:
            lines.append(f"⚠️ RECENT BETRAYALS: {'; '.join(self.recent_betrayals)}\n")
        if self.diplomatic_briefing:
            lines.append(f"📋 BRIEFING: {self.diplomatic_briefing}\n")
        lines.append(f"Institutional Trust: {self.institutional_trust:.0%}\n")
        return "\n".join(lines)


@dataclass
class AgentResponse:
    """Structured response from the training agent."""
    action: str
    reasoning: str = ""
    coalition_target: List[str] = field(default_factory=list)
    negotiation_argument: str = ""
    veto_prediction: List[str] = field(default_factory=list)
    stance: str = "cooperative"  # "aggressive" / "cooperative" / "deceptive"


@dataclass
class NegotiationOutcome:
    """Result of the council vote after negotiation."""
    final_action: str
    approved: bool
    support_count: int
    oppose_count: int
    vetoed: bool
    veto_by: str
    coalition_formed: bool
    cooperation_score: float
    supporters: List[str]
    opposers: List[str]
    # Theory-of-mind scoring
    veto_prediction_correct: bool
    tom_reward: float

    def to_dict(self) -> Dict:
        return {
            "final_action": self.final_action,
            "approved": self.approved,
            "support_count": self.support_count,
            "oppose_count": self.oppose_count,
            "vetoed": self.vetoed,
            "veto_by": self.veto_by,
            "coalition_formed": self.coalition_formed,
            "cooperation_score": round(self.cooperation_score, 3),
            "supporters": self.supporters,
            "opposers": self.opposers,
            "veto_prediction_correct": self.veto_prediction_correct,
            "tom_reward": round(self.tom_reward, 4),
        }


class NegotiationProtocol:
    """
    Orchestrates the 3-phase negotiation between agent and ministers.

    Usage:
        protocol = NegotiationProtocol(mode="scripted")
        protocol.reset(seed=42, num_ministers=5)

        # Phase 1: Get proposals
        context = protocol.phase_propose(state, active_events, step)

        # Phase 2: Agent reads context.to_narrative() and decides
        response = AgentResponse(action="subsidize_renewables", ...)

        # Phase 3: Resolve
        outcome = protocol.phase_resolve(state, response)
    """

    def __init__(self, mode: str = "scripted", **kwargs) -> None:
        self._minister_engine = LLMMinisterEngine(mode=mode, **kwargs)
        self._last_proposals: List[MinisterProposal] = []
        self._betrayal_history: List[str] = []
        self._coalition_history: List[List[str]] = []

    def reset(self, seed: int = 42, num_ministers: int = 5) -> None:
        """Reset for new episode."""
        self._minister_engine.reset(seed=seed, num_ministers=num_ministers)
        self._last_proposals = []
        self._betrayal_history = []
        self._coalition_history = []

    def phase_propose(
        self,
        state: Dict[str, float],
        active_events: List[str],
        step: int,
        briefing: str = "",
    ) -> NegotiationContext:
        """Phase 1: Ministers generate proposals."""
        self._last_proposals = self._minister_engine.generate_proposals(
            state, active_events, step
        )
        trusts = self._minister_engine.get_trust_levels()
        avg_trust = sum(trusts.values()) / max(len(trusts), 1)

        return NegotiationContext(
            minister_proposals=[p.to_dict() for p in self._last_proposals],
            active_coalitions=[
                n for n in self._minister_engine.get_minister_names()
                if trusts.get(n, 0) > 0.6
            ],
            institutional_trust=avg_trust,
            recent_betrayals=self._betrayal_history[-3:],
            step=step,
            diplomatic_briefing=briefing,
        )

    def phase_resolve(
        self,
        state: Dict[str, float],
        agent_response: AgentResponse,
    ) -> NegotiationOutcome:
        """Phase 3: Resolve the council vote."""
        result = self._minister_engine.resolve_vote(
            state=state,
            proposals=self._last_proposals,
            agent_action=agent_response.action,
            agent_coalition=agent_response.coalition_target,
            agent_argument=agent_response.negotiation_argument,
        )

        # Theory-of-mind scoring
        actual_opposers = result.get("opposers", [])
        predicted_vetoes = set()
        for v in (agent_response.veto_prediction or []):
            if isinstance(v, str):
                predicted_vetoes.add(v)
            elif isinstance(v, dict):
                predicted_vetoes.add(str(v.get("name", v.get("minister", ""))))
            else:
                predicted_vetoes.add(str(v))
        actual_vetoer = result.get("veto_by", "")

        tom_correct = False
        tom_reward = 0.0

        if actual_vetoer:
            tom_correct = actual_vetoer in predicted_vetoes
            tom_reward = 0.15 if tom_correct else -0.05
        elif predicted_vetoes:
            # Predicted veto but none happened
            tom_reward = -0.03
        else:
            # No prediction, no veto — neutral
            tom_reward = 0.02

        # Coalition quality reward
        if result.get("coalition_formed"):
            tom_reward += 0.08

        # Track betrayals
        if result.get("vetoed") and result.get("veto_by"):
            vetoer = result["veto_by"]
            if vetoer in agent_response.coalition_target:
                self._betrayal_history.append(
                    f"{vetoer} betrayed coalition at step {state.get('step', 0)}"
                )

        self._coalition_history.append(agent_response.coalition_target)

        final_action = agent_response.action
        if result.get("vetoed"):
            # If vetoed, fall back to most-supported minister's proposal
            if self._last_proposals:
                final_action = self._last_proposals[0].proposed_action

        return NegotiationOutcome(
            final_action=final_action,
            approved=result.get("approved", True),
            support_count=result.get("support", 0),
            oppose_count=result.get("oppose", 0),
            vetoed=result.get("vetoed", False),
            veto_by=result.get("veto_by", ""),
            coalition_formed=result.get("coalition_formed", False),
            cooperation_score=result.get("cooperation_score", 0.5),
            supporters=result.get("supporters", []),
            opposers=result.get("opposers", []),
            veto_prediction_correct=tom_correct,
            tom_reward=tom_reward,
        )

    def get_minister_names(self) -> List[str]:
        return self._minister_engine.get_minister_names()
