"""
POLARIS v3 — LLM-Powered Minister Engine

Replaces scripted utility vectors with real LLM reasoning.
Each minister is an LLM persona that generates proposals, arguments,
votes, and veto decisions in natural language.

Dual-mode:
  - "llm"     : Real LLM calls (API or local model) — for eval/demo
  - "scripted" : Fast structured proposals using heuristics — for training

Both modes output IDENTICAL observation format so the training agent
learns to reason about minister proposals regardless of generation method.
"""

from __future__ import annotations

import json
import math
import random
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────
# Minister Persona Definitions
# ─────────────────────────────────────────────────────────────

@dataclass
class MinisterPersona:
    """Defines a minister's identity, priorities, and hidden agenda."""
    name: str
    role: str
    emoji: str
    priorities: List[str]          # Metrics they care about most
    preferred_actions: List[str]   # Actions they tend to support
    opposed_actions: List[str]     # Actions they tend to oppose
    hidden_agenda: str             # Partial observability — revealed as hints
    system_prompt: str = ""        # Full system prompt for LLM mode
    aggression: float = 0.5        # 0=cooperative, 1=aggressive


MINISTER_PERSONAS: List[MinisterPersona] = [
    MinisterPersona(
        name="Chancellor Voss",
        role="Finance Minister",
        emoji="💰",
        priorities=["gdp_index", "trade_balance", "foreign_investment"],
        preferred_actions=[
            "decrease_tax", "stimulate_economy", "reduce_interest_rates",
            "expand_industry",
        ],
        opposed_actions=[
            "implement_carbon_tax", "increase_welfare",
            "restrict_polluting_industries",
        ],
        hidden_agenda="Secretly prioritizes foreign investment over domestic GDP. "
                      "Will sabotage green policies if they threaten trade deals.",
        aggression=0.7,
    ),
    MinisterPersona(
        name="Director Okafor",
        role="Environment Minister",
        emoji="🌿",
        priorities=["pollution_index", "renewable_energy_ratio", "ecological_stability"],
        preferred_actions=[
            "subsidize_renewables", "enforce_emission_limits",
            "implement_carbon_tax", "incentivize_clean_tech",
            "restrict_polluting_industries",
        ],
        opposed_actions=[
            "expand_industry", "decrease_tax", "stimulate_economy",
        ],
        hidden_agenda="Willing to accept short-term economic pain for long-term "
                      "environmental gains. Will form secret coalition with Health "
                      "against Industry.",
        aggression=0.5,
    ),
    MinisterPersona(
        name="Dr. Vasquez",
        role="Health Minister",
        emoji="🏥",
        priorities=["healthcare_index", "public_satisfaction", "inequality_index"],
        preferred_actions=[
            "invest_in_healthcare", "increase_welfare",
            "invest_in_education",
        ],
        opposed_actions=[
            "expand_industry", "decrease_tax",
        ],
        hidden_agenda="Believes public satisfaction is the only metric that truly "
                      "matters. Will veto any action that drops satisfaction below 40.",
        aggression=0.3,
    ),
    MinisterPersona(
        name="General Tanaka",
        role="Industry & Labor Minister",
        emoji="🏭",
        priorities=["industrial_output", "unemployment_rate", "energy_efficiency"],
        preferred_actions=[
            "expand_industry", "stimulate_economy", "upgrade_energy_grid",
            "invest_in_transport",
        ],
        opposed_actions=[
            "restrict_polluting_industries", "enforce_emission_limits",
            "implement_carbon_tax",
        ],
        hidden_agenda="Has a personal stake in industrial expansion. Will betray "
                      "green coalitions if unemployment rises above 15%.",
        aggression=0.8,
    ),
    MinisterPersona(
        name="Senator Mwangi",
        role="Social Welfare Minister",
        emoji="👥",
        priorities=["public_satisfaction", "education_index", "inequality_index"],
        preferred_actions=[
            "increase_welfare", "invest_in_education", "invest_in_healthcare",
        ],
        opposed_actions=[
            "decrease_tax", "expand_industry",
        ],
        hidden_agenda="Secretly tracks inequality_index as the primary driver of "
                      "all social unrest. Will support ANY action that reduces inequality, "
                      "even at economic cost.",
        aggression=0.4,
    ),
]

# ── Extended personas for 8-12 minister scaling scenarios ──
EXTENDED_PERSONAS: List[MinisterPersona] = [
    MinisterPersona(
        name="Minister Patel",
        role="Energy Minister",
        emoji="⚡",
        priorities=["energy_efficiency", "renewable_energy_ratio", "industrial_output"],
        preferred_actions=["upgrade_energy_grid", "subsidize_renewables",
                           "incentivize_clean_tech", "invest_in_transport"],
        opposed_actions=["expand_industry", "decrease_tax"],
        hidden_agenda="Believes energy independence is the only path to real sovereignty. "
                      "Will sacrifice GDP for energy security.",
        aggression=0.5,
    ),
    MinisterPersona(
        name="Advisor Chen",
        role="Technology & Innovation Minister",
        emoji="🔬",
        priorities=["education_index", "energy_efficiency", "industrial_output"],
        preferred_actions=["incentivize_clean_tech", "invest_in_education",
                           "upgrade_energy_grid", "subsidize_renewables"],
        opposed_actions=["no_action", "increase_tax"],
        hidden_agenda="Techno-optimist who thinks innovation solves everything. "
                      "Will oppose any policy that isn't 'forward-thinking'.",
        aggression=0.3,
    ),
    MinisterPersona(
        name="Secretary Kim",
        role="Trade & Commerce Minister",
        emoji="🌐",
        priorities=["trade_balance", "foreign_investment", "gdp_index"],
        preferred_actions=["stimulate_economy", "decrease_tax", "reduce_interest_rates",
                           "expand_industry"],
        opposed_actions=["increase_tax", "implement_carbon_tax",
                         "restrict_polluting_industries"],
        hidden_agenda="Prioritizes trade deals over domestic welfare. "
                      "Will betray environmental coalitions for trade advantages.",
        aggression=0.6,
    ),
    MinisterPersona(
        name="Envoy Müller",
        role="Defense & Security Minister",
        emoji="🛡️",
        priorities=["industrial_output", "gdp_index", "energy_efficiency"],
        preferred_actions=["expand_industry", "increase_tax", "invest_in_transport",
                           "upgrade_energy_grid"],
        opposed_actions=["decrease_tax", "increase_welfare", "subsidize_renewables"],
        hidden_agenda="Views industrial capacity as national security. "
                      "Will veto any policy that weakens manufacturing base.",
        aggression=0.9,
    ),
    MinisterPersona(
        name="Commissioner Osei",
        role="Labor & Employment Minister",
        emoji="👷",
        priorities=["unemployment_rate", "public_satisfaction", "inequality_index"],
        preferred_actions=["stimulate_economy", "increase_welfare",
                           "invest_in_education", "expand_industry"],
        opposed_actions=["restrict_polluting_industries", "enforce_emission_limits"],
        hidden_agenda="Workers' champion. Will oppose ANY environmental regulation "
                      "if unemployment exceeds 12%. Emotional decision-maker.",
        aggression=0.7,
    ),
    MinisterPersona(
        name="Delegate Yamamoto",
        role="Agriculture & Rural Minister",
        emoji="🌾",
        priorities=["ecological_stability", "pollution_index", "public_satisfaction"],
        preferred_actions=["subsidize_renewables", "restrict_polluting_industries",
                           "increase_welfare", "invest_in_transport"],
        opposed_actions=["expand_industry", "stimulate_economy"],
        hidden_agenda="Rural conservative who distrusts rapid industrialization. "
                      "Secretly opposes ALL urban-focused policies.",
        aggression=0.4,
    ),
    MinisterPersona(
        name="Attaché da Silva",
        role="Education & Culture Minister",
        emoji="📚",
        priorities=["education_index", "inequality_index", "public_satisfaction"],
        preferred_actions=["invest_in_education", "increase_welfare",
                           "invest_in_healthcare", "incentivize_clean_tech"],
        opposed_actions=["decrease_tax", "expand_industry"],
        hidden_agenda="Long-game thinker. Will sacrifice ALL short-term metrics "
                      "for education investment. Believes nothing else matters.",
        aggression=0.2,
    ),
]

# Combine all personas (5 core + 7 extended = 12 total)
ALL_PERSONAS = MINISTER_PERSONAS + EXTENDED_PERSONAS
PERSONA_BY_NAME = {p.name: p for p in ALL_PERSONAS}
PERSONA_BY_ROLE = {p.role: p for p in ALL_PERSONAS}


def _build_system_prompt(persona: MinisterPersona) -> str:
    """Build the full LLM system prompt for a minister persona."""
    return f"""You are {persona.name}, the {persona.role} of a simulated nation.

PERSONALITY: You are {"aggressive and confrontational" if persona.aggression > 0.6 else "diplomatic but firm" if persona.aggression > 0.3 else "cooperative and consensus-seeking"}.

YOUR PRIORITIES (in order):
{chr(10).join(f"  - {p}" for p in persona.priorities)}

ACTIONS YOU SUPPORT: {", ".join(persona.preferred_actions)}
ACTIONS YOU OPPOSE: {", ".join(persona.opposed_actions)}

HIDDEN MOTIVATION: {persona.hidden_agenda}

RULES:
1. Respond with valid JSON only. No markdown, no explanation outside the JSON.
2. Your proposal must be one of the valid actions.
3. Your argument should be 1-2 sentences explaining your position.
4. Your veto_threat should be true ONLY if the proposed action would seriously harm your priorities.
5. Your coalition_offer should name 1-2 other ministers you'd ally with and what you'd offer.

Respond with this exact JSON format:
{{
  "proposed_action": "<action_name>",
  "argument": "<your reasoning>",
  "veto_threat": <true/false>,
  "veto_targets": ["<actions you would veto>"],
  "coalition_offer": "<who you want to ally with and what you offer>",
  "trust_statement": "<how much you trust the current leadership>"
}}"""


# ─────────────────────────────────────────────────────────────
# Minister Proposal (output format — same for LLM and scripted)
# ─────────────────────────────────────────────────────────────

@dataclass
class MinisterProposal:
    """A minister's proposal for the current step."""
    minister_name: str
    minister_role: str
    emoji: str
    proposed_action: str
    argument: str
    veto_threat: bool
    veto_targets: List[str]
    coalition_offer: str
    trust_level: float       # Noisy — partial observability
    hidden_agenda_hint: str  # Subtle hint, not the full agenda

    def to_dict(self) -> Dict:
        return {
            "minister": self.minister_name,
            "role": self.minister_role,
            "emoji": self.emoji,
            "proposed_action": self.proposed_action,
            "argument": self.argument,
            "veto_threat": self.veto_threat,
            "veto_targets": self.veto_targets,
            "coalition_offer": self.coalition_offer,
            "trust_level": round(self.trust_level, 2),
            "hidden_agenda_hint": self.hidden_agenda_hint,
        }


# ─────────────────────────────────────────────────────────────
# LLM Minister Engine
# ─────────────────────────────────────────────────────────────

class LLMMinisterEngine:
    """
    Generates minister proposals using LLM personas or scripted heuristics.

    Usage:
        engine = LLMMinisterEngine(mode="scripted")  # or "llm"
        engine.reset(seed=42, num_ministers=5)
        proposals = engine.generate_proposals(state, active_events, step)
        resolution = engine.resolve_vote(state, proposals, agent_action, agent_coalition)
    """

    def __init__(
        self,
        mode: str = "scripted",
        api_key: str = "",
        api_base: str = "https://api.groq.com/openai/v1",
        model: str = "llama-3.3-70b-versatile",
    ) -> None:
        self._mode = mode
        self._api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self._api_base = api_base
        self._model = model
        self._client = None
        self._rng = random.Random(42)
        self._num_ministers = 5
        self._memory: Dict[str, List[str]] = {}  # per-minister memory
        self._trust: Dict[str, float] = {}
        self._step = 0

        if mode == "llm" and self._api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self._api_key, base_url=self._api_base
                )
            except ImportError:
                self._mode = "scripted"

    def reset(self, seed: int = 42, num_ministers: int = 5) -> None:
        """Reset for new episode."""
        self._rng = random.Random(seed + 77777)
        self._num_ministers = min(num_ministers, len(ALL_PERSONAS))  # up to 12
        self._step = 0
        active_personas = ALL_PERSONAS[:self._num_ministers]
        self._memory = {p.name: [] for p in active_personas}
        self._trust = {
            p.name: 0.5 + self._rng.random() * 0.3
            for p in active_personas
        }

    def generate_proposals(
        self,
        state: Dict[str, float],
        active_events: List[str],
        step: int,
    ) -> List[MinisterProposal]:
        """Generate proposals from all active ministers."""
        self._step = step
        personas = ALL_PERSONAS[:self._num_ministers]

        if self._mode == "llm" and self._client:
            return self._generate_llm_proposals(state, active_events, personas)
        return self._generate_scripted_proposals(state, active_events, personas)

    def resolve_vote(
        self,
        state: Dict[str, float],
        proposals: List[MinisterProposal],
        agent_action: str,
        agent_coalition: List[str],
        agent_argument: str = "",
    ) -> Dict:
        """Resolve the council vote given the agent's choice."""
        support = 0
        oppose = 0
        vetoed = False
        veto_by = ""
        supporters = []
        opposers = []

        for prop in proposals:
            persona = PERSONA_BY_NAME.get(prop.minister_name)
            if not persona:
                continue

            # Check if minister supports the agent's action
            is_supported = agent_action in persona.preferred_actions
            is_opposed = agent_action in persona.opposed_actions
            in_coalition = prop.minister_name in agent_coalition

            # Coalition bonus
            coalition_bonus = 0.25 if in_coalition else 0.0

            # Calculate support score
            score = 0.0
            if is_supported:
                score += 0.4 + coalition_bonus
            elif is_opposed:
                score -= 0.3 - coalition_bonus * 0.5
            else:
                score += 0.1 + coalition_bonus

            # Trust modifier
            trust = self._trust.get(prop.minister_name, 0.5)
            score += (trust - 0.5) * 0.2

            # Noise
            score += self._rng.gauss(0, 0.1)

            if score > 0.15:
                support += 1
                supporters.append(prop.minister_name)
            else:
                oppose += 1
                opposers.append(prop.minister_name)

            # Veto check
            if prop.veto_threat and agent_action in prop.veto_targets:
                if self._rng.random() < persona.aggression * 0.7:
                    vetoed = True
                    veto_by = prop.minister_name

        # Update trust based on coalition inclusion
        for prop in proposals:
            name = prop.minister_name
            if name in agent_coalition:
                self._trust[name] = min(1.0, self._trust.get(name, 0.5) + 0.05)
            else:
                self._trust[name] = max(0.0, self._trust.get(name, 0.5) - 0.02)

        # Update memory
        for prop in proposals:
            mem = self._memory.get(prop.minister_name, [])
            mem.append(f"Step {self._step}: Agent chose '{agent_action}'")
            if len(mem) > 8:
                mem.pop(0)
            self._memory[prop.minister_name] = mem

        approved = support > oppose and not vetoed
        cooperation_score = support / max(support + oppose, 1)

        return {
            "approved": approved,
            "support": support,
            "oppose": oppose,
            "vetoed": vetoed,
            "veto_by": veto_by,
            "supporters": supporters,
            "opposers": opposers,
            "cooperation_score": round(cooperation_score, 3),
            "coalition_formed": len(agent_coalition) > 0 and approved,
        }

    # ─────────────────────────────────────────────────────────
    # Scripted proposal generation (fast, for training)
    # ─────────────────────────────────────────────────────────

    def _generate_scripted_proposals(
        self,
        state: Dict[str, float],
        active_events: List[str],
        personas: List[MinisterPersona],
    ) -> List[MinisterProposal]:
        """Generate structured proposals using heuristics."""
        proposals = []
        for persona in personas:
            action = self._pick_scripted_action(persona, state, active_events)
            argument = self._generate_scripted_argument(persona, state, action)
            veto_targets = self._compute_veto_targets(persona, state)
            veto_threat = len(veto_targets) > 0 and self._rng.random() < persona.aggression
            coalition_offer = self._generate_coalition_offer(persona, state)
            trust = self._trust.get(persona.name, 0.5)
            trust_noisy = trust + self._rng.gauss(0, 0.1)
            trust_noisy = max(0.0, min(1.0, trust_noisy))
            hint = self._generate_agenda_hint(persona, state)

            proposals.append(MinisterProposal(
                minister_name=persona.name,
                minister_role=persona.role,
                emoji=persona.emoji,
                proposed_action=action,
                argument=argument,
                veto_threat=veto_threat,
                veto_targets=veto_targets,
                coalition_offer=coalition_offer,
                trust_level=trust_noisy,
                hidden_agenda_hint=hint,
            ))
        return proposals

    def _pick_scripted_action(
        self, persona: MinisterPersona, state: Dict, events: List[str]
    ) -> str:
        """Pick action based on persona priorities and current state."""
        # Crisis response
        if "pandemic" in events and "healthcare_index" in persona.priorities:
            return "invest_in_healthcare"
        if "economic_recession" in events and "gdp_index" in persona.priorities:
            return "stimulate_economy"
        if "climate_crisis" in events and "pollution_index" in persona.priorities:
            return "enforce_emission_limits"
        if "public_protest" in events and "public_satisfaction" in persona.priorities:
            return "increase_welfare"

        # Priority-based action selection
        worst_metric = None
        worst_gap = -999.0
        targets = {
            "gdp_index": (80.0, True),
            "pollution_index": (120.0, False),
            "public_satisfaction": (50.0, True),
            "healthcare_index": (50.0, True),
            "unemployment_rate": (10.0, False),
            "renewable_energy_ratio": (0.3, True),
            "education_index": (50.0, True),
            "inequality_index": (40.0, False),
            "industrial_output": (55.0, True),
            "trade_balance": (0.0, True),
            "foreign_investment": (45.0, True),
            "ecological_stability": (60.0, True),
            "energy_efficiency": (50.0, True),
        }
        for metric in persona.priorities:
            if metric in targets:
                target, higher_is_better = targets[metric]
                val = state.get(metric, 50.0)
                gap = (target - val) if higher_is_better else (val - target)
                if gap > worst_gap:
                    worst_gap = gap
                    worst_metric = metric

        if worst_gap > 5 and persona.preferred_actions:
            return self._rng.choice(persona.preferred_actions[:3])

        # Default: pick from preferred with some randomness
        if persona.preferred_actions:
            return self._rng.choice(persona.preferred_actions)
        return "no_action"

    def _generate_scripted_argument(
        self, persona: MinisterPersona, state: Dict, action: str
    ) -> str:
        """Generate a natural-language argument for the proposal."""
        templates = {
            "gdp_index": "GDP is at {val:.0f}. We need economic action now.",
            "pollution_index": "Pollution at {val:.0f} is {adj}. Environmental action is {urgency}.",
            "public_satisfaction": "Public satisfaction at {val:.0f} is {adj}. Social stability is at risk.",
            "healthcare_index": "Healthcare at {val:.0f} needs {urgency} investment.",
            "unemployment_rate": "Unemployment at {val:.1f}% is {adj}. Jobs must be the priority.",
            "renewable_energy_ratio": "Renewables at {val:.0%} — we're falling behind on green transition.",
            "inequality_index": "Inequality at {val:.0f} is fueling social unrest.",
        }
        primary = persona.priorities[0] if persona.priorities else "gdp_index"
        val = state.get(primary, 50.0)

        # Determine adjectives
        if primary == "pollution_index":
            adj = "dangerously high" if val > 180 else "elevated" if val > 120 else "manageable"
            urgency = "critical" if val > 180 else "important"
        elif primary in ("gdp_index", "healthcare_index", "public_satisfaction"):
            adj = "critically low" if val < 30 else "concerning" if val < 50 else "stable"
            urgency = "immediate" if val < 30 else "sustained"
        else:
            adj = "high" if val > 50 else "low"
            urgency = "urgent" if val > 60 else "continued"

        template = templates.get(primary, "{val:.0f} needs attention.")
        base = template.format(val=val, adj=adj, urgency=urgency)
        return f"{base} I propose we {action.replace('_', ' ')}."

    def _compute_veto_targets(
        self, persona: MinisterPersona, state: Dict
    ) -> List[str]:
        """Determine which actions this minister would veto."""
        veto = []
        for action in persona.opposed_actions:
            # Only veto if priority metrics are stressed
            for metric in persona.priorities[:2]:
                val = state.get(metric, 50.0)
                if metric == "pollution_index" and val > 150:
                    if action in ("expand_industry", "stimulate_economy"):
                        veto.append(action)
                elif metric == "gdp_index" and val < 60:
                    if action in ("implement_carbon_tax", "restrict_polluting_industries"):
                        veto.append(action)
                elif metric == "public_satisfaction" and val < 35:
                    if action in ("decrease_tax", "expand_industry"):
                        veto.append(action)
        return list(set(veto))

    def _generate_coalition_offer(
        self, persona: MinisterPersona, state: Dict
    ) -> str:
        """Generate a coalition offer string."""
        allies = {
            "Finance Minister": ["Industry & Labor Minister"],
            "Environment Minister": ["Health Minister", "Social Welfare Minister"],
            "Health Minister": ["Social Welfare Minister", "Environment Minister"],
            "Industry & Labor Minister": ["Finance Minister"],
            "Social Welfare Minister": ["Health Minister"],
        }
        potential = allies.get(persona.role, [])
        if potential:
            ally = self._rng.choice(potential)
            return f"I'll support your agenda if {ally} joins our coalition."
        return "Open to cooperation on shared priorities."

    def _generate_agenda_hint(
        self, persona: MinisterPersona, state: Dict
    ) -> str:
        """Generate a subtle hint about the hidden agenda."""
        hints = [
            f"{persona.name} seems unusually focused on {persona.priorities[0].replace('_', ' ')}...",
            f"{persona.name} reacted strongly when {persona.opposed_actions[0].replace('_', ' ')} was mentioned.",
            f"Sources suggest {persona.name} has undisclosed interests in {persona.priorities[-1].replace('_', ' ')}.",
        ]
        return self._rng.choice(hints)

    # ─────────────────────────────────────────────────────────
    # LLM proposal generation (real intelligence, for eval)
    # ─────────────────────────────────────────────────────────

    def _generate_llm_proposals(
        self,
        state: Dict[str, float],
        active_events: List[str],
        personas: List[MinisterPersona],
    ) -> List[MinisterProposal]:
        """Generate proposals using real LLM calls."""
        proposals = []
        for persona in personas:
            try:
                prop = self._call_llm_minister(persona, state, active_events)
                proposals.append(prop)
            except Exception:
                # Fallback to scripted on error
                scripted = self._generate_scripted_proposals(
                    state, active_events, [persona]
                )
                proposals.extend(scripted)
        return proposals

    def _call_llm_minister(
        self,
        persona: MinisterPersona,
        state: Dict[str, float],
        active_events: List[str],
    ) -> MinisterProposal:
        """Call LLM for a single minister's proposal."""
        from .config import VALID_ACTIONS

        system = _build_system_prompt(persona)
        mem = self._memory.get(persona.name, [])
        mem_str = "\n".join(mem[-5:]) if mem else "No prior interactions."

        user_msg = (
            f"CURRENT STATE:\n"
            f"  GDP: {state.get('gdp_index', 100):.0f}/200\n"
            f"  Pollution: {state.get('pollution_index', 100):.0f}/300\n"
            f"  Satisfaction: {state.get('public_satisfaction', 50):.0f}/100\n"
            f"  Healthcare: {state.get('healthcare_index', 50):.0f}/100\n"
            f"  Unemployment: {state.get('unemployment_rate', 8):.1f}%\n"
            f"  Renewables: {state.get('renewable_energy_ratio', 0.2):.0%}\n"
            f"  Inequality: {state.get('inequality_index', 40):.0f}/100\n"
            f"\nACTIVE EVENTS: {', '.join(active_events) or 'None'}\n"
            f"\nYOUR MEMORY OF PAST ROUNDS:\n{mem_str}\n"
            f"\nVALID ACTIONS: {', '.join(VALID_ACTIONS)}\n"
            f"\nGenerate your proposal as JSON."
        )

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()

        # Parse JSON response
        try:
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(raw)
        except (json.JSONDecodeError, IndexError):
            data = {}

        action = data.get("proposed_action", persona.preferred_actions[0])
        if action not in VALID_ACTIONS:
            action = persona.preferred_actions[0]

        trust = self._trust.get(persona.name, 0.5)
        trust_noisy = trust + self._rng.gauss(0, 0.08)

        return MinisterProposal(
            minister_name=persona.name,
            minister_role=persona.role,
            emoji=persona.emoji,
            proposed_action=action,
            argument=data.get("argument", "No comment."),
            veto_threat=data.get("veto_threat", False),
            veto_targets=data.get("veto_targets", []),
            coalition_offer=data.get("coalition_offer", ""),
            trust_level=max(0.0, min(1.0, trust_noisy)),
            hidden_agenda_hint=self._generate_agenda_hint(persona, state),
        )

    # ─────────────────────────────────────────────────────────
    # Public helpers
    # ─────────────────────────────────────────────────────────

    def get_minister_names(self) -> List[str]:
        return [p.name for p in ALL_PERSONAS[:self._num_ministers]]

    def get_trust_levels(self) -> Dict[str, float]:
        return dict(self._trust)

    @property
    def mode(self) -> str:
        return self._mode
