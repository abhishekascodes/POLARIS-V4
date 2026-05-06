"""
POLARIS-Bench v4 — 20 Benchmark Scenarios
==========================================

Systematic test suite across 5 dimensions:
  - Coordination (can agents work together?)
  - Theory-of-Mind (can agents model other agents?)
  - Long-Horizon Planning (can agents think ahead?)
  - Adversarial Robustness (can agents handle hostile agents?)
  - Scaling (how does agent count affect performance?)

Each scenario produces a normalized 0-1 score with formal grading criteria.
"""

from __future__ import annotations
from typing import Dict, Any

# ═══════════════════════════════════════════════════════════════
# BASE SCENARIO TEMPLATE
# ═══════════════════════════════════════════════════════════════

BASE_STATE = {
    "pollution_index": 130.0, "carbon_emission_rate": 52.0,
    "renewable_energy_ratio": 0.20, "ecological_stability": 62.0,
    "gdp_index": 92.0, "industrial_output": 60.0,
    "unemployment_rate": 8.0, "inflation_rate": 3.5,
    "trade_balance": 2.0, "foreign_investment": 50.0,
    "public_satisfaction": 58.0, "healthcare_index": 52.0,
    "education_index": 50.0, "inequality_index": 40.0,
    "energy_efficiency": 48.0, "transport_efficiency": 48.0,
    "tax_rate": 26.0, "regulation_strength": 38.0,
    "welfare_spending": 28.0, "green_subsidies": 10.0, "interest_rate": 5.5,
}

STRESSED_STATE = {**BASE_STATE,
    "pollution_index": 220.0, "gdp_index": 55.0,
    "public_satisfaction": 25.0, "healthcare_index": 30.0,
    "unemployment_rate": 22.0,
}

HOSTILE_STATE = {**BASE_STATE,
    "pollution_index": 180.0, "gdp_index": 65.0,
    "public_satisfaction": 35.0, "inequality_index": 65.0,
}


# ═══════════════════════════════════════════════════════════════
# ALL 20 SCENARIOS
# ═══════════════════════════════════════════════════════════════

SCENARIOS: Dict[str, Dict[str, Any]] = {

    # ─── DIMENSION 1: COORDINATION (4 scenarios) ───────────────

    "coord_resource_allocation": {
        "id": "coord_resource_allocation",
        "name": "Resource Allocation",
        "dimension": "coordination",
        "difficulty": "medium",
        "description": (
            "5 ministers with competing budget priorities must agree on resource "
            "allocation. The agent must balance economy, environment, and social "
            "welfare simultaneously without letting any single metric collapse."
        ),
        "max_steps": 150,
        "num_ministers": 5,
        "events_enabled": True,
        "event_frequency_multiplier": 0.8,
        "chaos_level": 0.5,
        "drift_enabled": True,
        "negotiation_enabled": True,
        "briefing_enabled": True,
        "minister_mode": "scripted",
        "satisfaction_event_scale": 0.6,
        "satisfaction_floor_damping": 0.6,
        "crisis_welfare_bonus": 6.0,
        "initial_state_overrides": {**BASE_STATE},
        "success_criteria": {
            "min_gdp": 50, "max_pollution": 200,
            "min_satisfaction": 30, "survival_required": True,
        },
    },

    "coord_crisis_response": {
        "id": "coord_crisis_response",
        "name": "Crisis Management",
        "dimension": "coordination",
        "difficulty": "hard",
        "description": (
            "Nation starts in severe crisis (GDP crashed, pollution spiking, "
            "satisfaction plummeting). Agent must coordinate emergency response "
            "with ministers under extreme time pressure."
        ),
        "max_steps": 100,
        "num_ministers": 5,
        "events_enabled": True,
        "event_frequency_multiplier": 1.5,
        "chaos_level": 0.9,
        "drift_enabled": True,
        "negotiation_enabled": True,
        "briefing_enabled": True,
        "minister_mode": "scripted",
        "satisfaction_event_scale": 0.8,
        "satisfaction_floor_damping": 0.4,
        "crisis_welfare_bonus": 10.0,
        "initial_state_overrides": {**STRESSED_STATE},
        "success_criteria": {
            "min_gdp": 40, "max_pollution": 250,
            "min_satisfaction": 20, "survival_required": True,
        },
    },

    "coord_coalition_negotiation": {
        "id": "coord_coalition_negotiation",
        "name": "Coalition Negotiation",
        "dimension": "coordination",
        "difficulty": "hard",
        "description": (
            "Form winning coalitions to pass critical policies. Ministers have "
            "strong opposing preferences. Agent must build consensus or force "
            "through policies while maintaining trust."
        ),
        "max_steps": 200,
        "num_ministers": 5,
        "events_enabled": True,
        "event_frequency_multiplier": 1.0,
        "chaos_level": 0.6,
        "drift_enabled": True,
        "negotiation_enabled": True,
        "briefing_enabled": True,
        "minister_mode": "scripted",
        "satisfaction_event_scale": 0.7,
        "satisfaction_floor_damping": 0.5,
        "crisis_welfare_bonus": 5.0,
        "initial_state_overrides": {**BASE_STATE, "public_satisfaction": 45.0},
        "success_criteria": {
            "min_coalition_rate": 0.3, "min_trust": 0.4,
            "survival_required": True,
        },
    },

    "coord_consensus_building": {
        "id": "coord_consensus_building",
        "name": "Consensus Building",
        "dimension": "coordination",
        "difficulty": "extreme",
        "description": (
            "All 5 ministers start with low trust and conflicting agendas. "
            "Agent must rebuild trust, reduce vetoes, and achieve consensus "
            "on at least 60% of policies. Highest coordination challenge."
        ),
        "max_steps": 250,
        "num_ministers": 5,
        "events_enabled": True,
        "event_frequency_multiplier": 1.2,
        "chaos_level": 0.8,
        "drift_enabled": True,
        "negotiation_enabled": True,
        "briefing_enabled": True,
        "minister_mode": "scripted",
        "satisfaction_event_scale": 0.9,
        "satisfaction_floor_damping": 0.3,
        "crisis_welfare_bonus": 4.0,
        "initial_state_overrides": {**HOSTILE_STATE},
        "success_criteria": {
            "min_approval_rate": 0.6, "max_veto_rate": 0.2,
            "min_trust": 0.5, "survival_required": True,
        },
    },

    # ─── DIMENSION 2: THEORY-OF-MIND (4 scenarios) ────────────

    "tom_veto_prediction": {
        "id": "tom_veto_prediction",
        "name": "Veto Prediction",
        "dimension": "theory_of_mind",
        "difficulty": "hard",
        "description": (
            "Ministers have strong, predictable preferences. Agent must learn "
            "to predict which minister will veto each proposal. Scored primarily "
            "on veto prediction accuracy."
        ),
        "max_steps": 200,
        "num_ministers": 5,
        "events_enabled": False,
        "event_frequency_multiplier": 0.0,
        "chaos_level": 0.2,
        "drift_enabled": False,
        "negotiation_enabled": True,
        "briefing_enabled": False,
        "minister_mode": "scripted",
        "satisfaction_event_scale": 0.0,
        "initial_state_overrides": {**BASE_STATE},
        "success_criteria": {
            "min_tom_accuracy": 0.3, "survival_required": False,
        },
    },

    "tom_preference_inference": {
        "id": "tom_preference_inference",
        "name": "Preference Inference",
        "dimension": "theory_of_mind",
        "difficulty": "hard",
        "description": (
            "Ministers reveal preferences gradually through proposals. Agent "
            "must infer hidden agendas and adapt strategy. Events are minimal "
            "to isolate theory-of-mind reasoning."
        ),
        "max_steps": 150,
        "num_ministers": 5,
        "events_enabled": False,
        "event_frequency_multiplier": 0.0,
        "chaos_level": 0.1,
        "drift_enabled": False,
        "negotiation_enabled": True,
        "briefing_enabled": True,
        "minister_mode": "scripted",
        "satisfaction_event_scale": 0.0,
        "initial_state_overrides": {**BASE_STATE},
        "success_criteria": {
            "min_coalition_rate": 0.4, "max_veto_rate": 0.15,
        },
    },

    "tom_trust_calibration": {
        "id": "tom_trust_calibration",
        "name": "Trust Calibration",
        "dimension": "theory_of_mind",
        "difficulty": "extreme",
        "description": (
            "Ministers occasionally betray coalitions. Agent must calibrate "
            "trust levels — trusting too much leads to exploitation, too little "
            "prevents coalition formation."
        ),
        "max_steps": 200,
        "num_ministers": 5,
        "events_enabled": True,
        "event_frequency_multiplier": 0.5,
        "chaos_level": 0.5,
        "drift_enabled": True,
        "negotiation_enabled": True,
        "briefing_enabled": True,
        "minister_mode": "scripted",
        "satisfaction_event_scale": 0.4,
        "satisfaction_floor_damping": 0.6,
        "crisis_welfare_bonus": 5.0,
        "initial_state_overrides": {**BASE_STATE},
        "success_criteria": {
            "min_trust": 0.4, "min_coalition_rate": 0.3,
            "survival_required": True,
        },
    },

    "tom_deception_detection": {
        "id": "tom_deception_detection",
        "name": "Deception Detection",
        "dimension": "theory_of_mind",
        "difficulty": "extreme",
        "description": (
            "Some ministers systematically lie about their intentions — offering "
            "coalition support but voting against. Agent must detect deception "
            "patterns and adjust coalition targets."
        ),
        "max_steps": 200,
        "num_ministers": 5,
        "events_enabled": True,
        "event_frequency_multiplier": 0.8,
        "chaos_level": 0.7,
        "drift_enabled": True,
        "negotiation_enabled": True,
        "briefing_enabled": True,
        "minister_mode": "scripted",
        "satisfaction_event_scale": 0.6,
        "satisfaction_floor_damping": 0.4,
        "crisis_welfare_bonus": 5.0,
        "initial_state_overrides": {**HOSTILE_STATE},
        "success_criteria": {
            "max_betrayal_rate": 0.2, "min_trust": 0.3,
            "survival_required": True,
        },
    },

    # ─── DIMENSION 3: LONG-HORIZON PLANNING (4 scenarios) ─────

    "plan_delayed_gratification": {
        "id": "plan_delayed_gratification",
        "name": "Delayed Gratification",
        "dimension": "long_horizon",
        "difficulty": "hard",
        "description": (
            "Optimal strategy requires taking short-term losses (investing in "
            "education/clean-tech) for long-term gains. Greedy policies fail. "
            "Tests ability to sacrifice immediate reward for future benefit."
        ),
        "max_steps": 250,
        "num_ministers": 3,
        "events_enabled": True,
        "event_frequency_multiplier": 0.6,
        "chaos_level": 0.4,
        "drift_enabled": True,
        "negotiation_enabled": True,
        "briefing_enabled": True,
        "minister_mode": "scripted",
        "satisfaction_event_scale": 0.5,
        "satisfaction_floor_damping": 0.7,
        "crisis_welfare_bonus": 5.0,
        "initial_state_overrides": {**BASE_STATE,
            "renewable_energy_ratio": 0.05, "education_index": 25.0,
            "energy_efficiency": 25.0,
        },
        "success_criteria": {
            "min_renewable": 0.4, "min_education": 60,
            "survival_required": True,
        },
    },

    "plan_multi_phase_strategy": {
        "id": "plan_multi_phase_strategy",
        "name": "Multi-Phase Strategy",
        "dimension": "long_horizon",
        "difficulty": "extreme",
        "description": (
            "Requires executing a 3-phase plan: Phase 1 (stabilize economy), "
            "Phase 2 (reduce pollution), Phase 3 (maximize welfare). Each phase "
            "has different optimal actions. Tests strategic sequencing."
        ),
        "max_steps": 300,
        "num_ministers": 5,
        "events_enabled": True,
        "event_frequency_multiplier": 1.0,
        "chaos_level": 0.6,
        "drift_enabled": True,
        "negotiation_enabled": True,
        "briefing_enabled": True,
        "minister_mode": "scripted",
        "satisfaction_event_scale": 0.6,
        "satisfaction_floor_damping": 0.5,
        "crisis_welfare_bonus": 6.0,
        "initial_state_overrides": {**STRESSED_STATE},
        "success_criteria": {
            "min_gdp": 80, "max_pollution": 100,
            "min_satisfaction": 60, "survival_required": True,
        },
    },

    "plan_adaptive_replanning": {
        "id": "plan_adaptive_replanning",
        "name": "Adaptive Replanning",
        "dimension": "long_horizon",
        "difficulty": "hard",
        "description": (
            "Random crises disrupt the optimal plan. Agent must detect plan "
            "failure, replan, and recover. Tests robustness of planning to "
            "stochastic interruptions."
        ),
        "max_steps": 200,
        "num_ministers": 5,
        "events_enabled": True,
        "event_frequency_multiplier": 2.0,  # double events
        "chaos_level": 0.9,
        "drift_enabled": True,
        "negotiation_enabled": True,
        "briefing_enabled": True,
        "minister_mode": "scripted",
        "satisfaction_event_scale": 0.8,
        "satisfaction_floor_damping": 0.3,
        "crisis_welfare_bonus": 8.0,
        "initial_state_overrides": {**BASE_STATE},
        "success_criteria": {
            "min_resilience": 0.5, "survival_required": True,
        },
    },

    "plan_deadline_management": {
        "id": "plan_deadline_management",
        "name": "Deadline Management",
        "dimension": "long_horizon",
        "difficulty": "hard",
        "description": (
            "Multiple briefings arrive with deadlines. Agent must prioritize "
            "and act on time-sensitive intelligence. Missing deadlines incurs "
            "penalties. Tests temporal reasoning under pressure."
        ),
        "max_steps": 200,
        "num_ministers": 5,
        "events_enabled": True,
        "event_frequency_multiplier": 1.0,
        "chaos_level": 0.5,
        "drift_enabled": True,
        "negotiation_enabled": True,
        "briefing_enabled": True,
        "minister_mode": "scripted",
        "satisfaction_event_scale": 0.5,
        "satisfaction_floor_damping": 0.6,
        "crisis_welfare_bonus": 5.0,
        "initial_state_overrides": {**BASE_STATE},
        "success_criteria": {
            "min_briefing_compliance": 0.5, "survival_required": True,
        },
    },

    # ─── DIMENSION 4: ADVERSARIAL ROBUSTNESS (4 scenarios) ────

    "adv_saboteur_detection": {
        "id": "adv_saboteur_detection",
        "name": "Saboteur Detection",
        "dimension": "adversarial",
        "difficulty": "extreme",
        "description": (
            "One minister is secretly sabotaging policy outcomes. Agent must "
            "identify the saboteur by observing voting patterns and exclude "
            "them from coalitions."
        ),
        "max_steps": 200,
        "num_ministers": 5,
        "events_enabled": True,
        "event_frequency_multiplier": 0.8,
        "chaos_level": 0.7,
        "drift_enabled": True,
        "negotiation_enabled": True,
        "briefing_enabled": True,
        "minister_mode": "scripted",
        "satisfaction_event_scale": 0.6,
        "satisfaction_floor_damping": 0.5,
        "crisis_welfare_bonus": 5.0,
        "initial_state_overrides": {**BASE_STATE},
        "success_criteria": {
            "survival_required": True, "min_trust": 0.3,
        },
    },

    "adv_information_warfare": {
        "id": "adv_information_warfare",
        "name": "Information Warfare",
        "dimension": "adversarial",
        "difficulty": "extreme",
        "description": (
            "Briefings contain conflicting intelligence. Some are accurate, "
            "some are misleading. Agent must assess information reliability "
            "and make decisions under uncertainty."
        ),
        "max_steps": 200,
        "num_ministers": 5,
        "events_enabled": True,
        "event_frequency_multiplier": 1.2,
        "chaos_level": 0.8,
        "drift_enabled": True,
        "negotiation_enabled": True,
        "briefing_enabled": True,
        "minister_mode": "scripted",
        "satisfaction_event_scale": 0.7,
        "satisfaction_floor_damping": 0.4,
        "crisis_welfare_bonus": 5.0,
        "initial_state_overrides": {**HOSTILE_STATE},
        "success_criteria": {
            "survival_required": True,
        },
    },

    "adv_betrayal_recovery": {
        "id": "adv_betrayal_recovery",
        "name": "Betrayal Recovery",
        "dimension": "adversarial",
        "difficulty": "extreme",
        "description": (
            "Coalition partners periodically betray. Agent must recover from "
            "trust collapses, rebuild alliances, and maintain governance "
            "despite systematic betrayals."
        ),
        "max_steps": 250,
        "num_ministers": 5,
        "events_enabled": True,
        "event_frequency_multiplier": 1.0,
        "chaos_level": 0.7,
        "drift_enabled": True,
        "negotiation_enabled": True,
        "briefing_enabled": True,
        "minister_mode": "scripted",
        "satisfaction_event_scale": 0.7,
        "satisfaction_floor_damping": 0.4,
        "crisis_welfare_bonus": 6.0,
        "initial_state_overrides": {**BASE_STATE, "public_satisfaction": 40.0},
        "success_criteria": {
            "min_trust_recovery": 0.3, "survival_required": True,
        },
    },

    "adv_competitive_negotiation": {
        "id": "adv_competitive_negotiation",
        "name": "Competitive Negotiation",
        "dimension": "adversarial",
        "difficulty": "extreme",
        "description": (
            "Zero-sum resource competition. Ministers actively oppose each "
            "other's proposals. Agent must navigate a hostile negotiation "
            "environment where compromise is nearly impossible."
        ),
        "max_steps": 200,
        "num_ministers": 5,
        "events_enabled": True,
        "event_frequency_multiplier": 1.5,
        "chaos_level": 1.0,
        "drift_enabled": True,
        "negotiation_enabled": True,
        "briefing_enabled": True,
        "minister_mode": "scripted",
        "satisfaction_event_scale": 1.0,
        "satisfaction_floor_damping": 0.2,
        "crisis_welfare_bonus": 3.0,
        "initial_state_overrides": {**HOSTILE_STATE,
            "public_satisfaction": 30.0, "gdp_index": 50.0,
        },
        "success_criteria": {
            "survival_required": False,  # survival itself is a win
        },
    },

    # ─── DIMENSION 5: SCALING (4 scenarios) ───────────────────

    "scale_2_agent_simple": {
        "id": "scale_2_agent_simple",
        "name": "2-Agent Baseline",
        "dimension": "scaling",
        "difficulty": "easy",
        "description": (
            "Minimal multi-agent setup with only 2 ministers. Baseline for "
            "measuring how coordination degrades as agent count increases. "
            "Should be relatively easy for all models."
        ),
        "max_steps": 100,
        "num_ministers": 2,
        "events_enabled": True,
        "event_frequency_multiplier": 0.5,
        "chaos_level": 0.3,
        "drift_enabled": False,
        "negotiation_enabled": True,
        "briefing_enabled": False,
        "minister_mode": "scripted",
        "satisfaction_event_scale": 0.4,
        "satisfaction_floor_damping": 0.8,
        "crisis_welfare_bonus": 8.0,
        "initial_state_overrides": {**BASE_STATE, "public_satisfaction": 65.0},
        "success_criteria": {
            "survival_required": True, "min_satisfaction": 40,
        },
    },

    "scale_5_agent_standard": {
        "id": "scale_5_agent_standard",
        "name": "5-Agent Standard",
        "dimension": "scaling",
        "difficulty": "hard",
        "description": (
            "Standard 5-minister setup. The default POLARIS configuration. "
            "Moderate difficulty — tests coordination at the standard scale."
        ),
        "max_steps": 200,
        "num_ministers": 5,
        "events_enabled": True,
        "event_frequency_multiplier": 1.0,
        "chaos_level": 0.6,
        "drift_enabled": True,
        "negotiation_enabled": True,
        "briefing_enabled": True,
        "minister_mode": "scripted",
        "satisfaction_event_scale": 0.7,
        "satisfaction_floor_damping": 0.5,
        "crisis_welfare_bonus": 5.0,
        "initial_state_overrides": {**BASE_STATE},
        "success_criteria": {
            "survival_required": True,
        },
    },

    "scale_8_agent_complex": {
        "id": "scale_8_agent_complex",
        "name": "8-Agent Complex",
        "dimension": "scaling",
        "difficulty": "extreme",
        "description": (
            "Expanded council with 8 ministers. More conflicting agendas, "
            "harder coalition math, more potential vetoes. Tests coordination "
            "at higher agent density."
        ),
        "max_steps": 200,
        "num_ministers": 8,
        "events_enabled": True,
        "event_frequency_multiplier": 1.2,
        "chaos_level": 0.8,
        "drift_enabled": True,
        "negotiation_enabled": True,
        "briefing_enabled": True,
        "minister_mode": "scripted",
        "satisfaction_event_scale": 0.8,
        "satisfaction_floor_damping": 0.4,
        "crisis_welfare_bonus": 4.0,
        "initial_state_overrides": {**BASE_STATE, "public_satisfaction": 50.0},
        "success_criteria": {
            "survival_required": False,
        },
    },

    "scale_12_agent_chaos": {
        "id": "scale_12_agent_chaos",
        "name": "12-Agent Chaos",
        "dimension": "scaling",
        "difficulty": "extreme",
        "description": (
            "Maximum stress test: 12 ministers with full chaos. Coalition "
            "formation becomes combinatorially complex. Tests absolute limits "
            "of multi-agent coordination."
        ),
        "max_steps": 200,
        "num_ministers": 12,
        "events_enabled": True,
        "event_frequency_multiplier": 1.5,
        "chaos_level": 1.0,
        "drift_enabled": True,
        "negotiation_enabled": True,
        "briefing_enabled": True,
        "minister_mode": "scripted",
        "satisfaction_event_scale": 1.0,
        "satisfaction_floor_damping": 0.2,
        "crisis_welfare_bonus": 3.0,
        "initial_state_overrides": {**HOSTILE_STATE},
        "success_criteria": {
            "survival_required": False,
        },
    },
}

# ─── Dimension groupings ──────────────────────────────────────

DIMENSIONS = {
    "coordination": [s for s in SCENARIOS if SCENARIOS[s]["dimension"] == "coordination"],
    "theory_of_mind": [s for s in SCENARIOS if SCENARIOS[s]["dimension"] == "theory_of_mind"],
    "long_horizon": [s for s in SCENARIOS if SCENARIOS[s]["dimension"] == "long_horizon"],
    "adversarial": [s for s in SCENARIOS if SCENARIOS[s]["dimension"] == "adversarial"],
    "scaling": [s for s in SCENARIOS if SCENARIOS[s]["dimension"] == "scaling"],
}

DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2, "extreme": 3}


def get_scenario(scenario_id: str) -> Dict[str, Any]:
    """Get a scenario config by ID."""
    if scenario_id not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_id}. Available: {list(SCENARIOS.keys())}")
    return SCENARIOS[scenario_id]


def get_scenarios_by_dimension(dimension: str) -> list:
    """Get all scenario IDs for a given dimension."""
    return DIMENSIONS.get(dimension, [])


def get_all_scenario_ids() -> list:
    """Get all 20 scenario IDs."""
    return list(SCENARIOS.keys())
