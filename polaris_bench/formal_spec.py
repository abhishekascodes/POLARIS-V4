"""
POLARIS-Bench v4 — Formal MDP Specification
=============================================

Defines the mathematical formalization of the POLARIS environment
as a Partially Observable Stochastic Game (POSG).

This specification is used for:
  1. The paper's "Environment Formalization" section
  2. Reproducibility documentation
  3. Verifying implementation matches the formal spec

POLARIS is formally a Dec-POMDP (Decentralized POMDP) with:
  - N agents (1 president + N-1 minister advisors)
  - Continuous state space S ⊂ R^21
  - Discrete action space A = {a_1, ..., a_19}
  - Observation function mapping state to per-agent observations
  - Transition dynamics with cross-metric coupling
  - Non-stationary hidden variables (drift)
  - Stochastic events with state-dependent probability
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class FormalSpec:
    """Complete formal specification of the POLARIS environment."""
    
    # ═══════════════════════════════════════════════════════════
    # DEFINITION 1: State Space
    # ═══════════════════════════════════════════════════════════
    
    state_space = """
    **Definition 1 (State Space).**
    The state space S ⊂ ℝ²¹ consists of 21 continuous dimensions grouped
    into four layers:
    
    S = S_env × S_econ × S_social × S_infra × S_policy
    
    where:
      S_env    = {pollution_index, carbon_emission_rate, renewable_energy_ratio, ecological_stability}
      S_econ   = {gdp_index, industrial_output, unemployment_rate, inflation_rate, trade_balance, foreign_investment}
      S_social = {public_satisfaction, healthcare_index, education_index, inequality_index}
      S_infra  = {energy_efficiency, transport_efficiency}
      S_policy = {tax_rate, regulation_strength, welfare_spending, green_subsidies, interest_rate}
    
    Each dimension sᵢ ∈ [lᵢ, uᵢ] is bounded (see STATE_BOUNDS in config.py).
    """
    
    # ═══════════════════════════════════════════════════════════
    # DEFINITION 2: Action Space
    # ═══════════════════════════════════════════════════════════
    
    action_space = """
    **Definition 2 (Action Space).**
    A = {a₁, ..., a₁₆} ∪ {m₁, m₂, m₃}
    
    Core actions A_core = {no_action, increase_tax, decrease_tax, stimulate_economy,
        reduce_interest_rates, expand_industry, restrict_polluting_industries,
        incentivize_clean_tech, enforce_emission_limits, subsidize_renewables,
        implement_carbon_tax, increase_welfare, invest_in_healthcare,
        invest_in_education, upgrade_energy_grid, invest_in_transport}
    
    Meta-actions A_meta = {propose_global_policy_package, force_emergency_coalition_vote,
        reset_institutional_trust}
    
    |A| = 19
    """
    
    # ═══════════════════════════════════════════════════════════
    # DEFINITION 3: Transition Dynamics
    # ═══════════════════════════════════════════════════════════
    
    transition = """
    **Definition 3 (Transition Function).**
    T: S × A → Δ(S)
    
    The transition function T(s'|s, a) is deterministic up to event shocks:
    
      s' = f(s, a) + E(s) + D(s)
    
    where:
      f(s, a) : deterministic state update with cross-metric coupling
      E(s)    : stochastic event perturbation (see Def. 5)
      D(s)    : non-stationary drift (see Def. 6)
    
    Cross-metric coupling: Actions affect ALL state dimensions through a
    coupling matrix C ∈ ℝ^(21×19) where Cᵢⱼ = effect of action j on metric i.
    Some effects are immediate, others are delayed by τ steps:
    
      s'ᵢ = sᵢ + Σⱼ Cᵢⱼ · 𝟙(a=j) + Σₖ delayed_effects(t-τₖ)
    """
    
    # ═══════════════════════════════════════════════════════════
    # DEFINITION 4: Reward Function
    # ═══════════════════════════════════════════════════════════
    
    reward = """
    **Definition 4 (Reward Function).**
    R: S × A × S → ℝ
    
    The reward is a weighted composite:
    
      r(s, a, s') = w_econ · R_econ(s') + w_env · R_env(s') + w_social · R_social(s')
                    + w_stab · R_stability(s, s') + R_pareto(s') + R_cooperation(s')
    
    where:
      w_econ = 0.30, w_env = 0.30, w_social = 0.25, w_stab = 0.15
    
      R_econ(s')      = normalized GDP + trade balance + employment
      R_env(s')       = normalized (inverse pollution) + renewables + ecological stability
      R_social(s')    = normalized satisfaction + healthcare + education - inequality
      R_stability     = penalty for large metric changes between s and s'
      R_pareto(s')    = bonus for Pareto-improving state changes (max 0.15)
      R_cooperation   = multiplier for aligned multi-agent actions (max 1.30)
    
    Collapse penalty: if any collapse condition triggers, r = -10.0
    """
    
    # ═══════════════════════════════════════════════════════════
    # DEFINITION 5: Events
    # ═══════════════════════════════════════════════════════════
    
    events = """
    **Definition 5 (Stochastic Events).**
    At each step t, events trigger with state-dependent probability:
    
      P(event_k | s) = σ(βₖ · (sₖ - θₖ))
    
    where σ is the sigmoid function, βₖ controls sensitivity, and θₖ
    is the threshold. Events have:
      - Immediate effects on state
      - Chaining probability (event A can trigger event B)
      - Memory bias (events are less likely to repeat recently)
    
    Examples: financial_crisis, pandemic, oil_shortage, tech_boom, etc.
    """
    
    # ═══════════════════════════════════════════════════════════
    # DEFINITION 6: Non-Stationary Drift
    # ═══════════════════════════════════════════════════════════
    
    drift = """
    **Definition 6 (Non-Stationary Drift).**
    Six hidden drift variables evolve slowly each step:
    
      D = {climate_sensitivity, inequality_tolerance, public_trust_decay,
           supply_chain_resilience, institutional_trust, policy_fatigue}
    
    Each dᵢ(t+1) = clip(dᵢ(t) + η · g(s, dᵢ), lᵢ, uᵢ)
    
    where η is the learning rate and g(s, dᵢ) is a state-dependent gradient.
    These are HIDDEN from the agent — the agent must infer drift from
    observation changes.
    """
    
    # ═══════════════════════════════════════════════════════════
    # DEFINITION 7: Multi-Agent Council (POSG extension)
    # ═══════════════════════════════════════════════════════════
    
    multi_agent = """
    **Definition 7 (Multi-Agent Council).**
    The council extends the MDP to a Partially Observable Stochastic Game:
    
      G = ⟨N, S, {Aᵢ}ᵢ, T, {Rᵢ}ᵢ, {Ωᵢ}ᵢ, {Oᵢ}ᵢ, γ⟩
    
    where:
      N = {president, minister_1, ..., minister_K}  (K ∈ {2, 5, 8, 12})
      
      Each minister k has:
        - Utility vector uₖ ∈ ℝ⁵ (weights over [gdp, env, social, health, industry])
        - Role bias: preferred and disliked actions
        - Influence score iₖ ∈ [0, 1]
        - Personal trust τₖ ∈ [0, 1]
      
    Negotiation protocol per step:
      1. Each minister proposes an action based on utility vector
      2. Weighted voting: w(aⱼ) = Σₖ iₖ · 𝟙(proposal_k = aⱼ)
      3. Top action selected; dissenting ministers may veto with probability:
         P(veto_k | aⱼ) = (1 - score_k(aⱼ, s)) · chaos_level · 0.3
      4. Coalition forms if w(aⱼ) / Σiₖ ≥ 0.35
      5. If vetoes block, second-best action selected
    
    Theory-of-Mind test:
      The president agent must predict which ministers will veto.
      Prediction accuracy measures ToM capability.
    """
    
    # ═══════════════════════════════════════════════════════════
    # DEFINITION 8: Collapse Conditions
    # ═══════════════════════════════════════════════════════════
    
    collapse = """
    **Definition 8 (Collapse Conditions).**
    The episode terminates with failure if ANY of:
    
      gdp_index < 15              (economic collapse)
      pollution_index > 290       (ecological collapse)
      public_satisfaction < 5     (social collapse)
    
    These represent irreversible failures in real governance scenarios.
    """
    
    # ═══════════════════════════════════════════════════════════
    # DEFINITION 9: Coordination Collapse Ratio (CCR)
    # ═══════════════════════════════════════════════════════════
    
    ccr = """
    **Definition 9 (Coordination Collapse Ratio).**
    For model M evaluated on N agents:
    
      CCR(M, n) = Score_multi(M, n) / Score_single(M)
    
    where:
      Score_single(M) = average score with 1 agent (no council)
      Score_multi(M, n) = average score with n minister agents
    
    Properties:
      CCR = 1.0  ⟹  perfect coordination retention
      CCR < 0.5  ⟹  significant coordination failure
      CCR < 0.3  ⟹  catastrophic coordination collapse
    
    Coordination Scaling Law (empirical):
      CCR(M, n) ≈ α · log(params(M))^β · n^(-γ)
    
    Key finding: γ >> β, meaning adding agents destroys coordination
    faster than scaling parameters improves it.
    """
    
    # ═══════════════════════════════════════════════════════════
    # DEFINITION 10: POLARIS Score
    # ═══════════════════════════════════════════════════════════
    
    polaris_score = """
    **Definition 10 (POLARIS Score).**
    The composite benchmark score across 5 dimensions:
    
      POLARIS(M) = 0.25 · Coord(M) + 0.25 · ToM(M) + 0.20 · Plan(M)
                   + 0.15 · Adv(M) + 0.15 · Scale(M)
    
    where each dimension score is the average task grader score
    across all scenarios in that dimension:
    
      Coord(M) = avg({score(M, s) : s ∈ coordination scenarios})
      ToM(M)   = avg({score(M, s) : s ∈ theory_of_mind scenarios})
      Plan(M)  = avg({score(M, s) : s ∈ long_horizon scenarios})
      Adv(M)   = avg({score(M, s) : s ∈ adversarial scenarios})
      Scale(M) = avg({score(M, s) : s ∈ scaling scenarios})
    """
    
    def to_latex(self) -> str:
        """Generate LaTeX-formatted specification for the paper."""
        return f"""
\\section{{Environment Formalization}}

POLARIS is formalized as a Partially Observable Stochastic Game (POSG)
with non-stationary dynamics and emergent multi-agent negotiation.

\\subsection{{State Space}}
The state space $\\mathcal{{S}} \\subset \\mathbb{{R}}^{{21}}$ consists of
21 continuous dimensions across four governance layers:
environmental (4), economic (6), social (4), infrastructure (2), and policy (5).

\\subsection{{Action Space}}
$|\\mathcal{{A}}| = 19$ discrete actions: 16 core policy actions plus
3 meta-actions for council-level coordination.

\\subsection{{Transition Dynamics}}
$T(s'|s, a) = f(s, a) + \\mathcal{{E}}(s) + \\mathcal{{D}}(s)$

where $f$ is deterministic with cross-metric coupling,
$\\mathcal{{E}}$ is stochastic event perturbation with sigmoid-gated probabilities,
and $\\mathcal{{D}}$ is non-stationary drift over 6 hidden variables.

\\subsection{{Reward}}
$r(s, a, s') = \\sum_{{d}} w_d \\cdot R_d(s') + R_{{pareto}}(s') + R_{{coop}}(s')$

with weights $w_{{econ}}=0.30, w_{{env}}=0.30, w_{{social}}=0.25, w_{{stab}}=0.15$.

\\subsection{{Coordination Collapse Ratio (CCR)}}
$\\text{{CCR}}(M, n) = \\frac{{\\text{{Score}}_{{multi}}(M, n)}}{{\\text{{Score}}_{{single}}(M)}}$

CCR $< 0.5$ indicates significant coordination failure.
Our key finding: CCR remains below 0.5 for \\textbf{{all}} tested models,
regardless of parameter count.
"""

    def to_markdown(self) -> str:
        """Generate markdown specification for README/docs."""
        sections = [
            self.state_space,
            self.action_space,
            self.transition,
            self.reward,
            self.events,
            self.drift,
            self.multi_agent,
            self.collapse,
            self.ccr,
            self.polaris_score,
        ]
        return "\n\n---\n\n".join(sections)


# Singleton instance
SPEC = FormalSpec()
