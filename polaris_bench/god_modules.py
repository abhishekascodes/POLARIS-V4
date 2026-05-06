#!/usr/bin/env python3
"""
POLARIS v4 — GOD-LEVEL RL UPGRADES
====================================
4 cutting-edge RL modules that transform POLARIS from a benchmark into
a research-grade multi-agent coordination laboratory.

Modules:
  1. ICM  — Intrinsic Curiosity Module (exploration bonus)
  2. CommNet — Learned inter-agent communication
  3. HRL  — Hierarchical RL with Options framework
  4. WorldModel — Learned dynamics model for imagination-based planning

Each module is self-contained and composable.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional


# ═══════════════════════════════════════════════════════════════
# 1. INTRINSIC CURIOSITY MODULE (ICM)
#    Paper: "Curiosity-driven Exploration by Self-Supervised Prediction"
#    Why: Agents explore novel states instead of getting stuck
# ═══════════════════════════════════════════════════════════════

class ICM(nn.Module):
    """
    Intrinsic Curiosity Module — generates exploration bonuses
    based on prediction error of state transitions.
    
    r_intrinsic = ||f(s_t, a_t) - phi(s_{t+1})||^2
    
    High error = novel state = explore more.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128, 
                 feature_dim: int = 64, eta: float = 0.01):
        super().__init__()
        self.eta = eta  # intrinsic reward scaling
        
        # Feature encoder: raw state -> learned features
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, feature_dim),
        )
        
        # Forward model: predicts next state features from (features, action)
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, feature_dim),
        )
        
        # Inverse model: predicts action from (state_features, next_state_features)
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )
    
    def forward(self, state: torch.Tensor, next_state: torch.Tensor, 
                action_onehot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encode states
        phi_s = self.encoder(state)
        phi_s_next = self.encoder(next_state)
        
        # Forward prediction (curiosity signal)
        pred_phi_next = self.forward_model(torch.cat([phi_s, action_onehot], dim=-1))
        forward_loss = F.mse_loss(pred_phi_next, phi_s_next.detach(), reduction='none').sum(dim=-1)
        
        # Inverse prediction (self-supervised auxiliary)
        pred_action = self.inverse_model(torch.cat([phi_s, phi_s_next], dim=-1))
        inverse_loss = F.cross_entropy(pred_action, action_onehot.argmax(dim=-1))
        
        # Intrinsic reward = scaled forward prediction error
        intrinsic_reward = self.eta * forward_loss.detach()
        
        return intrinsic_reward, forward_loss.mean(), inverse_loss
    
    def get_bonus(self, state: torch.Tensor, next_state: torch.Tensor, 
                  action_onehot: torch.Tensor) -> float:
        """Get curiosity bonus for a single transition."""
        with torch.no_grad():
            r, _, _ = self.forward(state.unsqueeze(0), next_state.unsqueeze(0), 
                                    action_onehot.unsqueeze(0))
            return r.item()


# ═══════════════════════════════════════════════════════════════
# 2. COMMNET — Learned Multi-Agent Communication
#    Paper: "Learning Multiagent Communication with Backpropagation"
#    Why: Agents learn WHAT to communicate, not just IF
# ═══════════════════════════════════════════════════════════════

class CommChannel(nn.Module):
    """
    Communication channel between agents.
    Each agent broadcasts a message, receives aggregated messages,
    and uses them to condition its policy.
    
    m_i = f(h_i)                    # generate message
    c_i = mean(m_j for j != i)      # aggregate incoming
    h_i' = g(h_i, c_i)             # update hidden state
    """
    def __init__(self, hidden_dim: int, msg_dim: int = 32, num_rounds: int = 2):
        super().__init__()
        self.num_rounds = num_rounds
        self.msg_dim = msg_dim
        
        # Message generation
        self.msg_encoder = nn.Linear(hidden_dim, msg_dim)
        
        # Message integration (update hidden with received messages)
        self.msg_integrator = nn.Sequential(
            nn.Linear(hidden_dim + msg_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Attention over messages (who to listen to)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim + msg_dim, 1),
        )
    
    def forward(self, agent_hiddens: torch.Tensor) -> torch.Tensor:
        """
        agent_hiddens: (num_agents, hidden_dim)
        Returns: updated hiddens after communication rounds
        """
        h = agent_hiddens  # (N, H)
        N = h.shape[0]
        
        for _ in range(self.num_rounds):
            # Generate messages
            messages = self.msg_encoder(h)  # (N, msg_dim)
            
            # For each agent, aggregate messages from others with attention
            updated = []
            for i in range(N):
                # Messages from all OTHER agents
                other_msgs = torch.cat([messages[:i], messages[i+1:]], dim=0)  # (N-1, msg_dim)
                
                if other_msgs.shape[0] == 0:
                    updated.append(h[i])
                    continue
                
                # Attention weights
                h_i_expanded = h[i].unsqueeze(0).expand(other_msgs.shape[0], -1)  # (N-1, H)
                attn_input = torch.cat([h_i_expanded, other_msgs], dim=-1)  # (N-1, H+msg)
                attn_weights = F.softmax(self.attention(attn_input).squeeze(-1), dim=0)  # (N-1,)
                
                # Weighted message aggregation
                aggregated = (attn_weights.unsqueeze(-1) * other_msgs).sum(dim=0)  # (msg_dim,)
                
                # Update hidden
                new_h = self.msg_integrator(torch.cat([h[i], aggregated], dim=-1))
                updated.append(new_h)
            
            h = torch.stack(updated, dim=0)
        
        return h


class CommNetPolicy(nn.Module):
    """
    Multi-agent policy with learned communication.
    President + N ministers each have a policy head,
    but they communicate through CommNet before deciding.
    """
    def __init__(self, state_dim: int, action_dim: int, 
                 hidden: int = 128, msg_dim: int = 32, max_agents: int = 12):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.comm = CommChannel(hidden, msg_dim, num_rounds=2)
        self.policy_head = nn.Linear(hidden, action_dim)
        self.value_head = nn.Linear(hidden, 1)
    
    def forward(self, agent_states: torch.Tensor):
        """
        agent_states: (num_agents, state_dim) — each agent's observation
        Returns: (logits, values) for each agent
        """
        h = self.state_encoder(agent_states)  # (N, hidden)
        h_comm = self.comm(h)  # (N, hidden) — after communication
        logits = self.policy_head(h_comm)  # (N, action_dim)
        values = self.value_head(h_comm).squeeze(-1)  # (N,)
        return logits, values


# ═══════════════════════════════════════════════════════════════
# 3. HIERARCHICAL RL — Options Framework
#    Paper: "The Option-Critic Architecture"
#    Why: Learn WHEN to switch strategies, not just WHAT action
# ═══════════════════════════════════════════════════════════════

class OptionsFramework(nn.Module):
    """
    Hierarchical RL with Options (macro-actions).
    
    High-level policy selects an OPTION (strategy):
      - "economic_focus", "environmental_focus", "social_focus", 
        "crisis_response", "balanced", "aggressive_growth"
    
    Low-level policy selects primitive ACTION conditioned on option.
    Termination function decides when to switch options.
    
    This decomposes the problem: learn STRATEGY selection separately
    from ACTION selection within a strategy.
    """
    
    NUM_OPTIONS = 6
    OPTION_NAMES = [
        "economic_growth",     # prioritize GDP
        "green_transition",    # prioritize environment
        "social_welfare",      # prioritize satisfaction
        "crisis_response",     # emergency mode
        "balanced_governance", # spread actions
        "diplomatic_play",     # focus on coalitions/negotiation
    ]
    
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        
        # High-level: option selection policy (over options)
        self.option_policy = nn.Linear(hidden, self.NUM_OPTIONS)
        
        # Low-level: per-option action policies
        self.action_policies = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden // 2), nn.ReLU(),
                nn.Linear(hidden // 2, action_dim),
            ) for _ in range(self.NUM_OPTIONS)
        ])
        
        # Termination functions: probability of ending current option
        self.terminations = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, 32), nn.ReLU(),
                nn.Linear(32, 1), nn.Sigmoid(),
            ) for _ in range(self.NUM_OPTIONS)
        ])
        
        # Value heads
        self.option_value = nn.Linear(hidden, self.NUM_OPTIONS)  # Q(s, omega)
        self.state_value = nn.Linear(hidden, 1)  # V(s)
        
        # Current option
        self.current_option = 0
    
    def forward(self, state: torch.Tensor):
        h = self.encoder(state)
        return h
    
    def select_option(self, state: torch.Tensor) -> int:
        """High-level: pick which strategy to use."""
        h = self.encoder(state.unsqueeze(0))
        logits = self.option_policy(h)
        dist = torch.distributions.Categorical(logits=logits)
        option = dist.sample().item()
        self.current_option = option
        return option
    
    def select_action(self, state: torch.Tensor, option: int) -> Tuple[int, float]:
        """Low-level: pick action within current option."""
        h = self.encoder(state.unsqueeze(0))
        logits = self.action_policies[option](h)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item()
    
    def should_terminate(self, state: torch.Tensor, option: int) -> bool:
        """Should we switch to a different option?"""
        h = self.encoder(state.unsqueeze(0))
        term_prob = self.terminations[option](h).item()
        return torch.rand(1).item() < term_prob
    
    def get_option_values(self, state: torch.Tensor) -> torch.Tensor:
        h = self.encoder(state.unsqueeze(0))
        return self.option_value(h).squeeze(0)


# ═══════════════════════════════════════════════════════════════
# 4. WORLD MODEL — Learned Dynamics for Imagination
#    Inspired by: DreamerV3, MuZero
#    Why: Plan ahead without expensive real rollouts
# ═══════════════════════════════════════════════════════════════

class WorldModel(nn.Module):
    """
    Learned world model for imagination-based planning.
    
    Components:
      - Encoder: state -> latent z
      - Dynamics: (z, a) -> z'  (predict next latent)
      - Reward:   z -> r        (predict reward)
      - Decoder:  z -> state    (reconstruct for interpretability)
      - Collapse: z -> p(collapse) (predict collapse probability)
    
    Use case: Given current state, imagine K future trajectories
    and pick the action sequence with highest expected reward.
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 latent_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
        # Encoder: state -> latent
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )
        
        # Dynamics model: (latent, action) -> next_latent
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )
        
        # Reward predictor: latent -> reward
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        
        # Collapse predictor: latent -> P(collapse)
        self.collapse_head = nn.Sequential(
            nn.Linear(latent_dim, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1), nn.Sigmoid(),
        )
        
        # State decoder: latent -> reconstructed state
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, state_dim),
        )
    
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.encoder(state)
    
    def imagine_step(self, z: torch.Tensor, action_onehot: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        """One step of imagination: predict next latent, reward, collapse prob."""
        z_next = self.dynamics(torch.cat([z, action_onehot], dim=-1))
        reward = self.reward_head(z_next).squeeze(-1)
        collapse_prob = self.collapse_head(z_next).squeeze(-1)
        return z_next, reward, collapse_prob
    
    def imagine_trajectory(self, state: torch.Tensor, action_sequence: List[int], 
                           gamma: float = 0.99) -> Dict[str, float]:
        """
        Imagine a full trajectory from current state using learned dynamics.
        Returns total imagined reward and collapse probability.
        """
        z = self.encode(state.unsqueeze(0))
        total_reward = 0.0
        max_collapse_prob = 0.0
        discount = 1.0
        
        for action_idx in action_sequence:
            action_oh = F.one_hot(torch.tensor(action_idx), self.action_dim).float().unsqueeze(0)
            z, reward, collapse_prob = self.imagine_step(z, action_oh)
            total_reward += discount * reward.item()
            max_collapse_prob = max(max_collapse_prob, collapse_prob.item())
            discount *= gamma
        
        return {
            "imagined_reward": total_reward,
            "max_collapse_prob": max_collapse_prob,
            "horizon": len(action_sequence),
        }
    
    def plan(self, state: torch.Tensor, num_candidates: int = 64, 
             horizon: int = 5, num_actions: int = 16) -> int:
        """
        Model Predictive Control (MPC) planning.
        Generate random action sequences, imagine outcomes, pick best.
        """
        best_reward = float('-inf')
        best_first_action = 0
        
        with torch.no_grad():
            for _ in range(num_candidates):
                actions = [torch.randint(0, num_actions, (1,)).item() for _ in range(horizon)]
                result = self.imagine_trajectory(state, actions)
                
                # Penalize trajectories that lead to collapse
                adjusted_reward = result["imagined_reward"] - 10.0 * result["max_collapse_prob"]
                
                if adjusted_reward > best_reward:
                    best_reward = adjusted_reward
                    best_first_action = actions[0]
        
        return best_first_action
    
    def compute_loss(self, states: torch.Tensor, actions: torch.Tensor, 
                     next_states: torch.Tensor, rewards: torch.Tensor,
                     collapsed: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Train the world model on real transitions."""
        z = self.encode(states)
        action_oh = F.one_hot(actions, self.action_dim).float()
        
        # Dynamics loss
        z_next_pred = self.dynamics(torch.cat([z, action_oh], dim=-1))
        z_next_true = self.encode(next_states).detach()
        dynamics_loss = F.mse_loss(z_next_pred, z_next_true)
        
        # Reward loss
        reward_pred = self.reward_head(z_next_pred).squeeze(-1)
        reward_loss = F.mse_loss(reward_pred, rewards)
        
        # Collapse prediction loss
        collapse_pred = self.collapse_head(z_next_pred).squeeze(-1)
        collapse_loss = F.binary_cross_entropy(collapse_pred, collapsed.float())
        
        # Reconstruction loss (decoder)
        state_recon = self.decoder(z)
        recon_loss = F.mse_loss(state_recon, states)
        
        return {
            "dynamics": dynamics_loss,
            "reward": reward_loss,
            "collapse": collapse_loss,
            "reconstruction": recon_loss,
            "total": dynamics_loss + reward_loss + collapse_loss + 0.1 * recon_loss,
        }


# ═══════════════════════════════════════════════════════════════
# COMPOSABLE AGENT — Combines all modules
# ═══════════════════════════════════════════════════════════════

class PolarisGodAgent(nn.Module):
    """
    The ultimate POLARIS agent — combines all 4 modules:
      - ICM for exploration
      - CommNet for multi-agent communication
      - Options for hierarchical decision-making
      - WorldModel for planning
    """
    
    def __init__(self, state_dim: int = 21, action_dim: int = 16, hidden: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.icm = ICM(state_dim, action_dim, hidden)
        self.comm = CommNetPolicy(state_dim, action_dim, hidden)
        self.options = OptionsFramework(state_dim, action_dim, hidden)
        self.world_model = WorldModel(state_dim, action_dim)
        
        self._current_option = 0
        self._option_steps = 0
    
    def select_action(self, state: torch.Tensor, 
                      agent_states: Optional[torch.Tensor] = None,
                      use_planning: bool = True) -> Dict:
        """
        Full decision pipeline:
          1. Check if current option should terminate
          2. If yes, select new option (high-level)
          3. If multi-agent, communicate
          4. Select action (low-level, conditioned on option)
          5. Optionally use world model for planning
        """
        # Step 1-2: Hierarchical option management
        if self._option_steps == 0 or self.options.should_terminate(state, self._current_option):
            self._current_option = self.options.select_option(state)
            self._option_steps = 0
        self._option_steps += 1
        
        # Step 3: Communication (if multi-agent)
        comm_info = None
        if agent_states is not None and agent_states.shape[0] > 1:
            logits, values = self.comm(agent_states)
            comm_info = {"logits": logits, "values": values}
        
        # Step 4: Action selection
        action_idx, log_prob = self.options.select_action(state, self._current_option)
        
        # Step 5: World model planning (override if planning finds better)
        if use_planning:
            planned_action = self.world_model.plan(state, num_candidates=32, horizon=5,
                                                    num_actions=self.action_dim)
            # Use planned action if world model is confident
            plan_result = self.world_model.imagine_trajectory(state, [planned_action])
            if plan_result["max_collapse_prob"] < 0.3:
                action_idx = planned_action
        
        return {
            "action_idx": action_idx,
            "log_prob": log_prob,
            "option": self._current_option,
            "option_name": OptionsFramework.OPTION_NAMES[self._current_option],
            "comm_info": comm_info,
        }
    
    def get_param_count(self) -> Dict[str, int]:
        """Parameter count breakdown."""
        return {
            "icm": sum(p.numel() for p in self.icm.parameters()),
            "comm": sum(p.numel() for p in self.comm.parameters()),
            "options": sum(p.numel() for p in self.options.parameters()),
            "world_model": sum(p.numel() for p in self.world_model.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }


# ═══════════════════════════════════════════════════════════════
# QUICK VALIDATION
# ═══════════════════════════════════════════════════════════════

def validate():
    """Quick sanity check that all modules work."""
    S, A = 21, 16
    print("="*60)
    print("  POLARIS v4 — GOD-LEVEL MODULE VALIDATION")
    print("="*60)
    
    # ICM
    icm = ICM(S, A)
    s = torch.randn(4, S)
    s_next = torch.randn(4, S)
    a = F.one_hot(torch.randint(0, A, (4,)), A).float()
    r_intr, fl, il = icm(s, s_next, a)
    print(f"  [ICM] intrinsic_reward={r_intr.mean():.4f} fwd_loss={fl:.4f} inv_loss={il:.4f}")
    
    # CommNet
    comm = CommNetPolicy(S, A)
    agent_obs = torch.randn(5, S)  # 5 agents
    logits, values = comm(agent_obs)
    print(f"  [CommNet] 5 agents -> logits={logits.shape} values={values.shape}")
    
    # Options
    opts = OptionsFramework(S, A)
    state = torch.randn(S)
    option = opts.select_option(state)
    action, lp = opts.select_action(state, option)
    term = opts.should_terminate(state, option)
    print(f"  [Options] option={option}({opts.OPTION_NAMES[option]}) action={action} terminate={term}")
    
    # World Model
    wm = WorldModel(S, A)
    plan_action = wm.plan(state, num_candidates=16, horizon=3, num_actions=A)
    result = wm.imagine_trajectory(state, [0, 1, 2])
    print(f"  [WorldModel] planned_action={plan_action} imagined_reward={result['imagined_reward']:.4f} "
          f"collapse_prob={result['max_collapse_prob']:.4f}")
    
    # Full agent
    agent = PolarisGodAgent(S, A)
    decision = agent.select_action(state, agent_obs)
    params = agent.get_param_count()
    print(f"\n  [PolarisGodAgent] action={decision['action_idx']} "
          f"option={decision['option_name']}")
    print(f"  Parameters: {params}")
    print(f"  Total: {params['total']:,} params")
    print("="*60)
    print("  ALL MODULES VALIDATED")
    print("="*60)


if __name__ == "__main__":
    validate()
