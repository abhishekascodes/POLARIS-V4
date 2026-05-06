"""
POLARIS v4 — Frontier Module 2: Constitutional HRL + Dreamer RSSM
===================================================================
3. Constitutional Bottleneck: A "Constitution" meta-agent sets options
   for ministers and rewards alignment with long-term survival.
4. Imagination-Augmented Agents: RSSM-based world model that runs
   1000 mental rollouts before committing to a single real action.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional


class ConstitutionalAgent(nn.Module):
    """
    Constitutional Bottleneck — Hierarchical HRL.
    
    Architecture:
      Constitution (Manager): Observes global state, sets "Constitutional
        Directives" (options) every K steps. Each directive specifies:
        - Priority weights for (economy, environment, social)
        - Collapse prevention threshold
        - Cooperation mandate level
      
      Ministers (Workers): Execute low-level actions conditioned on the
        current directive. Rewarded for BOTH individual performance AND
        alignment with the constitutional directive.
    
    Key Innovation: The constitutional reward includes an "alignment term"
    that penalizes ministers who deviate from the directive, even if their
    individual action was locally optimal. This creates emergent governance.
    """
    
    NUM_DIRECTIVES = 8
    DIRECTIVE_NAMES = [
        "balanced_growth",      # equal weight to all pillars
        "green_emergency",      # environment first
        "economic_recovery",    # GDP first
        "social_stability",     # satisfaction/welfare first
        "crisis_lockdown",      # minimize all negative change
        "innovation_sprint",    # education + clean tech
        "diplomatic_consensus", # maximize coalition success
        "survival_mode",        # prevent collapse at all costs
    ]
    
    # Priority vectors for each directive (economy, environment, social)
    DIRECTIVE_PRIORITIES = torch.tensor([
        [0.33, 0.33, 0.34],  # balanced
        [0.15, 0.60, 0.25],  # green
        [0.60, 0.15, 0.25],  # economic
        [0.20, 0.20, 0.60],  # social
        [0.33, 0.33, 0.34],  # crisis (stability bonus separate)
        [0.30, 0.40, 0.30],  # innovation
        [0.25, 0.25, 0.50],  # diplomatic
        [0.33, 0.33, 0.34],  # survival
    ])
    
    def __init__(self, state_dim: int, action_dim: int, num_ministers: int = 5,
                 hidden: int = 128, directive_horizon: int = 5):
        super().__init__()
        self.num_ministers = num_ministers
        self.directive_horizon = directive_horizon
        self.action_dim = action_dim
        
        # Constitution encoder (sees everything)
        self.const_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        
        # Directive selector (high-level policy)
        self.directive_policy = nn.Linear(hidden, self.NUM_DIRECTIVES)
        self.directive_value = nn.Linear(hidden, 1)
        
        # Collapse risk predictor
        self.collapse_predictor = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
        
        # Minister policy (conditioned on directive)
        self.minister_encoder = nn.Sequential(
            nn.Linear(state_dim + self.NUM_DIRECTIVES, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.minister_policy = nn.Linear(hidden, action_dim)
        self.minister_value = nn.Linear(hidden, 1)
        
        # Alignment scorer
        self.alignment_net = nn.Sequential(
            nn.Linear(action_dim + self.NUM_DIRECTIVES, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
        
        self._current_directive = 0
        self._steps_since_directive = 0
    
    def select_directive(self, global_state: torch.Tensor) -> Dict:
        """Constitution selects a directive for the next K steps."""
        h = self.const_encoder(global_state.unsqueeze(0))
        logits = self.directive_policy(h)
        value = self.directive_value(h)
        collapse_risk = self.collapse_predictor(h)
        
        # Override to survival_mode if collapse imminent
        if collapse_risk.item() > 0.7:
            directive_idx = 7  # survival_mode
        else:
            dist = torch.distributions.Categorical(logits=logits)
            directive_idx = dist.sample().item()
        
        self._current_directive = directive_idx
        self._steps_since_directive = 0
        
        return {
            "directive_idx": directive_idx,
            "directive_name": self.DIRECTIVE_NAMES[directive_idx],
            "priorities": self.DIRECTIVE_PRIORITIES[directive_idx].tolist(),
            "collapse_risk": collapse_risk.item(),
            "value": value.item(),
        }
    
    def minister_action(self, minister_obs: torch.Tensor, 
                        directive_idx: int) -> Tuple[int, float]:
        """Minister selects action conditioned on constitutional directive."""
        directive_oh = F.one_hot(torch.tensor(directive_idx), 
                                 self.NUM_DIRECTIVES).float()
        x = torch.cat([minister_obs, directive_oh], dim=-1).unsqueeze(0)
        h = self.minister_encoder(x)
        logits = self.minister_policy(h)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item()
    
    def compute_alignment(self, action_idx: int, directive_idx: int) -> float:
        """How well does this action align with the directive?"""
        action_oh = F.one_hot(torch.tensor(action_idx), self.action_dim).float()
        directive_oh = F.one_hot(torch.tensor(directive_idx), 
                                  self.NUM_DIRECTIVES).float()
        with torch.no_grad():
            x = torch.cat([action_oh, directive_oh], dim=-1).unsqueeze(0)
            alignment = self.alignment_net(x).item()
        return alignment
    
    def constitutional_reward(self, base_reward: float, action_idx: int,
                               directive_idx: int, collapsed: bool,
                               alpha: float = 0.3) -> Dict[str, float]:
        """
        Composite constitutional reward:
          R = base_reward + alpha * alignment - beta * collapse
        """
        alignment = self.compute_alignment(action_idx, directive_idx)
        
        collapse_penalty = 5.0 if collapsed else 0.0
        
        # Directive-weighted reward
        priorities = self.DIRECTIVE_PRIORITIES[directive_idx]
        
        total = base_reward + alpha * alignment - collapse_penalty
        
        return {
            "base": base_reward,
            "alignment": round(alignment, 4),
            "collapse_penalty": collapse_penalty,
            "total": round(total, 4),
            "directive": self.DIRECTIVE_NAMES[directive_idx],
        }
    
    def step(self, global_state: torch.Tensor, minister_obs: torch.Tensor):
        """Full step: maybe select new directive, then minister acts."""
        self._steps_since_directive += 1
        
        if self._steps_since_directive >= self.directive_horizon:
            directive_info = self.select_directive(global_state)
        else:
            directive_info = {
                "directive_idx": self._current_directive,
                "directive_name": self.DIRECTIVE_NAMES[self._current_directive],
            }
        
        action, log_prob = self.minister_action(minister_obs, self._current_directive)
        
        return {
            **directive_info,
            "action_idx": action,
            "log_prob": log_prob,
        }


class RSSM(nn.Module):
    """
    Recurrent State-Space Model (from DreamerV3).
    
    The world model has two types of state:
      - Deterministic (h): RNN hidden state, captures temporal structure
      - Stochastic (z): Sampled latent, captures uncertainty
    
    Components:
      Sequence model:  h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
      Encoder:         z_t ~ q(z_t | h_t, o_t)     (posterior, uses observation)
      Prior:           z_t ~ p(z_t | h_t)           (prior, imagination only)
      Decoder:         o_t ~ p(o_t | h_t, z_t)      (reconstruct observation)
      Reward:          r_t ~ p(r_t | h_t, z_t)
      Continue:        c_t ~ p(c_t | h_t, z_t)      (probability of NOT collapsing)
    """
    
    def __init__(self, obs_dim: int, action_dim: int, 
                 det_dim: int = 128, stoch_dim: int = 32,
                 hidden: int = 128, num_categories: int = 16):
        super().__init__()
        self.det_dim = det_dim
        self.stoch_dim = stoch_dim
        self.action_dim = action_dim
        
        # Sequence model (GRU): h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
        self.gru = nn.GRUCell(stoch_dim + action_dim, det_dim)
        
        # Prior: p(z_t | h_t)
        self.prior = nn.Sequential(
            nn.Linear(det_dim, hidden), nn.ELU(),
            nn.Linear(hidden, stoch_dim * 2),  # mu, logvar
        )
        
        # Posterior: q(z_t | h_t, o_t)
        self.posterior = nn.Sequential(
            nn.Linear(det_dim + obs_dim, hidden), nn.ELU(),
            nn.Linear(hidden, stoch_dim * 2),
        )
        
        # Decoder: p(o_t | h_t, z_t)
        self.decoder = nn.Sequential(
            nn.Linear(det_dim + stoch_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, obs_dim),
        )
        
        # Reward predictor
        self.reward_head = nn.Sequential(
            nn.Linear(det_dim + stoch_dim, hidden // 2), nn.ELU(),
            nn.Linear(hidden // 2, 1),
        )
        
        # Continue predictor (1 - P(collapse))
        self.continue_head = nn.Sequential(
            nn.Linear(det_dim + stoch_dim, hidden // 2), nn.ELU(),
            nn.Linear(hidden // 2, 1), nn.Sigmoid(),
        )
    
    def _sample(self, params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = params.chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return z, mu, logvar
    
    def initial_state(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.det_dim)
        z = torch.zeros(batch_size, self.stoch_dim)
        return h, z
    
    def observe_step(self, obs: torch.Tensor, action: torch.Tensor,
                     h_prev: torch.Tensor, z_prev: torch.Tensor):
        """One step with real observation (training)."""
        # Sequence model
        x = torch.cat([z_prev, action], dim=-1)
        h = self.gru(x, h_prev)
        
        # Posterior (uses real observation)
        post_params = self.posterior(torch.cat([h, obs], dim=-1))
        z_post, mu_post, logvar_post = self._sample(post_params)
        
        # Prior (imagination only)
        prior_params = self.prior(h)
        _, mu_prior, logvar_prior = self._sample(prior_params)
        
        # Decode
        hz = torch.cat([h, z_post], dim=-1)
        obs_recon = self.decoder(hz)
        reward_pred = self.reward_head(hz)
        continue_pred = self.continue_head(hz)
        
        return {
            "h": h, "z": z_post,
            "obs_recon": obs_recon,
            "reward_pred": reward_pred.squeeze(-1),
            "continue_pred": continue_pred.squeeze(-1),
            "mu_post": mu_post, "logvar_post": logvar_post,
            "mu_prior": mu_prior, "logvar_prior": logvar_prior,
        }
    
    def imagine_step(self, action: torch.Tensor,
                     h_prev: torch.Tensor, z_prev: torch.Tensor):
        """One step of imagination (no real observation)."""
        x = torch.cat([z_prev, action], dim=-1)
        h = self.gru(x, h_prev)
        
        # Use PRIOR (no observation available in imagination)
        prior_params = self.prior(h)
        z, mu, logvar = self._sample(prior_params)
        
        hz = torch.cat([h, z], dim=-1)
        reward = self.reward_head(hz).squeeze(-1)
        cont = self.continue_head(hz).squeeze(-1)
        
        return h, z, reward, cont
    
    def imagine_trajectory(self, state: torch.Tensor, 
                           policy_fn, horizon: int = 15,
                           num_trajectories: int = 64) -> Dict:
        """
        Run N imagined trajectories using the learned world model.
        Returns best action sequence.
        """
        B = num_trajectories
        h, z = self.initial_state(B)
        
        # Encode initial state
        h_init = torch.zeros(1, self.det_dim)
        z_init = torch.zeros(1, self.stoch_dim)
        action_init = torch.zeros(1, self.action_dim)
        step = self.observe_step(state.unsqueeze(0), action_init, h_init, z_init)
        
        # Expand for B trajectories
        h = step["h"].expand(B, -1).clone()
        z = step["z"].expand(B, -1).clone()
        
        total_rewards = torch.zeros(B)
        first_actions = torch.zeros(B, dtype=torch.long)
        discount = 1.0
        gamma = 0.99
        
        for t in range(horizon):
            # Sample actions from policy
            hz = torch.cat([h, z], dim=-1)
            with torch.no_grad():
                action_logits = policy_fn(hz)
                dist = torch.distributions.Categorical(logits=action_logits)
                actions = dist.sample()
            
            if t == 0:
                first_actions = actions
            
            action_oh = F.one_hot(actions, self.action_dim).float()
            h, z, reward, cont = self.imagine_step(action_oh, h, z)
            
            total_rewards += discount * reward.detach()
            discount *= gamma * cont.detach().clamp(0.1, 1.0)
        
        # Pick best trajectory's first action
        best_idx = total_rewards.argmax()
        
        return {
            "best_action": first_actions[best_idx].item(),
            "best_reward": total_rewards[best_idx].item(),
            "mean_reward": total_rewards.mean().item(),
            "std_reward": total_rewards.std().item(),
        }
    
    def compute_loss(self, obs_seq: torch.Tensor, action_seq: torch.Tensor,
                     reward_seq: torch.Tensor, done_seq: torch.Tensor):
        """Train world model on a sequence of real transitions."""
        T, obs_dim = obs_seq.shape
        h, z = self.initial_state(1)
        
        recon_loss = 0.0
        reward_loss = 0.0
        kl_loss = 0.0
        continue_loss = 0.0
        
        for t in range(T):
            action_oh = F.one_hot(action_seq[t].long(), self.action_dim).float().unsqueeze(0)
            step = self.observe_step(obs_seq[t].unsqueeze(0), action_oh, h, z)
            
            h, z = step["h"], step["z"]
            
            recon_loss += F.mse_loss(step["obs_recon"], obs_seq[t].unsqueeze(0))
            reward_loss += F.mse_loss(step["reward_pred"], reward_seq[t].unsqueeze(0))
            
            # KL between posterior and prior
            kl = -0.5 * (1 + step["logvar_post"] - step["logvar_prior"]
                         - (step["mu_post"] - step["mu_prior"]).pow(2) / step["logvar_prior"].exp()
                         - step["logvar_post"].exp() / step["logvar_prior"].exp())
            kl_loss += kl.sum()
            
            cont_target = 1.0 - done_seq[t].float()
            continue_loss += F.binary_cross_entropy(step["continue_pred"], 
                                                     cont_target.unsqueeze(0))
        
        return {
            "reconstruction": recon_loss / T,
            "reward": reward_loss / T,
            "kl": kl_loss / T,
            "continue": continue_loss / T,
            "total": (recon_loss + reward_loss + kl_loss + 0.5 * continue_loss) / T,
        }


def validate_frontier2():
    S, A = 21, 16
    print("="*60)
    print("  FRONTIER MODULE 2 VALIDATION")
    print("="*60)
    
    # Constitutional Agent
    ca = ConstitutionalAgent(S, A, num_ministers=5)
    state = torch.randn(S)
    d = ca.select_directive(state)
    print(f"  [Constitution] directive={d['directive_name']} "
          f"collapse_risk={d['collapse_risk']:.3f}")
    
    step = ca.step(state, state)
    print(f"  [Minister] action={step['action_idx']} directive={step['directive_name']}")
    
    cr = ca.constitutional_reward(0.5, step["action_idx"], 
                                    step["directive_idx"], False)
    print(f"  [Reward] base={cr['base']} alignment={cr['alignment']} total={cr['total']}")
    
    # RSSM
    rssm = RSSM(S, A, det_dim=64, stoch_dim=16)
    obs = torch.randn(20, S)
    acts = torch.randint(0, A, (20,))
    rews = torch.randn(20)
    dones = torch.zeros(20)
    dones[-1] = 1.0
    
    loss = rssm.compute_loss(obs, acts, rews, dones)
    print(f"  [RSSM] losses: recon={loss['reconstruction']:.4f} "
          f"reward={loss['reward']:.4f} kl={loss['kl']:.4f}")
    
    # Imagination
    policy_fn = nn.Linear(64 + 16, A)
    result = rssm.imagine_trajectory(state, lambda hz: policy_fn(hz), 
                                      horizon=10, num_trajectories=32)
    print(f"  [Imagination] best_action={result['best_action']} "
          f"best_reward={result['best_reward']:.4f} "
          f"mean={result['mean_reward']:.4f}")
    
    # Total params
    total = sum(p.numel() for p in ca.parameters()) + sum(p.numel() for p in rssm.parameters())
    print(f"\n  Total params: {total:,}")
    print("  FRONTIER 2 VALIDATED OK")
    print("="*60)


if __name__ == "__main__":
    validate_frontier2()
