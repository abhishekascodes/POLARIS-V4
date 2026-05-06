"""
POLARIS v4 — Frontier Module 1: Latent Diplomacy + COMA
=========================================================
1. Latent Diplomacy: Differentiable bottleneck communication with
   information-theoretic cost. Solves the "Cheap Talk" problem.
2. COMA: Counterfactual Multi-Agent Policy Gradients with centralized
   critic. Solves the "Credit Assignment" problem.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional


class LatentDiplomacy(nn.Module):
    """
    Differentiable Bottleneck Communication.
    
    Instead of free-form text, agents compress intent into a latent vector,
    pass it through a noise channel (simulating real-world uncertainty),
    and pay a KL cost for communication (information bottleneck).
    
    Key: The KL penalty forces agents to only communicate what MATTERS.
    If beta=0, it's free talk. As beta increases, agents must be selective.
    
    Architecture:
      encode: obs -> mu, logvar (variational encoder)
      noise:  z = mu + sigma * epsilon (reparameterization trick)
      decode: z -> intent_vector (what the receiver understands)
      cost:   KL(q(z|obs) || p(z)) — information cost
    """
    
    def __init__(self, obs_dim: int, latent_dim: int = 16, hidden: int = 64,
                 beta: float = 0.01, noise_scale: float = 0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        self.noise_scale = noise_scale
        
        # Variational encoder: observation -> (mu, logvar)
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden, latent_dim)
        self.logvar_head = nn.Linear(hidden, latent_dim)
        
        # Decoder: received latent -> intent understanding
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, obs_dim),
        )
        
        # Channel capacity tracker
        self._total_kl = 0.0
        self._msg_count = 0
    
    def encode(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(obs)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        # Add channel noise
        z = z + self.noise_scale * torch.randn_like(z)
        return z, mu, logvar
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL(q(z|x) || N(0,I)) — communication cost."""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    
    def communicate(self, sender_obs: torch.Tensor, 
                    receiver_obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full communication round:
          1. Sender encodes observation into latent message
          2. Message passes through noisy channel
          3. Receiver decodes the message
          4. KL cost computed
        """
        z, mu, logvar = self.encode(sender_obs)
        decoded = self.decode(z)
        kl = self.kl_divergence(mu, logvar)
        
        self._total_kl += kl.sum().item()
        self._msg_count += sender_obs.shape[0]
        
        return {
            "message": z,
            "decoded_intent": decoded,
            "kl_cost": kl,
            "info_cost": self.beta * kl.mean(),
            "channel_capacity": mu.abs().mean().item(),
        }
    
    def broadcast(self, agent_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        All agents broadcast simultaneously.
        agent_obs: (N, obs_dim)
        Returns: (messages: (N, latent_dim), total_kl_cost: scalar)
        """
        N = agent_obs.shape[0]
        z, mu, logvar = self.encode(agent_obs)
        kl = self.kl_divergence(mu, logvar)
        
        # Each agent receives mean of OTHER agents' messages
        received = []
        for i in range(N):
            others = torch.cat([z[:i], z[i+1:]], dim=0)
            if others.shape[0] > 0:
                received.append(others.mean(dim=0))
            else:
                received.append(torch.zeros(self.latent_dim, device=z.device))
        
        received = torch.stack(received, dim=0)  # (N, latent_dim)
        return received, self.beta * kl.mean()
    
    @property
    def avg_channel_usage(self) -> float:
        if self._msg_count == 0: return 0.0
        return self._total_kl / self._msg_count


class COMAcritic(nn.Module):
    """
    Counterfactual Multi-Agent Policy Gradient (COMA) Critic.
    
    Centralized critic that sees EVERYTHING (global state + all actions).
    Computes counterfactual baselines: "What would have happened if
    agent i had taken a DIFFERENT action, with everyone else the same?"
    
    This solves Credit Assignment: we know exactly which agent caused
    the economy to crash or the environment to thrive.
    
    Q(s, a) -> expected return given state s and joint action a
    Advantage_i = Q(s, a) - Σ_a'_i π_i(a'_i|o_i) * Q(s, (a'_i, a_{-i}))
    
    The counterfactual baseline marginalizes over agent i's actions
    while keeping everyone else's actions fixed.
    """
    
    def __init__(self, state_dim: int, action_dim: int, max_agents: int = 12,
                 hidden: int = 256):
        super().__init__()
        self.action_dim = action_dim
        self.max_agents = max_agents
        
        # Input: global state + all agent actions (one-hot encoded)
        input_dim = state_dim + max_agents * action_dim
        
        # Centralized Q-network: outputs Q-value for each possible action of each agent
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, max_agents * action_dim),
        )
    
    def forward(self, global_state: torch.Tensor, 
                joint_actions: torch.Tensor, num_agents: int) -> torch.Tensor:
        """
        global_state: (batch, state_dim)
        joint_actions: (batch, max_agents) — action indices
        Returns: Q-values (batch, max_agents, action_dim)
        """
        B = global_state.shape[0]
        # One-hot encode all actions
        actions_oh = F.one_hot(joint_actions.long(), self.action_dim).float()
        actions_flat = actions_oh.view(B, -1)  # (B, max_agents * action_dim)
        
        # Pad if fewer agents than max
        if actions_flat.shape[1] < self.max_agents * self.action_dim:
            pad = torch.zeros(B, self.max_agents * self.action_dim - actions_flat.shape[1],
                            device=global_state.device)
            actions_flat = torch.cat([actions_flat, pad], dim=1)
        
        x = torch.cat([global_state, actions_flat], dim=1)
        q_all = self.critic(x)  # (B, max_agents * action_dim)
        q_all = q_all.view(B, self.max_agents, self.action_dim)
        return q_all[:, :num_agents, :]
    
    def counterfactual_advantage(self, global_state: torch.Tensor,
                                  joint_actions: torch.Tensor,
                                  agent_policies: torch.Tensor,
                                  num_agents: int) -> torch.Tensor:
        """
        Compute COMA counterfactual advantage for each agent.
        
        A_i = Q(s, a) - Σ_{a'_i} π_i(a'_i) * Q(s, (a'_i, a_{-i}))
        
        agent_policies: (batch, num_agents, action_dim) — action probabilities
        Returns: advantages (batch, num_agents)
        """
        B = global_state.shape[0]
        q_values = self.forward(global_state, joint_actions, num_agents)
        
        advantages = []
        for i in range(num_agents):
            # Q-value of actual action taken
            actual_action = joint_actions[:, i].long()
            q_actual = q_values[:, i].gather(1, actual_action.unsqueeze(1)).squeeze(1)
            
            # Counterfactual baseline: expected Q under agent i's policy
            pi_i = agent_policies[:, i]  # (B, action_dim)
            q_i = q_values[:, i]  # (B, action_dim)
            baseline = (pi_i * q_i).sum(dim=1)
            
            advantages.append(q_actual - baseline)
        
        return torch.stack(advantages, dim=1)  # (B, num_agents)


class COMAPolicy(nn.Module):
    """Decentralized actor — each agent only sees its own observation."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 128,
                 comm_dim: int = 16):
        super().__init__()
        # Input: own observation + received communication
        self.net = nn.Sequential(
            nn.Linear(obs_dim + comm_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )
    
    def forward(self, obs: torch.Tensor, comm: torch.Tensor) -> torch.Tensor:
        """Returns action logits."""
        x = torch.cat([obs, comm], dim=-1)
        return self.net(x)
    
    def get_action(self, obs: torch.Tensor, comm: torch.Tensor,
                   deterministic: bool = False):
        logits = self.forward(obs, comm)
        if deterministic:
            return logits.argmax(dim=-1)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()


def validate_frontier1():
    """Quick test."""
    S, A, N = 21, 16, 5
    print("="*60)
    print("  FRONTIER MODULE 1 VALIDATION")
    print("="*60)
    
    # Latent Diplomacy
    ld = LatentDiplomacy(S, latent_dim=16, beta=0.01)
    obs = torch.randn(N, S)
    received, kl_cost = ld.broadcast(obs)
    print(f"  [LatentDiplomacy] received={received.shape} kl_cost={kl_cost:.4f}")
    
    # COMA
    critic = COMAcritic(S, A, max_agents=12)
    policy = COMAPolicy(S, A, comm_dim=16)
    
    gs = torch.randn(2, S)  # batch=2
    ja = torch.randint(0, A, (2, 5))
    q = critic(gs, ja, N)
    print(f"  [COMA Critic] q_values={q.shape}")
    
    # Counterfactual advantage
    pi = F.softmax(torch.randn(2, N, A), dim=-1)
    adv = critic.counterfactual_advantage(gs, ja, pi, N)
    print(f"  [COMA Advantage] shape={adv.shape} mean={adv.mean():.4f}")
    
    comm = torch.randn(S, 16)
    a, lp, ent = policy.get_action(torch.randn(1, S), torch.randn(1, 16))
    print(f"  [COMA Policy] action={a.item()} entropy={ent.item():.4f}")
    
    print("  FRONTIER 1 VALIDATED OK")
    print("="*60)


if __name__ == "__main__":
    validate_frontier1()
