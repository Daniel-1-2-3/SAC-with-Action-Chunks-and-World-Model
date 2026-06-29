import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import re


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def weight_init(m):
    """
    Standard orthogonal weight initialization.
    Carried over from the original DrQV2 codebase.
    """
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def soft_update_params(net, target_net, tau):
    """
    Soft (Polyak) update of target network parameters.
    target = tau * net + (1 - tau) * target
    Used to slowly update the target critic toward the online critic.
    """
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1.0 - tau) * target_param.data
        )


def schedule(schdl, step):
    """
    Parses a stddev schedule string like 'linear(1.0, 0.1, 100000)'
    and returns the current stddev value at the given step.
    Used to anneal exploration noise over training.
    """
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
    raise NotImplementedError(schdl)


# ---------------------------------------------------------------------------
# TruncatedNormal distribution (from original DrQV2)
# ---------------------------------------------------------------------------

class TruncatedNormal(torch.distributions.Normal):
    """
    Normal distribution with actions clipped to [-1, 1].
    Used by the actor to keep actions in a valid range.
    """

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps   = torch.distributions.utils._standard_normal(
            shape, dtype=self.loc.dtype, device=self.loc.device
        )
        eps  *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return torch.clamp(x, -1, 1)


# ---------------------------------------------------------------------------
# Actor
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    """
    DrQV2 actor. Maps latent state s_t to an action distribution.

    In the original DrQV2, the actor took a CNN embedding of pixels.
    Here it takes the world model's latent state s_t = concat(h_t, z_t)
    directly — same MLP structure, different input.

    The trunk (LayerNorm + Tanh) normalizes the latent state before
    the policy layers, which helps stability.
    """

    def __init__(self, state_dim, action_dim, feature_dim=256, hidden_dim=1024):
        super().__init__()

        # normalizes the latent state before the policy network
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh()
        )

        # maps normalized feature to a mean action vector
        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )

        self.apply(weight_init)

    def forward(self, state, std=0.1):
        """
        Args:
          state: (B, state_dim) — latent state from world model
          std:   exploration noise stddev (annealed during training)

        Returns:
          TruncatedNormal distribution over actions
        """
        h      = self.trunk(state)
        mu     = torch.tanh(self.policy(h))             # mean action in [-1, 1]
        std    = torch.ones_like(mu) * std              # fixed std (annealed externally)
        return TruncatedNormal(mu, std)

    def act(self, state, std):
        """Convenience method: returns a sampled action as a numpy array."""
        dist   = self.forward(state, std)
        action = dist.sample(clip=None)
        return action


# ---------------------------------------------------------------------------
# Critic (two Q-networks for clipped double Q)
# ---------------------------------------------------------------------------

class Critic(nn.Module):
    """
    DrQV2 critic with two Q-networks (clipped double Q).
    Takes latent state s_t and action, outputs Q1 and Q2.

    Having two Q-networks and taking the minimum prevents overestimation
    of Q-values (a common failure mode in off-policy RL).

    Same structure as original DrQV2, just operating on latent states
    instead of CNN pixel embeddings.
    """

    def __init__(self, state_dim, action_dim, feature_dim=256, hidden_dim=1024):
        super().__init__()

        # shared trunk to normalize latent state
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh()
        )

        # Q1: first Q-network
        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        # Q2: second Q-network (same architecture, different weights)
        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.apply(weight_init)

    def forward(self, state, action):
        """
        Args:
          state:  (B, state_dim)
          action: (B, action_dim)

        Returns:
          Q1, Q2: (B, 1) each
        """
        h        = self.trunk(state)
        h_action = torch.cat([h, action], dim=-1)
        q1       = self.Q1(h_action)
        q2       = self.Q2(h_action)
        return q1, q2


# ---------------------------------------------------------------------------
# DrQV2 Agent (operating on world model latent states)
# ---------------------------------------------------------------------------

class DrQV2Agent:
    """
    DrQV2 actor-critic agent, modified to operate on latent states from
    the world model rather than raw pixel observations.

    The world model produces s_t = concat(h_t, z_t) for each timestep.
    The actor and critic both take s_t as input directly — no additional
    image encoder needed since the world model handles that.

    Training follows the standard DrQV2 update:
      1. Critic update: minimize Bellman error using target Q-networks
      2. Actor update: maximize Q-value (policy gradient in Q-space)
      3. Soft update target critic toward online critic

    The key difference from the original DrQV2:
      - No CNN encoder (world model handles obs encoding)
      - No image augmentation (RandomShiftsAug) — not needed for latent states
      - Input is s_t (latent) instead of pixel obs
    """

    def __init__(
        self,
        state_dim,          # world model latent state dimension
        action_dim,         # environment action dimension
        device,
        lr              = 1e-4,
        feature_dim     = 256,
        hidden_dim      = 1024,
        critic_target_tau = 0.01,   # soft update rate for target critic
        stddev_schedule = 0.2,      # exploration noise (fixed float or schedule string)
        stddev_clip     = 0.3,      # clip exploration noise at this value
        discount        = 0.99,     # reward discount factor
        update_every_steps = 2,     # how often to update (every N env steps)
    ):
        self.device              = device
        self.critic_target_tau   = critic_target_tau
        self.stddev_schedule     = stddev_schedule
        self.stddev_clip         = stddev_clip
        self.discount            = discount
        self.update_every_steps  = update_every_steps

        # --- Actor ---
        self.actor     = Actor(state_dim, action_dim, feature_dim, hidden_dim).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # --- Critic (online + target) ---
        # target critic is a slow-moving copy of the online critic
        # used for stable Bellman targets — if we used the online critic
        # for both the prediction and the target, training would be unstable
        self.critic            = Critic(state_dim, action_dim, feature_dim, hidden_dim).to(device)
        self.critic_target     = Critic(state_dim, action_dim, feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())  # start identical
        self.critic_opt        = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def act(self, state, step, eval_mode=False):
        """
        Select an action given a latent state.

        Args:
          state:     (state_dim,) numpy array or tensor — latent state from world model
          step:      current training step (used for stddev schedule)
          eval_mode: if True, use mean action (no noise) for evaluation

        Returns:
          action: (action_dim,) numpy array
        """
        state  = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        stddev = schedule(self.stddev_schedule, step) if isinstance(
            self.stddev_schedule, str) else self.stddev_schedule

        dist   = self.actor(state, stddev)

        if eval_mode:
            action = dist.mean                      # deterministic for eval
        else:
            action = dist.sample(clip=None)         # stochastic for training

        return action.cpu().detach().numpy()[0]

    def update_critic(self, states, actions, rewards, next_states, dones, step):
        """
        Update the critic using the Bellman equation.

        Target Q = r + gamma * min(Q1_target(s', a'), Q2_target(s', a'))

        We use clipped double Q (min of two Q-networks) to prevent
        Q-value overestimation, and the target critic for stable targets.

        states, next_states: latent states from the world model (B, state_dim)
        """
        metrics = {}

        with torch.no_grad():
            stddev     = schedule(self.stddev_schedule, step) if isinstance(
                self.stddev_schedule, str) else self.stddev_schedule

            # actor picks next action from next latent state
            dist       = self.actor(next_states, stddev)
            next_action= dist.sample(clip=self.stddev_clip)

            # target Q-value: take minimum of two target Q-networks
            # to prevent overestimation
            tQ1, tQ2   = self.critic_target(next_states, next_action)
            target_V   = torch.min(tQ1, tQ2)
            target_Q   = rewards + (1.0 - dones) * self.discount * target_V

        # online Q-values
        Q1, Q2      = self.critic(states, actions)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        metrics['critic_loss']     = critic_loss.item()
        metrics['critic_target_q'] = target_Q.mean().item()
        metrics['critic_q1']       = Q1.mean().item()
        metrics['critic_q2']       = Q2.mean().item()
        return metrics

    def update_actor(self, states, step):
        """
        Update the actor by maximizing the Q-value.

        Actor loss = -Q(s, actor(s))
        We want the actor to output actions that the critic thinks are good.

        States are detached — we don't backprop actor gradients into the
        world model's latent states.
        """
        metrics = {}

        stddev  = schedule(self.stddev_schedule, step) if isinstance(
            self.stddev_schedule, str) else self.stddev_schedule

        dist    = self.actor(states, stddev)
        action  = dist.sample(clip=self.stddev_clip)

        # use online critic (not target) for actor update
        Q1, Q2  = self.critic(states, action)
        Q       = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        metrics['actor_loss'] = actor_loss.item()
        metrics['actor_ent']  = dist.entropy().sum(dim=-1).mean().item()
        return metrics

    def update(self, states, actions, rewards, next_states, dones, step):
        """
        Full DrQV2 update step:
          1. Update critic (Bellman error)
          2. Update actor (maximize Q)
          3. Soft update target critic

        All inputs are latent states (B, state_dim) from the world model —
        NOT raw observations. The world model has already handled encoding.
        """
        metrics = {}

        if step % self.update_every_steps != 0:
            return metrics

        metrics.update(
            self.update_critic(states, actions, rewards, next_states, dones, step)
        )

        # actor update uses detached states — we don't want actor gradients
        # flowing back into the world model during the policy update
        metrics.update(
            self.update_actor(states.detach(), step)
        )

        # slowly move target critic toward online critic
        soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

        return metrics