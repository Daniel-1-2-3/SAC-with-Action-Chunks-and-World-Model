""" SEAR agent (Nagy et al. 2026), implemented from the paper.

    Policy: factorized tanh-squashed Gaussian over the whole chunk,
    pi(a_{t:t+N} | s_t), with a learned temperature alpha (MaxEnt).

    Critic update -- the paper's Eqs. 6-8, transcribed:
      G^(k)(s_t, a_{t:t+k}) = sum_{i<k} gamma^i r_{t+i}
                              + gamma^k * Qhat^(N)(s_{t+k}, a'_{1:N}),
      a' ~ pi(.|s_{t+k}),
      Qhat^(N) = min_{j in {1,2}} Q_j^(N) - alpha * sum_i gamma^i log pi(a'_i),
      J(phi) = 1/(2N) sum_k (Q^(k) - G^(k))^2,
    i.e. every chunk prefix is a training target, and every prefix
    bootstraps with the FULL-horizon target Q at that prefix's end state.

    Actor update -- Eq. 9's reverse-KL, which reduces to the SAC form:
      maximize E[ Q^(N)(s, atilde) - alpha * sum_i gamma^i log pi(atilde_i|s) ].

    Appendix C values (extracted, no longer assumptions): AdamW lr 3e-4
    weight decay 1e-4; target critic update ratio (tau) 0.05; target
    entropy -dim(A^N) = -N*act_dim; batch 256; UTD 1; distributional
    transformer critic (see critic.py). The entropy term inside targets
    keeps Eq. 7's gamma^i discounting.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sear.critic import TwinCritic

LOG_STD_MIN, LOG_STD_MAX = -8.0, 2.0


class ChunkGaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, chunk_len, hidden=512, layers=3):
        super().__init__()
        seq, d = [], obs_dim
        for _ in range(layers):
            seq += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        self.trunk = nn.Sequential(*seq)
        self.mean = nn.Linear(hidden, chunk_len * act_dim)
        self.log_std = nn.Linear(hidden, chunk_len * act_dim)
        self.chunk_len, self.act_dim = chunk_len, act_dim

    def dist(self, obs):
        h = self.trunk(obs)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs, deterministic=False):
        """ Returns actions (B, N, act_dim) in [-1, 1] and per-step log
            probs (B, N) (summed over action dims, tanh-corrected). """
        mean, log_std = self.dist(obs)
        if deterministic:
            pre = mean
        else:
            pre = mean + torch.randn_like(mean) * log_std.exp()
        act = torch.tanh(pre)
        logp = (-0.5 * ((pre - mean) / log_std.exp()) ** 2
                - log_std - 0.5 * np.log(2 * np.pi))
        logp = logp - torch.log(1 - act ** 2 + 1e-6)
        B = obs.shape[0]
        act = act.reshape(B, self.chunk_len, self.act_dim)
        logp = logp.reshape(B, self.chunk_len, self.act_dim).sum(-1)
        return act, logp


class SEARAgent:
    """ One instance = one learner. The trainer builds two: a task agent
        (real replay, env reward) and an explorer (imagined transitions,
        disagreement reward). Identical code, different diet. """

    def __init__(self, obs_dim, act_dim, chunk_len, gamma=0.99, tau=0.05,
                 lr=3e-4, alpha_min=1e-3, device='cpu', critic_kw=None):
        self.chunk_len, self.act_dim, self.gamma = chunk_len, act_dim, gamma
        self.tau, self.device = tau, device
        self.policy = ChunkGaussianPolicy(obs_dim, act_dim, chunk_len
                                          ).to(device)
        self.critic = TwinCritic(obs_dim, act_dim, chunk_len,
                                 **(critic_kw or {})).to(device)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.p_opt = torch.optim.AdamW(self.policy.parameters(), lr=lr,
                                       weight_decay=1e-4)
        self.c_opt = torch.optim.AdamW(
            list(self.critic.q1.parameters()) +
            list(self.critic.q2.parameters()), lr=lr, weight_decay=1e-4)
        self.a_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        # Alpha floor (ours, labeled): with target entropy -dim(A^N) the
        # fresh policy's entropy exceeds the target by ~40 nats, so SAC
        # correctly decays alpha toward zero until Q sharpens. The floor
        # keeps a minimal entropy pressure alive so the policy can never
        # freeze completely while Q is flat.
        self.alpha_min = alpha_min
        g = gamma ** torch.arange(chunk_len, dtype=torch.float32)
        self.gammas = g.to(device)                       # (N,)
        # Appendix C: target entropy = -dim(A^N) = -(N * act_dim)
        self.target_entropy = -float(act_dim * chunk_len)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    # ---------- acting ----------
    @torch.no_grad()
    def act(self, obs_np, deterministic=False):
        obs = torch.as_tensor(obs_np, dtype=torch.float32,
                              device=self.device).reshape(1, -1)
        act, _ = self.policy.sample(obs, deterministic)
        return act[0].cpu().numpy()                      # (N, act_dim)

    # ---------- learning ----------
    def _entropy_bonus(self, logp):
        # sum_i gamma^i log pi(a_i | s): (B, N) -> (B,)
        return (self.gammas * logp).sum(-1)

    def update(self, batch):
        """ batch: torch tensors on device --
            obs (B, D), actions (B, N, A), rewards (B, N),
            next_obs (B, N, D)  [next_obs[:, k-1] = s_{t+k}],
            boot (B, N)  [0 where the k-step bootstrap is cut by terminal],
            valid (B, N) [1 where the k-step target itself is in-episode].
        """
        obs, actions = batch['obs'], batch['actions']
        rewards, next_obs = batch['rewards'], batch['next_obs']
        boot, valid = batch['boot'], batch['valid']
        B, N = rewards.shape

        # --- multi-horizon targets (Eqs. 6-7) ---
        with torch.no_grad():
            flat_next = next_obs.reshape(B * N, -1)
            a_next, logp_next = self.policy.sample(flat_next)
            q_next = self.critic.target_min(flat_next, a_next)   # (B*N, N)
            qhat = q_next[:, -1] - self.alpha.detach().squeeze() * \
                self._entropy_bonus(logp_next)                    # (B*N,)
            qhat = qhat.reshape(B, N)
            disc_rew = self.gammas * rewards                      # (B, N)
            nstep = torch.cumsum(disc_rew, dim=1)                 # (B, N)
            k = torch.arange(1, N + 1, device=self.device, dtype=torch.float32)
            targets = nstep + (self.gamma ** k) * boot * qhat     # (B, N)

        l1, l2 = self.critic.online_logits(obs, actions)     # (B, N, bins)
        tgt_dist = self.critic.q1.two_hot(targets)           # (B, N, bins)
        ce1 = -(tgt_dist * F.log_softmax(l1, -1)).sum(-1)    # (B, N)
        ce2 = -(tgt_dist * F.log_softmax(l2, -1)).sum(-1)
        critic_loss = (valid * (ce1 + ce2)).sum() / \
            (2 * valid.sum().clamp(min=1.0))
        with torch.no_grad():
            q1 = (F.softmax(l1, -1) * self.critic.q1.bin_values).sum(-1)
        self.c_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.critic.q1.parameters()) +
            list(self.critic.q2.parameters()), 10.0)
        self.c_opt.step()

        # --- actor (Eq. 9) ---
        a_pi, logp_pi = self.policy.sample(obs)
        ent = self._entropy_bonus(logp_pi)                        # (B,)
        q_pi = torch.minimum(*self.critic.online(obs, a_pi))[:, -1]
        actor_loss = (self.alpha.detach().squeeze() * ent - q_pi).mean()
        self.p_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
        self.p_opt.step()

        # --- temperature --- (ent is the discounted log-prob sum; compare
        # against the undiscounted paper target via the raw per-step sum)
        alpha_loss = -(self.log_alpha *
                       (logp_pi.sum(-1).detach()
                        + self.target_entropy)).mean()
        self.a_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.a_opt.step()
        with torch.no_grad():
            self.log_alpha.clamp_(min=float(np.log(self.alpha_min)))

        self.critic.soft_update(self.tau)
        with torch.no_grad():
            chunk_abs = a_pi.abs().mean()
        return {'critic_loss': float(critic_loss.item()),
                'actor_loss': float(actor_loss.item()),
                'alpha': float(self.alpha.item()),
                'entropy': float((-ent / self.gammas.sum()).mean().item()),
                'entropy_total': float((-logp_pi.sum(-1)).mean().item()),
                'q_mean': float(q1[:, -1].mean().item()),
                'q_max': float(q1[:, -1].max().item()),
                'chunk_abs_mean': float(chunk_abs.item()),
                'batch_reward_max': float(rewards.max().item())}

    # ---------- persistence ----------
    def state_dict(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'log_alpha': self.log_alpha.detach().cpu()}

    def load_state_dict(self, sd):
        self.policy.load_state_dict(sd['policy'])
        self.critic.load_state_dict(sd['critic'])
        with torch.no_grad():
            self.log_alpha.copy_(sd['log_alpha'].to(self.device))
