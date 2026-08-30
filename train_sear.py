""" Plan2Explore's exploration agent, faithful to the paper:

      "We learn the exploration policy using Dreamer: the exploration
       policy is optimized purely from trajectories imagined under the
       model to maximize the intrinsic rewards computed by the model
       itself."  (P2E Sec. 3.1)

    Actor and value operate on LATENT model states. Training: start states
    are posterior latents from replay sequences; the actor rolls the prior
    forward H steps with gradients flowing through the dynamics (pathwise,
    straight-through categorical samples); each imagined (state, action)
    is rewarded with ensemble disagreement; values regress lambda-returns
    against an EMA target network; the actor ascends the lambda-return plus
    a small entropy bonus (Dreamer's actent).

    Deviations, labeled: single-step actor (P2E's is single-step too --
    the CHUNKED task learner is SEAR's side of the house; replay stores
    per-step actions, so SEAR consumes explorer-collected data unchanged);
    the RSSM substrate is the DreamerV3-style categorical port rather than
    P2E's Gaussian PlaNet latents.
"""
import numpy as np
import torch
import torch.nn as nn

from sear.agent import LOG_STD_MAX, LOG_STD_MIN


def mlp(inp, units, layers, out):
    seq, d = [], inp
    for _ in range(layers):
        seq += [nn.Linear(d, units), nn.SiLU()]
        d = units
    seq += [nn.Linear(d, out)]
    return nn.Sequential(*seq)


class P2EExplorer:
    def __init__(self, feat_dim, act_dim, horizon=15, gamma=0.99, lam=0.95,
                 actent=3e-4, lr=3e-4, slow_rate=0.02, device='cpu'):
        self.feat_dim, self.act_dim = feat_dim, act_dim
        self.horizon, self.gamma, self.lam = horizon, gamma, lam
        self.actent, self.device = actent, device
        self.actor = mlp(feat_dim, 512, 3, 2 * act_dim).to(device)
        self.value = mlp(feat_dim, 512, 3, 1).to(device)
        self.slow_value = mlp(feat_dim, 512, 3, 1).to(device)
        self.slow_value.load_state_dict(self.value.state_dict())
        for p in self.slow_value.parameters():
            p.requires_grad_(False)
        self.slow_rate = slow_rate
        self.a_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.v_opt = torch.optim.Adam(self.value.parameters(), lr=lr)

    # ---------- policy ----------
    def _dist(self, feat):
        out = self.actor(feat)
        mean, log_std = out.chunk(2, -1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, feat, deterministic=False):
        mean, log_std = self._dist(feat)
        pre = mean if deterministic else \
            mean + torch.randn_like(mean) * log_std.exp()
        act = torch.tanh(pre)
        logp = (-0.5 * ((pre - mean) / log_std.exp()) ** 2 - log_std
                - 0.5 * np.log(2 * np.pi))
        logp = (logp - torch.log(1 - act ** 2 + 1e-6)).sum(-1)
        return act, logp

    @torch.no_grad()
    def act(self, feat, deterministic=False):
        a, _ = self.sample(feat, deterministic)
        return a.cpu().numpy()

    # ---------- imagination training (Dreamer-style) ----------
    def update(self, wm, ensemble, start_carry):
        """ start_carry: dict of latent tensors for B start states (posterior,
            detached). Rolls H steps through the prior WITH gradients. """
        carry = {k: v.detach() for k, v in start_carry.items()}
        feats, rewards, logps = [], [], []
        for _ in range(self.horizon):
            feat = wm.feat(carry)
            a, logp = self.sample(feat)
            rewards.append(ensemble.disagreement(feat, a))
            feats.append(feat)
            logps.append(logp)
            carry = wm.img_step_grad(carry, a)
        feats.append(wm.feat(carry))
        feats = torch.stack(feats, 1)                    # (B, H+1, F)
        rewards = torch.stack(rewards, 1)                # (B, H)
        logps = torch.stack(logps, 1)                    # (B, H)

        with torch.no_grad():
            slow_v = self.slow_value(feats).squeeze(-1)  # (B, H+1)
        # lambda-returns, backward recursion (Dreamer):
        # R_t = r_t + gamma * ((1 - lam) * v_{t+1} + lam * R_{t+1})
        returns = [slow_v[:, -1]]
        for t in reversed(range(self.horizon)):
            returns.append(rewards[:, t] + self.gamma * (
                (1 - self.lam) * slow_v[:, t + 1]
                + self.lam * returns[-1]))
        returns = torch.stack(returns[::-1][:-1], 1)     # (B, H)

        # actor: ascend the return through the dynamics + entropy bonus
        actor_loss = (-returns + self.actent * logps).mean()
        self.a_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)
        self.a_opt.step()

        # value: regress lambda-returns on detached features
        v = self.value(feats[:, :-1].detach()).squeeze(-1)
        value_loss = ((v - returns.detach()) ** 2).mean()
        self.v_opt.zero_grad(set_to_none=True)
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.value.parameters(), 100.0)
        self.v_opt.step()
        with torch.no_grad():
            for p, sp in zip(self.value.parameters(),
                             self.slow_value.parameters()):
                sp.mul_(1 - self.slow_rate).add_(self.slow_rate * p)
        return {'actor_loss': float(actor_loss.item()),
                'value_loss': float(value_loss.item()),
                'imag_reward_mean': float(rewards.mean().item()),
                'imag_reward_max': float(rewards.max().item()),
                'imag_value_mean': float(returns.mean().item()),
                'imag_action_abs': float(
                    logps.new_tensor(0.0).item()) if False else float(
                    torch.tanh(self._dist(feats[:, 0].detach())[0]
                               ).abs().mean().item())}

    # ---------- persistence ----------
    def state_dict(self):
        return {'actor': self.actor.state_dict(),
                'value': self.value.state_dict(),
                'slow_value': self.slow_value.state_dict()}

    def load_state_dict(self, sd):
        self.actor.load_state_dict(sd['actor'])
        self.value.load_state_dict(sd['value'])
        self.slow_value.load_state_dict(sd['slow_value'])