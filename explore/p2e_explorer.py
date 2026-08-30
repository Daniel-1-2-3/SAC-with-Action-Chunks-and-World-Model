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
    a small entropy bonus (Dreamer's actent). Returns are normalized by a
    running 5th-95th percentile range (DreamerV3's return normalization)
    so the entropy bonus stays proportionate at any disagreement scale.

    Extensions behind flags (both default-on in the method config):
    - reward_mix: the imagined reward becomes normalized disagreement plus
      reward_mix * normalized env-reward ADVANTAGE (a second value head is
      trained on the reward head's predictions over the same rollouts).
      Advantage, not raw reward: mastered reward has zero advantage, so
      the pull fades per rung instead of parking the explorer at the first
      cube's zone (OWM's advantage logic applied to the explorer's diet).
    - The update also returns a detached rollout cache (deter states,
      sampled latent indices, advantages) for the OWM optimistic dynamics
      loss applied by the world model.

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
                 actent=3e-4, lr=3e-4, slow_rate=0.02, reward_mix=0.0,
                 device='cpu'):
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
        self.ret_scale = 1.0            # EMA of return percentile range
        self.reward_mix = reward_mix
        # env-reward value stream (for advantages; trained on the reward
        # head's predictions over the same imagined rollouts)
        self.env_value = mlp(feat_dim, 512, 3, 1).to(device)
        self.slow_env_value = mlp(feat_dim, 512, 3, 1).to(device)
        self.slow_env_value.load_state_dict(self.env_value.state_dict())
        for p in self.slow_env_value.parameters():
            p.requires_grad_(False)
        self.ev_opt = torch.optim.Adam(self.env_value.parameters(), lr=lr)
        # scales init as None and seed from the FIRST batch's stats --
        # initializing at 1.0 made the (noisy, untrained) advantage term
        # 30-300x louder than real disagreement (~0.01) for the first
        # ~500 updates, training the explorer on noise
        self.adv_scale = None
        self.dis_scale = None
        self.dis_clip = None            # EMA of disagreement 97.5th pct
        self.mix_warmup = 2000          # updates before the mix reaches full
        self._updates = 0

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
        feats, rewards, logps, deters, zs = [], [], [], [], []
        # anti-exploitation clip: disagreement above a running upper
        # percentile is capped, so single off-manifold spikes cannot
        # dominate the objective -- the explorer must earn reward broadly
        # (standard intrinsic-reward practice, e.g. RND reward clipping)
        for _ in range(self.horizon):
            feat = wm.feat(carry)
            a, logp = self.sample(feat)
            rewards.append(ensemble.disagreement(feat, a))
            feats.append(feat)
            logps.append(logp)
            carry = wm.img_step_grad(carry, a)
            deters.append(carry['deter'].detach())
            zs.append(carry['stoch'].detach())
        feats.append(wm.feat(carry))
        feats = torch.stack(feats, 1)                    # (B, H+1, F)
        rewards = torch.stack(rewards, 1)                # (B, H)
        logps = torch.stack(logps, 1)                    # (B, H)
        with torch.no_grad():
            hi = float(torch.quantile(rewards, 0.975).item())
            self.dis_clip = hi if self.dis_clip is None \
                else 0.99 * self.dis_clip + 0.01 * hi
        rewards = rewards.clamp(max=self.dis_clip)

        def lam_returns(rews, slow_v):
            # lambda-returns, backward recursion (Dreamer):
            # R_t = r_t + gamma * ((1 - lam) * v_{t+1} + lam * R_{t+1})
            rets = [slow_v[:, -1]]
            for t in reversed(range(self.horizon)):
                rets.append(rews[:, t] + self.gamma * (
                    (1 - self.lam) * slow_v[:, t + 1]
                    + self.lam * rets[-1]))
            return torch.stack(rets[::-1][:-1], 1)       # (B, H)

        with torch.no_grad():
            # env-reward stream: predicted reward on arrival states,
            # advantages against the env value head
            env_rews = wm.pred_reward(feats[:, 1:].detach())     # (B, H)
            slow_ev = self.slow_env_value(feats.detach()).squeeze(-1)
            env_rets = lam_returns(env_rews, slow_ev)
            env_adv = env_rets - slow_ev[:, :-1]                 # (B, H)
            a_range = float((torch.quantile(env_adv, 0.95)
                             - torch.quantile(env_adv, 0.05)).item())
            d_range = float((torch.quantile(rewards, 0.95)
                             - torch.quantile(rewards, 0.05)).item())
            self.adv_scale = a_range if self.adv_scale is None \
                else 0.99 * self.adv_scale + 0.01 * a_range
            self.dis_scale = d_range if self.dis_scale is None \
                else 0.99 * self.dis_scale + 0.01 * d_range
        self._updates += 1
        raw_disagreement = rewards            # kept for the diagnostic
        if self.reward_mix > 0:
            # mix ramps in over mix_warmup updates so the env-value head
            # has learned a baseline before its advantages steer anything
            mix = self.reward_mix * min(1.0, self._updates / self.mix_warmup)
            rewards = rewards / max(1e-8, self.dis_scale) + \
                mix * env_adv / max(1e-8, self.adv_scale)

        with torch.no_grad():
            slow_v = self.slow_value(feats).squeeze(-1)  # (B, H+1)
        returns = lam_returns(rewards, slow_v)

        # DreamerV3 return normalization: divide by an EMA of the batch
        # returns' 5th-95th percentile range (floored at 1e-8 -> capped by
        # max(1, .) semantics below), keeping actent scale-free
        with torch.no_grad():
            lo = torch.quantile(returns, 0.05)
            hi = torch.quantile(returns, 0.95)
            self.ret_scale = 0.99 * self.ret_scale + \
                0.01 * float((hi - lo).item())
        norm = max(1e-8, self.ret_scale)
        # actor: ascend the (normalized) return + entropy bonus
        actor_loss = (-returns / norm + self.actent * logps).mean()
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
        # env-reward value: regress its own lambda-returns
        ev = self.env_value(feats[:, :-1].detach()).squeeze(-1)
        env_value_loss = ((ev - env_rets) ** 2).mean()
        self.ev_opt.zero_grad(set_to_none=True)
        env_value_loss.backward()
        self.ev_opt.step()
        with torch.no_grad():
            for p, sp in zip(self.env_value.parameters(),
                             self.slow_env_value.parameters()):
                sp.mul_(1 - self.slow_rate).add_(self.slow_rate * p)
        with torch.no_grad():
            for p, sp in zip(self.value.parameters(),
                             self.slow_value.parameters()):
                sp.mul_(1 - self.slow_rate).add_(self.slow_rate * p)
        # diagnostic: explorer's found novelty vs chance at the same states
        with torch.no_grad():
            rand_a = torch.rand_like(
                torch.empty(feats.shape[0], self.act_dim,
                            device=feats.device)) * 2 - 1
            d_rand = ensemble.disagreement(feats[:, 0].detach(), rand_a)
        # raw disagreement vs random -- unpolluted by the mix
        ratio = float((raw_disagreement[:, 0].detach().mean()
                       / d_rand.mean().clamp(min=1e-12)).item())
        # rollout cache for the OWM optimistic dynamics loss (detached)
        self.last_rollout = dict(
            deters=torch.stack(deters, 1),               # (B, H, deter)
            zs=torch.stack(zs, 1),                       # (B, H, S, C)
            advantages=env_adv / max(1e-8, self.adv_scale))
        return {'actor_loss': float(actor_loss.item()),
                'vs_random_ratio': ratio,
                'return_scale': float(norm),
                'env_value_loss': float(env_value_loss.item()),
                'env_adv_mean': float(env_adv.mean().item()),
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