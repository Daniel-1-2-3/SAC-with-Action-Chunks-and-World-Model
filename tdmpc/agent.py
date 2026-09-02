""" The TD-MPC2 latent model as this repo uses it: trained on real replay
    windows alongside QC-FQL, consumed by the arms in whatever way each arm
    defines (chunk scoring, value expansion, exploration bonus).

    Port of tdmpc2.py `_update` / `update_pi` / `_td_target`, checked against
    github.com/nicklashansen/tdmpc2. Deviations are marked DEVIATION.

    Note on termination: TD-MPC2 (non-episodic config) has no termination
    head, so imagined rollouts are discounted but never truncated. The TD
    targets DO respect real terminations through the replay mask; only
    imagination treats the episode as ongoing. QC's own bootstrap makes the
    same assumption inside a chunk, so the arms are not scored under
    different termination models. """

import numpy as np
import torch
import torch.nn.functional as F

from tdmpc.model import TDMPC2Nets, soft_ce


class RunningScale:
    """ common/scale.py: running trimmed scale. Tracks the 5th-95th percentile
        range of a batch of values (clamped to at least 1) with an EMA at rate
        tau. Used to normalise Q in the policy-prior loss and returns in the
        optimistic loss, so their coefficients mean the same thing at step 1k
        and step 1M. """

    def __init__(self, tau=0.01):
        self.tau = tau
        self.value = 1.0

    @torch.no_grad()
    def update(self, x):
        x = x.detach().flatten().float()
        if x.numel() < 2:
            return
        lo, hi = torch.quantile(x, torch.tensor([0.05, 0.95], device=x.device))
        span = float(torch.clamp(hi - lo, min=1.0))
        if np.isfinite(span):
            self.value = (1 - self.tau) * self.value + self.tau * span

    def __call__(self):
        return self.value


class TDMPC2Model:
    """ Owns the nets, the optimizers and the update. The scoring-side API
        (encode, rollout_chunk, rollout_pi_chunk, terminal_value,
        chunk_disagreement) is everything the arms call at act time. """

    def __init__(self, obs_dim, action_dim, device, cfg, gamma, optimism=None,
                 num_dyn=None):
        """ optimism: None, or a config block {alpha, eta, lam, imag_batch,
            horizon, sample_mix} enabling the Optimistic-World-Models loss.
            num_dyn: dynamics heads; overrides cfg.num_dyn (explore arm). """
        self.device = device
        self.action_dim = action_dim
        self.gamma = gamma
        self.cfg = cfg
        self.horizon = int(cfg.horizon)
        self.rho = float(cfg.rho)
        self.tau = float(cfg.tau)
        self.num_q = int(cfg.num_q)
        self.optimism = optimism

        self.net = TDMPC2Nets(
            obs_dim, action_dim, latent_dim=cfg.latent_dim, mlp_dim=cfg.mlp_dim,
            enc_dim=cfg.enc_dim, enc_layers=cfg.enc_layers,
            simnorm_dim=cfg.simnorm_dim, num_q=cfg.num_q, dropout=cfg.dropout,
            num_bins=cfg.num_bins, vmin=cfg.vmin, vmax=cfg.vmax,
            num_dyn=(cfg.num_dyn if num_dyn is None else num_dyn)).to(device)
        # Running mean of ensemble disagreement on REAL transitions, updated
        # every update(). The explore arm's bonus is measured against this,
        # so "novel" means "more uncertain than the data", and the bonus
        # anneals as the heads converge on the data.
        self.data_disagreement = 1e-8

        # tdmpc2.py: the encoder trains at lr * enc_lr_scale, every other
        # module at lr, and the policy prior has its OWN optimizer (eps 1e-5)
        # and is not in this one.
        enc_params = list(self.net.encoder.parameters())
        held = {id(p) for p in enc_params}
        held |= {id(p) for p in self.net.pi_net.parameters()}
        rest = [p for p in self.net.parameters()
                if id(p) not in held and p.requires_grad]
        self.opt = torch.optim.Adam([
            {'params': enc_params, 'lr': cfg.lr * cfg.enc_lr_scale},
            {'params': rest, 'lr': cfg.lr}])
        self.pi_opt = torch.optim.Adam(self.net.pi_net.parameters(), lr=cfg.lr,
                                       eps=1e-5)
        self.scale = RunningScale(tau=self.tau)
        self.ret_scale = RunningScale(tau=0.01)
        # Eval mode by default. The Q heads carry dropout, and dropout during
        # SCORING would draw a different mask for every candidate in the
        # batch -- injecting noise into the exact comparison the arms exist to
        # make. update() turns it on for its own duration only, as tdmpc2.py
        # does.
        self.net.eval()

    # ---------------------------------------------------------------- scoring

    @torch.no_grad()
    def encode(self, obs):
        """ obs: (B, obs_dim) -> latent (B, latent_dim). No carry, no history:
            the encoder is Markov, so nothing is kept between steps. """
        return self.net.encode(obs)

    @torch.no_grad()
    def rollout_chunk(self, z, chunk_actions, discount0=1.0):
        """ Imagine one chunk of GIVEN actions.

            z: (B, latent_dim). chunk_actions: (B, chunk_len, action_dim).
            Returns (z_end, pooled_reward, discount_end, disagreement) where
            pooled_reward = sum_k discount0 * gamma^k * r_k, discount_end =
            discount0 * gamma^chunk_len, and disagreement is the mean ensemble
            variance along the rollout (zero with one dynamics head). """
        pooled = torch.zeros(z.shape[0], 1, device=z.device, dtype=z.dtype)
        dis = torch.zeros_like(pooled)
        disc = discount0
        n = chunk_actions.shape[1]
        for k in range(n):
            a = chunk_actions[:, k]
            pooled = pooled + disc * self.net.reward_pred(z, a)
            z, d = self._step(z, a)
            dis = dis + d / n
            disc = disc * self.gamma
        return z, pooled, disc, dis

    def _step(self, z, a):
        """ One dynamics step: (mean next latent, ensemble disagreement),
            from ONE forward of the ensemble. """
        preds = self.net.next_all(z, a)
        if preds.shape[0] == 1:
            return preds[0], torch.zeros(z.shape[0], 1, device=z.device, dtype=z.dtype)
        return preds.mean(0), preds.var(0, unbiased=False).mean(-1, keepdim=True)

    @torch.no_grad()
    def rollout_pi_chunk(self, z, chunk_len, discount0=1.0):
        """ Same, with actions from the policy prior at each imagined latent.
            This is how a score looks more than one chunk ahead without
            leaving latent space.

            DEVIATION: uses the prior's MEAN action. TD-MPC2's planner samples,
            but sampling injects per-candidate noise into a comparison whose
            whole point is ranking candidates against each other. """
        pooled = torch.zeros(z.shape[0], 1, device=z.device, dtype=z.dtype)
        dis = torch.zeros_like(pooled)
        disc = discount0
        for _ in range(chunk_len):
            a = self.net.pi(z)[0]
            pooled = pooled + disc * self.net.reward_pred(z, a)
            z, d = self._step(z, a)
            dis = dis + d / chunk_len
            disc = disc * self.gamma
        return z, pooled, disc, dis

    @torch.no_grad()
    def terminal_value(self, z):
        """ Q(z, pi(z)) at the end of the imagined horizon, (B, 1).

            DEVIATION from _estimate_value, which uses a sampled prior action
            and the average of two random Q heads: both are fresh noise per
            call, and every candidate in one decision must be scored by the
            same function. So: the prior's mean action, mean over the whole
            ensemble. """
        a = self.net.pi(z)[0]
        return self.net.q_values(z, a).mean(0)

    # --------------------------------------------------------------- training

    def _td_target(self, next_z, reward, mask):
        """ tdmpc2.py _td_target: reward + gamma * mask * min over two random
            TARGET Q heads at a SAMPLED prior action. """
        a = self.net.pi(next_z)[1]
        return reward + self.gamma * mask * \
            self.net.q_subset(next_z, a, reduce='min', target=True)

    def update(self, obs, next_obs, action, reward, mask, valid, metrics_on=True):
        """ One TD-MPC2 joint update on a batch of consecutive real
            transitions (see ChunkTransitionReplay.sample_model_windows).

            obs, next_obs: (B, H, obs_dim)     action: (B, H, action_dim)
            reward, mask, valid: (B, H, 1)     mask 0 at a real termination,
                                               valid 0 once past episode end

            Three losses on one latent rollout, exactly as tdmpc2.py _update:
              consistency  rolled latent vs sg(enc(next_obs)) -- what makes
                           the latent predictive without reconstruction
              reward       two-hot CE against the real reward
              value        two-hot CE against the TD target
            then the policy prior on the detached rollout latents, then the
            target-Q soft update. With a dynamics ensemble, every head gets
            the consistency loss and the rollout continues through the mean.
            With `optimism`, the RBMLE loss is added to the total. """
        cfg = self.cfg
        B, horizon = action.shape[:2]
        self.net.train()

        with torch.no_grad():
            next_z_real = self.net.encode(next_obs)                # (B, H, latent)
            td_targets = torch.stack([
                self._td_target(next_z_real[:, t], reward[:, t], mask[:, t])
                for t in range(horizon)], dim=1)                   # (B, H, 1)

        z = self.net.encode(obs[:, 0])
        zs = [z]
        consistency_loss = 0.0
        reward_loss = 0.0
        value_loss = 0.0
        rho = 1.0
        for t in range(horizon):
            w = valid[:, t]
            reward_loss = reward_loss + rho * (w * soft_ce(
                self.net.reward_logits(z, action[:, t]), reward[:, t],
                cfg.vmin, cfg.vmax, cfg.num_bins)).mean()
            q_logits = self.net.q_logits(z, action[:, t])          # (num_q, B, bins)
            for i in range(self.num_q):
                value_loss = value_loss + rho * (w * soft_ce(
                    q_logits[i], td_targets[:, t],
                    cfg.vmin, cfg.vmax, cfg.num_bins)).mean() / self.num_q
            preds = self.net.next_all(z, action[:, t])             # (num_dyn, B, latent)
            if preds.shape[0] > 1 and t == 0:
                d = preds.detach().var(0, unbiased=False).mean().item()
                self.data_disagreement = 0.99 * self.data_disagreement + 0.01 * d \
                    if self.data_disagreement > 1e-8 else d
            for i in range(preds.shape[0]):
                consistency_loss = consistency_loss + rho * (
                    w * F.mse_loss(preds[i], next_z_real[:, t], reduction='none')
                    .mean(-1, keepdim=True)).mean() / preds.shape[0]
            z = preds.mean(0)
            zs.append(z)
            rho *= self.rho

        # tdmpc2.py divides each term by horizon (value_loss also by num_q,
        # done inside the loop above), not by the sum of rho weights.
        consistency_loss = consistency_loss / horizon
        reward_loss = reward_loss / horizon
        value_loss = value_loss / horizon
        total = (cfg.consistency_coef * consistency_loss
                 + cfg.reward_coef * reward_loss
                 + cfg.value_coef * value_loss)

        opt_metrics = {}
        if self.optimism is not None:
            opt_loss, opt_metrics = self._optimistic_loss(obs[:, 0], metrics_on)
            total = total + opt_loss

        self.opt.zero_grad(set_to_none=True)
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for g in self.opt.param_groups for p in g['params']],
            cfg.grad_clip_norm)
        self.opt.step()

        # update_pi(zs.detach()) in the reference takes ALL horizon+1 latents,
        # the final rolled one included. Latent t+1 is meaningful when step t
        # was inside the episode; latent 0 always is.
        valid_z = torch.cat([torch.ones_like(valid[:, :1]), valid], dim=1)
        pi_metrics = self._update_pi(torch.stack(zs, dim=1).detach(),
                                     valid_z, metrics_on)
        self._soft_update_target()
        self.net.eval()

        if not metrics_on:
            return {}
        item = lambda x: (x.detach().item() if torch.is_tensor(x) else float(x))
        metrics = {
            'loss_total': total.item(),
            'loss_consistency': item(consistency_loss),
            'loss_reward': item(reward_loss),
            'loss_value': item(value_loss),
            'grad_norm': grad_norm.item(),
            'diagnosis/td_target_mean': td_targets.mean().item(),
            'diagnosis/td_target_std': td_targets.std().item(),
            'diagnosis/reward_batch_std': reward.std().item(),
        }
        metrics.update(pi_metrics)
        metrics.update(opt_metrics)
        return metrics

    def _update_pi(self, zs, valid, metrics_on=True):
        """ tdmpc2.py update_pi: maximum-entropy policy prior on detached
            latents. This prior is never executed in the environment -- it
            exists so a score can continue past the candidate chunk and so
            the TD target has an action to bootstrap with.

              pi_loss = mean_t rho^t * mean_b [ -(entropy_coef * scaled_entropy
                                                  + Q_avg / scale) ]

            Q is the average of two random heads with its PARAMETERS
            detached (the reference's _detach_Qs): gradient reaches the prior
            through the action input only. scaled_entropy is -log_prob times
            action_dim. Only Q is divided by the running scale. """
        cfg = self.cfg
        _, action, log_prob = self.net.pi(zs)
        flat = lambda x: x.reshape(-1, x.shape[-1])
        q_params = list(self.net.Qs.parameters())
        for p in q_params:
            p.requires_grad_(False)
        try:
            q = self.net.q_subset(flat(zs), flat(action), reduce='avg')
        finally:
            for p in q_params:
                p.requires_grad_(True)
        q = q.reshape(*zs.shape[:-1], 1)
        self.scale.update(q[:, 0])
        q = q / self.scale()
        scaled_entropy = -log_prob * self.action_dim
        rho = torch.tensor(
            [self.rho ** t for t in range(zs.shape[1])],
            device=zs.device, dtype=zs.dtype)
        per_step = -(cfg.entropy_coef * scaled_entropy + q) * valid      # (B, T, 1)
        pi_loss = (per_step.mean(dim=(0, 2)) * rho).mean()

        self.pi_opt.zero_grad(set_to_none=True)
        pi_loss.backward()
        pi_grad = torch.nn.utils.clip_grad_norm_(
            self.net.pi_net.parameters(), cfg.grad_clip_norm)
        self.pi_opt.step()

        if not metrics_on:
            return {}
        return {
            'loss_pi': pi_loss.item(),
            'diagnosis/pi_grad_norm': pi_grad.item(),
            'diagnosis/pi_entropy': (-log_prob).mean().item(),
            'diagnosis/pi_q_scaled': q.mean().item(),
            'diagnosis/q_scale': self.scale(),
        }

    def _optimistic_loss(self, obs0, metrics_on):
        """ Optimistic World Models (Mete et al. 2026), eq. 10, on the SimNorm
            latent read as G categoricals:

              L_opt = -alpha * sum_l A_l * log p(z_{l+1} | z_l, a_l)
                      - eta  * sum_l H(p(. | z_l, a_l))

            An imagined trajectory is rolled from real encoded starts with
            the policy prior, SAMPLING each next latent from the dynamics'
            categoricals (straight-through) so the draw has a likelihood.
            A_l = (G^lam_l - V(z_l)) / max(1, S), the lambda-return over
            imagined rewards bootstrapped by Q(z, pi(z)), S the running
            5-95 percentile range of those returns. Gradient reaches the
            DYNAMICS only: starts are detached, advantages are detached.

            The effect is RBMLE's: transitions that turned out better than
            the value expected get their likelihood raised, so imagination
            drifts optimistic, and the scorer that reads it prefers chunks
            the optimistic model likes. alpha must be tiny (paper: 1e-4). """
        o = self.optimism
        L = int(o.horizon)
        n = min(int(o.imag_batch), obs0.shape[0])
        mix = float(o.sample_mix)
        with torch.no_grad():
            z = self.net.encode(obs0[:n])
        logps, ents, rewards, values = [], [], [], []
        for _ in range(L):
            with torch.no_grad():
                a = self.net.pi(z)[1]
                rewards.append(self.net.reward_pred(z, a))
                values.append(self.net.q_subset(z, a, reduce='avg'))
            p_next = self.net.next(z, a)
            ents.append(self.net.latent_entropy(p_next))
            onehot, logp = self.net.sample_latent(p_next)
            # The heads only ever train on soft SimNorm latents, so a pure
            # one-hot draw is off-distribution for them. Continue the rollout
            # on a convex mix of the draw and the prediction: still inside
            # each simplex, still dependent on the draw (so the advantage
            # depends on it and the RBMLE gradient is non-zero in
            # expectation), with sample_mix setting how far toward the
            # vertex it goes.
            z = (1.0 - mix) * p_next + mix * onehot
            logps.append(logp)
        with torch.no_grad():
            v_end = self.terminal_value(z.detach())
            # lambda-return backwards from the bootstrap:
            #   G_l = r_l + gamma * [(1 - lam) V(z_{l+1}) + lam * G_{l+1}],
            #   G_{L-1} = r_{L-1} + gamma * V(z_L).
            # values[l] is V(z_l), so the bootstrap for step l is values[l+1]
            # and v_end for the last step.
            lam = float(o.lam)
            next_values = values[1:] + [v_end]
            ret = v_end
            returns = [None] * L
            for l in reversed(range(L)):
                if l == L - 1:
                    ret = rewards[l] + self.gamma * v_end
                else:
                    ret = rewards[l] + self.gamma * ((1 - lam) * next_values[l] + lam * ret)
                returns[l] = ret
            returns_t = torch.stack(returns, 0)                    # (L, n, 1)
            values_t = torch.stack(values, 0)
            self.ret_scale.update(returns_t)
            adv = (returns_t - values_t) / max(1.0, self.ret_scale())
        logp_t = torch.stack(logps, 0)
        ent_t = torch.stack(ents, 0)
        loss = -float(o.alpha) * (adv * logp_t).sum(0).mean() \
               - float(o.eta) * ent_t.sum(0).mean()
        if not metrics_on:
            return loss, {}
        return loss, {
            'optimism/loss': loss.item(),
            'optimism/adv_mean': adv.mean().item(),
            'optimism/adv_std': adv.std().item(),
            'optimism/logp_mean': logp_t.mean().item(),
            'optimism/latent_entropy': ent_t.mean().item(),
            'optimism/return_scale': self.ret_scale(),
        }

    @torch.no_grad()
    def _soft_update_target(self):
        for p, tp in zip(self.net.Qs.parameters(), self.net.Qs_target.parameters()):
            tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    def param_norm(self):
        with torch.no_grad():
            sq = sum(float(p.pow(2).sum()) for p in self.net.parameters()
                     if p.requires_grad)
        return sq ** 0.5

    def state_dict_all(self):
        return {'net': self.net.state_dict()}

    def load_state_dict_all(self, state):
        self.net.load_state_dict(state['net'])
