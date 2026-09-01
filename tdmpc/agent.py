""" The TD-MPC2 latent model as this repo uses it: trained on real replay
    sequences alongside QC-FQL, consumed ONLY by the ChunkSelector.

    It has no influence on QC-FQL training. The actor and critic still live in
    observation space, still update on real chunk transitions from replay, and
    still use the target R_real + gamma^h * mask * Q(s_next). This model only
    decides WHICH of the policy's candidate chunks gets executed, so its
    entire effect on the run flows through the data that selection collects.

    Note on termination: TD-MPC2 has no continue head, so imagined rollouts
    are discounted but never truncated. The TD targets DO respect real
    terminations through the replay mask; it is only the imagined score that
    treats the episode as ongoing. On these cube tasks that is the same
    assumption QC's own best-of-N makes, so the two arms are not being
    scored under different termination models. """

import numpy as np
import torch
import torch.nn.functional as F

from tdmpc.model import TDMPC2Nets, soft_ce


class RunningScale:
    """ TD-MPC2's running scale for the policy-prior loss. Q magnitudes drift
        by orders of magnitude over training; dividing by this keeps the
        prior's gradient scale roughly constant so entropy_coef means the same
        thing at step 1k and step 1M. """

    def __init__(self, tau=0.01):
        self.tau = tau
        self.value = 1.0

    def update(self, x):
        x = float(x)
        if not np.isfinite(x) or x <= 0:
            return
        self.value = (1 - self.tau) * self.value + self.tau * x

    def __call__(self):
        return max(self.value, 1e-6)


class TDMPC2Model:
    """ Owns the nets, the optimizers and the update. Everything the selector
        needs is here: encode, rollout_chunk, rollout_pi_chunk, terminal_value. """

    def __init__(self, obs_dim, action_dim, device, cfg, gamma):
        self.device = device
        self.action_dim = action_dim
        self.gamma = gamma
        self.cfg = cfg
        self.horizon = int(cfg.horizon)
        self.rho = float(cfg.rho)
        self.tau = float(cfg.tau)
        self.num_q = int(cfg.num_q)

        self.net = TDMPC2Nets(
            obs_dim, action_dim, latent_dim=cfg.latent_dim, mlp_dim=cfg.mlp_dim,
            enc_layers=cfg.enc_layers, simnorm_dim=cfg.simnorm_dim,
            num_q=cfg.num_q, dropout=cfg.dropout, num_bins=cfg.num_bins,
            vmin=cfg.vmin, vmax=cfg.vmax).to(device)

        # The encoder trains at a lower rate than the heads (TD-MPC2's
        # enc_lr_scale). It is the one module every other loss backs through,
        # so letting it move at full speed makes the reward and value targets
        # non-stationary for reasons unrelated to the data.
        enc_params = list(self.net.encoder.parameters())
        # The policy prior has its own optimizer and its own loss, so it must
        # NOT also sit in the main one -- TD-MPC2 keeps the two disjoint.
        held = {id(p) for p in enc_params}
        held |= {id(p) for p in self.net.pi_net.parameters()}
        rest = [p for p in self.net.parameters()
                if id(p) not in held and p.requires_grad]
        self.opt = torch.optim.Adam([
            {'params': enc_params, 'lr': cfg.lr * cfg.enc_lr_scale},
            {'params': rest, 'lr': cfg.lr}])
        self.pi_opt = torch.optim.Adam(self.net.pi_net.parameters(), lr=cfg.lr)
        self.scale = RunningScale()
        # Eval mode by default. The Q heads carry dropout, and dropout during
        # SCORING would draw a different mask for every candidate in the
        # batch -- injecting noise into the exact comparison the arm exists to
        # make. update() turns it on for the duration of the update only, the
        # same way TD-MPC2 does.
        self.net.eval()

    # ---------------------------------------------------------------- scoring

    @torch.no_grad()
    def encode(self, obs):
        """ obs: (B, obs_dim) torch tensor -> latent (B, latent_dim).

            No carry, no history: the encoder is Markov, so there is nothing
            to keep filtered between steps. This is why the selector no longer
            needs an observe/record_action pass on every environment step. """
        return self.net.encode(obs)

    @torch.no_grad()
    def rollout_chunk(self, z, chunk_actions, discount0=1.0):
        """ Imagine one chunk of given actions.

            z: (B, latent_dim). chunk_actions: (B, chunk_len, action_dim).
            Returns (z_end, pooled_reward, discount_end) where pooled_reward is
            sum_k discount0 * gamma^k * r_k and discount_end is
            discount0 * gamma^chunk_len. """
        pooled = torch.zeros(z.shape[0], 1, device=z.device, dtype=z.dtype)
        disc = discount0
        for k in range(chunk_actions.shape[1]):
            a = chunk_actions[:, k]
            pooled = pooled + disc * self.net.reward_pred(z, a)
            z = self.net.next(z, a)
            disc = disc * self.gamma
        return z, pooled, disc

    @torch.no_grad()
    def rollout_pi_chunk(self, z, chunk_len, discount0=1.0):
        """ Same, but the actions come from the policy prior at each imagined
            latent. This is how the score looks more than one chunk ahead
            without ever leaving latent space. Uses the prior's MEAN action:
            sampling would inject per-candidate noise into a comparison whose
            whole point is ranking candidates against each other. """
        pooled = torch.zeros(z.shape[0], 1, device=z.device, dtype=z.dtype)
        disc = discount0
        for _ in range(chunk_len):
            a = self.net.pi(z)[0]
            pooled = pooled + disc * self.net.reward_pred(z, a)
            z = self.net.next(z, a)
            disc = disc * self.gamma
        return z, pooled, disc

    @torch.no_grad()
    def terminal_value(self, z):
        """ Q(z, pi(z)) at the end of the imagined horizon, (B, 1).

            Mean over the whole ensemble, not TD-MPC2's two random members:
            the members are re-sampled per call there, and here every
            candidate in one decision must be scored by the same function or
            the ranking picks up ensemble noise instead of value. """
        a = self.net.pi(z)[0]
        return self.net.q_values(z, a).mean(0)

    # --------------------------------------------------------------- training

    def _td_target(self, next_z, reward, mask):
        """ reward + gamma * mask * min over two random target Q members.
            mask is 0 where the real transition terminated, so the bootstrap
            respects real terminations even though imagination does not. """
        a = self.net.pi(next_z)[1]
        idx = torch.randperm(self.num_q, device=next_z.device)[:2]
        q = self.net.q_values(next_z, a, target=True)[idx].min(0).values
        return reward + self.gamma * mask * q

    def update(self, obs, action, reward, mask, valid, metrics_on=True):
        """ One TD-MPC2 joint update on a batch of short real windows.

            obs:    (B, H+1, obs_dim)   real observations
            action: (B, H, action_dim)  actions taken
            reward: (B, H, 1)           reward attributable to each action
            mask:   (B, H, 1)           0 where the transition terminated
            valid:  (B, H, 1)           0 once the window ran past episode end

            Three losses on one latent rollout, exactly as TD-MPC2:
              consistency  the rolled latent must match the encoding of the
                           real next observation (this is what makes the
                           latent predictive without any reconstruction)
              reward       two-hot CE against the real reward
              value        two-hot CE against the TD target
            The policy prior is then updated on the detached rollout latents. """
        cfg = self.cfg
        horizon = action.shape[1]
        self.net.train()

        with torch.no_grad():
            next_z_real = self.net.encode(obs[:, 1:])          # (B, H, latent)
            td_targets = torch.stack([
                self._td_target(next_z_real[:, t], reward[:, t], mask[:, t])
                for t in range(horizon)], dim=1)                # (B, H, 1)

        z = self.net.encode(obs[:, 0])
        zs = [z]
        consistency_loss = 0.0
        reward_loss = 0.0
        value_loss = 0.0
        rho = 1.0
        denom = 0.0
        for t in range(horizon):
            w = valid[:, t]
            reward_loss = reward_loss + rho * (w * soft_ce(
                self.net.reward_logits(z, action[:, t]), reward[:, t],
                cfg.vmin, cfg.vmax, cfg.num_bins)).mean()
            q_logits = self.net.q_logits(z, action[:, t])       # (num_q, B, bins)
            for i in range(self.num_q):
                value_loss = value_loss + rho * (w * soft_ce(
                    q_logits[i], td_targets[:, t],
                    cfg.vmin, cfg.vmax, cfg.num_bins)).mean() / self.num_q
            z = self.net.next(z, action[:, t])
            consistency_loss = consistency_loss + rho * (
                w * F.mse_loss(z, next_z_real[:, t], reduction='none')
                .mean(-1, keepdim=True)).mean()
            zs.append(z)
            denom += rho
            rho *= self.rho

        total = (cfg.consistency_coef * consistency_loss
                 + cfg.reward_coef * reward_loss
                 + cfg.value_coef * value_loss) / max(denom, 1e-8)

        self.opt.zero_grad(set_to_none=True)
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for g in self.opt.param_groups for p in g['params']],
            cfg.grad_clip_norm)
        self.opt.step()

        pi_metrics = self._update_pi(torch.stack(zs[:-1], dim=1).detach(),
                                     valid, metrics_on)
        self._soft_update_target()
        self.net.eval()

        if not metrics_on:
            return {}
        norm = max(denom, 1e-8)
        item = lambda x: (x.detach().item() if torch.is_tensor(x) else float(x))
        metrics = {
            'loss_total': total.item(),
            'loss_consistency': item(consistency_loss) / norm,
            'loss_reward': item(reward_loss) / norm,
            'loss_value': item(value_loss) / norm,
            'grad_norm': grad_norm.item(),
            'diagnosis/td_target_mean': td_targets.mean().item(),
            'diagnosis/td_target_std': td_targets.std().item(),
            'diagnosis/reward_batch_std': reward.std().item(),
        }
        metrics.update(pi_metrics)
        return metrics

    def _update_pi(self, zs, valid, metrics_on=True):
        """ Maximum-entropy policy prior on detached latents. This prior is
            never executed in the environment -- it exists so the score can
            continue past the candidate chunk and so the TD target has an
            action to bootstrap with. """
        cfg = self.cfg
        _, action, log_prob = self.net.pi(zs)
        q = self.net.q_values(zs.reshape(-1, zs.shape[-1]),
                              action.reshape(-1, action.shape[-1]))
        q = q.mean(0).reshape(*zs.shape[:-1], 1)
        self.scale.update(q.abs().mean().detach())
        rho = torch.tensor(
            [self.rho ** t for t in range(zs.shape[1])],
            device=zs.device, dtype=zs.dtype).view(1, -1, 1)
        w = rho * valid
        pi_loss = ((cfg.entropy_coef * log_prob - q) * w).mean() / self.scale()

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
            'diagnosis/pi_q': q.mean().item(),
            'diagnosis/q_scale': self.scale(),
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
