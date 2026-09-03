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

import torch

from tdmpc.model import TDMPC2Nets, soft_ce


class RunningScale:
    """ common/scale.py: running trimmed scale. Tracks the 5th-95th percentile
        range of a batch of values (clamped to at least 1) with an EMA at rate
        tau. Used to normalise Q in the policy-prior loss, so its coefficient
        means the same thing at step 1k and step 1M. """

    def __init__(self, tau=0.01, device='cpu'):
        self.tau = tau
        # Kept as a 0-d tensor on the device so update() never syncs the
        # GPU; read it with float() only when logging.
        self.value = torch.ones((), device=device)
        self._pct = torch.tensor([0.05, 0.95], device=device)

    @torch.no_grad()
    def update(self, x):
        x = x.detach().flatten().float()
        if x.numel() < 2:
            return
        lo, hi = torch.quantile(x, self._pct)
        span = torch.clamp(hi - lo, min=1.0)
        new = (1 - self.tau) * self.value + self.tau * span
        self.value = torch.where(torch.isfinite(span), new, self.value)

    def __call__(self):
        return self.value


class TDMPC2Model:
    """ Owns the nets, the optimizers and the update. The scoring-side API
        (encode, rollout_chunk, rollout_pi_chunk, terminal_value,
        chunk_disagreement) is everything the arms call at act time. """

    def __init__(self, obs_dim, action_dim, device, cfg, gamma,
                 num_dyn=None, novelty='mean', novelty_at='path',
                 rollout_chunks=1, chunk_len=None):
        """ num_dyn: dynamics heads; overrides cfg.num_dyn (explore arm).
            novelty: how ensemble disagreement over the latent dims is
            reduced to one number, for BOTH the data reference and the
            explore arm's candidates so the ratio stays meaningful:
              'mean'    plain mean over latent dims (Pathak et al.)
              'reward'  weighted by |d reward_head / d z| -- disagreement in
                        dims the reward head reads counts, disagreement in
                        dims it ignores (arm pose) does not. Early, when the
                        reward head is untrained, the weights are ~uniform
                        and this equals 'mean'. cfg.reward_weight_shrink
                        blends the weights back toward uniform.
            novelty_at, rollout_chunks, chunk_len: how a candidate's path is
            measured by path_disagreement (see there). With
            cfg.ref_mode='rollout' the data reference is measured the same
            way on real replay windows, so the ratio compares like with
            like; 'step' is the earlier one-step reference. """
        self.device = device
        self.action_dim = action_dim
        self.gamma = gamma
        self.cfg = cfg
        self.horizon = int(cfg.horizon)
        self.rho = float(cfg.rho)
        self.tau = float(cfg.tau)
        self.num_q = int(cfg.num_q)
        self.novelty = novelty
        self.novelty_at = novelty_at
        self.rollout_chunks = max(1, int(rollout_chunks))
        self.chunk_len = int(chunk_len) if chunk_len is not None else self.horizon
        self.ref_mode = str(getattr(cfg, 'ref_mode', 'step'))
        assert self.ref_mode in ('step', 'rollout'), self.ref_mode
        self.reward_weight_shrink = float(getattr(cfg, 'reward_weight_shrink', 0.0))
        assert 0.0 <= self.reward_weight_shrink <= 1.0, self.reward_weight_shrink

        self.net = TDMPC2Nets(
            obs_dim, action_dim, latent_dim=cfg.latent_dim, mlp_dim=cfg.mlp_dim,
            enc_dim=cfg.enc_dim, enc_layers=cfg.enc_layers,
            simnorm_dim=cfg.simnorm_dim, num_q=cfg.num_q, dropout=cfg.dropout,
            num_bins=cfg.num_bins, vmin=cfg.vmin, vmax=cfg.vmax,
            num_dyn=(cfg.num_dyn if num_dyn is None else num_dyn)).to(device)
        # Running mean of ensemble disagreement on REAL transitions, updated
        # every update(). The explore arm's bonus is measured against this,
        # so "novel" means "more uncertain than the data", and the bonus
        # anneals as the heads converge on the data. ref_mode 'step': the
        # one-step disagreement at the first real transition of each window.
        # 'rollout': the same path measure a candidate gets
        # (path_disagreement over the window's real actions), so imagined
        # drift is in the reference too and the ratio is not inflated by it.
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
        self.scale = RunningScale(tau=self.tau, device=device)
        self._target_params = list(self.net.Qs_target.parameters())
        self._online_q_params = list(self.net.Qs.parameters())
        self._opt_params = [p for g in self.opt.param_groups for p in g['params']]

        # Same lever as ChunkAgent: the update is a few thousand tiny kernels
        # and launch-bound, so fuse them. mode='default', no CUDA graphs
        # (two backward passes per update). Pure speed change.
        if bool(getattr(cfg, 'compile_nets', False)):
            try:
                opts = dict(mode='default', fullgraph=False)
                self._losses = torch.compile(self._losses, **opts)
                self._td_target = torch.compile(self._td_target, **opts)
                self._pi_loss = torch.compile(self._pi_loss, **opts)
                self.encode = torch.compile(self.encode, **opts)
                self.rollout_chunk = torch.compile(self.rollout_chunk, **opts)
                self.rollout_pi_chunk = torch.compile(self.rollout_pi_chunk, **opts)
                self.terminal_value = torch.compile(self.terminal_value, **opts)
                print('[compile] torch.compile(default) enabled on latent model')
            except Exception as e:
                print(f'[compile] FAILED on latent model, running uncompiled: {e}')
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

    def reward_weights(self, z, a):
        """ |d reward_pred / d z| at (z, a), scaled to MEAN 1 per row: how
            much the reward head reads each latent dim, in units where the
            plain mean is weight 1 everywhere. A row with no gradient at all
            (the head's output layer is zero-initialised, so this is exactly
            the untrained case) falls back to uniform weights, i.e. 'mean'
            mode. (B, latent).

            reward_weight_shrink (lambda) blends toward uniform:
            w <- (1 - lambda) * w + lambda. A sharply peaked reward head
            otherwise puts nearly all the weight on a handful of dims, and
            the novelty ratio then hinges on those dims' ensemble noise.
            lambda 0 = raw weights, 1 = 'mean' mode. """
        with torch.enable_grad():
            zg = z.detach().requires_grad_(True)
            r = self.net.reward_pred(zg, a).sum()
            g, = torch.autograd.grad(r, zg)
        w = g.abs()
        mean = w.mean(-1, keepdim=True)
        w = torch.where(mean > 0, w / (mean + 1e-12), torch.ones_like(w))
        if self.reward_weight_shrink > 0.0:
            lam = self.reward_weight_shrink
            w = (1.0 - lam) * w + lam
        return w

    @torch.no_grad()
    def reduce_disagreement(self, var, z, a):
        """ Per-dim ensemble variance (B, latent) -> one number per row
            (B, 1), by the configured `novelty` rule. Same function for the
            data reference and for candidates. 'reward' is a weighted mean
            with mean-1 weights, so its scale matches 'mean'. """
        if self.novelty == 'reward':
            return (var * self.reward_weights(z, a)).mean(-1, keepdim=True)
        return var.mean(-1, keepdim=True)

    @torch.no_grad()
    def path_disagreement(self, z, actions):
        """ Dynamics-ensemble disagreement of an imagined path, one number
            per row (B,). z: (B, latent) encoded start. actions: (B, T,
            action_dim) rolled through the ensemble MEAN, each step's per-dim
            variance across heads reduced by `novelty`; then rollout_chunks-1
            further chunks of chunk_len steps with the policy prior's mean
            action. novelty_at 'path' = mean over all steps, 'end' = the last
            step only.

            One function for the explore arm's candidates AND (ref_mode
            'rollout') the data reference, so both are measured alike.
            Uncompiled on purpose: the reward-weighted rule needs a gradient
            through the reward head. """
        steps = []
        for k in range(actions.shape[1]):
            a = actions[:, k]
            preds = self.net.next_all(z, a)
            var = preds.var(0, unbiased=False)
            steps.append(self.reduce_disagreement(var, z, a))
            z = preds.mean(0)
        for _ in range(self.rollout_chunks - 1):
            for _ in range(self.chunk_len):
                a = self.net.pi(z)[0]
                preds = self.net.next_all(z, a)
                var = preds.var(0, unbiased=False)
                steps.append(self.reduce_disagreement(var, z, a))
                z = preds.mean(0)
        if self.novelty_at == 'end':
            return steps[-1].squeeze(-1)
        return torch.stack(steps, 0).mean(0).squeeze(-1)

    @torch.no_grad()
    def rollout_pi_chunk(self, z, chunk_len, discount0=1.0):
        """ Same, with actions from the policy prior at each imagined latent.
            This is how a score looks more than one chunk ahead without
            leaving latent space.

            DEVIATION: uses the prior's MEAN action. TD-MPC2's planner samples,
            but sampling injects per-candidate noise into a comparison whose
            whole point is ordering candidates against each other. """
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

    @torch.no_grad()
    def _td_target(self, next_obs, reward, mask):
        """ tdmpc2.py _td_target: reward + gamma * mask * min over two random
            TARGET Q heads at a SAMPLED prior action. Runs on the whole
            (B, H) batch in one pass, as the reference does (one random head
            pair per batch). Returns (B, H, 1) targets and the encoded
            next latents (B, H, latent). """
        B, H = reward.shape[:2]
        next_z = self.net.encode(next_obs)
        flat_z = next_z.reshape(B * H, -1)
        a = self.net.pi(flat_z)[1]
        q = self.net.q_subset(flat_z, a, reduce='min', target=True)
        return reward + self.gamma * mask * q.reshape(B, H, 1), next_z

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
            the consistency loss and the rollout continues through the mean. """
        cfg = self.cfg
        self.net.train()
        td_targets, next_z_real = self._td_target(next_obs, reward, mask)
        total, consistency_loss, reward_loss, value_loss, zs, var0 = self._losses(
            obs[:, 0], next_z_real, action, reward, valid, td_targets)
        if var0 is not None:
            self.update_novelty_reference(zs, action, valid, var0=var0)

        self.opt.zero_grad(set_to_none=True)
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self._opt_params, cfg.grad_clip_norm)
        self.opt.step()

        # update_pi(zs.detach()) in the reference takes ALL horizon+1 latents,
        # the final rolled one included. Latent t+1 is meaningful when step t
        # was inside the episode; latent 0 always is.
        valid_z = torch.cat([torch.ones_like(valid[:, :1]), valid], dim=1)
        pi_metrics = self._update_pi(zs.detach(), valid_z, metrics_on)
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
        return metrics

    def update_novelty_reference(self, zs, action, valid, var0=None):
        """ EMA of the data disagreement (see __init__), from the SAME
            forward pass / parameters the losses used. Returns the new value.

            zs: (B, H+1, latent) rollout latents, zs[:, 0] the encoded real
            start. action: (B, H, action_dim) real actions. valid: (B, H, 1).
            var0: (B, latent) per-dim ensemble variance at step 0, needed by
            ref_mode 'step' only.

            'step':    reduce_disagreement(var0) at the window's first real
                       transition, averaged over the batch.
            'rollout': path_disagreement over the first chunk_len real
                       actions of every window that stays inside one
                       episode; skipped (value unchanged) when none does. """
        if self.ref_mode == 'rollout':
            T = min(self.chunk_len, action.shape[1])
            keep = (valid[:, :T, 0].min(dim=1).values > 0.5).nonzero().squeeze(-1)
            if len(keep) == 0:
                return self.data_disagreement
            with torch.no_grad():
                d = self.path_disagreement(zs[keep, 0].detach(), action[keep, :T]).mean().item()
        else:
            assert var0 is not None, "ref_mode 'step' needs the step-0 variance"
            d = self.reduce_disagreement(var0, zs[:, 0].detach(), action[:, 0]).mean().item()
        self.data_disagreement = 0.99 * self.data_disagreement + 0.01 * d \
            if self.data_disagreement > 1e-8 else d
        return self.data_disagreement

    def _losses(self, obs0, next_z_real, action, reward, valid, td_targets):
        """ The three TD-MPC2 losses on one latent rollout (see update()).
            Every Q head and every dynamics head is evaluated in one batched
            call per step; the per-head means are identical to the old
            per-head loops. Returns the total, the three terms, the rollout
            latents (B, H+1, latent) and the step-0 per-dim dynamics
            disagreement (B, latent) (None with one head). """
        cfg = self.cfg
        horizon = action.shape[1]
        z = self.net.encode(obs0)
        zs = [z]
        consistency_loss = 0.0
        reward_loss = 0.0
        value_loss = 0.0
        rho = 1.0
        dis0 = None
        for t in range(horizon):
            w = valid[:, t]
            reward_loss = reward_loss + rho * (w * soft_ce(
                self.net.reward_logits(z, action[:, t]), reward[:, t],
                cfg.vmin, cfg.vmax, cfg.num_bins)).mean()
            q_logits = self.net.q_logits(z, action[:, t])          # (num_q, B, bins)
            # sum_i mean_b(w * ce_i) / num_q == mean over (i, b) of w * ce
            value_loss = value_loss + rho * (w * soft_ce(
                q_logits, td_targets[:, t], cfg.vmin, cfg.vmax, cfg.num_bins)).mean()
            preds = self.net.next_all(z, action[:, t])             # (num_dyn, B, latent)
            if preds.shape[0] > 1 and t == 0:
                dis0 = preds.detach().var(0, unbiased=False)         # (B, latent)
            consistency_loss = consistency_loss + rho * (
                w * (preds - next_z_real[:, t]).pow(2).mean(-1, keepdim=True)).mean()
            z = preds.mean(0)
            zs.append(z)
            rho *= self.rho

        # tdmpc2.py divides each term by horizon (value_loss also by num_q,
        # folded into the mean above), not by the sum of rho weights.
        consistency_loss = consistency_loss / horizon
        reward_loss = reward_loss / horizon
        value_loss = value_loss / horizon
        total = (cfg.consistency_coef * consistency_loss
                 + cfg.reward_coef * reward_loss
                 + cfg.value_coef * value_loss)
        return total, consistency_loss, reward_loss, value_loss, torch.stack(zs, dim=1), dis0

    def _pi_loss(self, zs):
        """ Network half of _update_pi (prior action, log-prob, average of
            two random Q heads), separated so it can be compiled. """
        _, action, log_prob = self.net.pi(zs)
        flat = lambda x: x.reshape(-1, x.shape[-1])
        q = self.net.q_subset(flat(zs), flat(action), reduce='avg')
        return q.reshape(*zs.shape[:-1], 1), log_prob

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
        for p in self._online_q_params:
            p.requires_grad_(False)
        try:
            q, log_prob = self._pi_loss(zs)
        finally:
            for p in self._online_q_params:
                p.requires_grad_(True)
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
            'diagnosis/q_scale': float(self.scale()),
        }

    @torch.no_grad()
    def _soft_update_target(self):
        # tp <- (1 - tau) tp + tau p, one fused call for all heads.
        torch._foreach_lerp_(self._target_params, self._online_q_params, self.tau)

    def param_norm(self):
        with torch.no_grad():
            norms = torch._foreach_norm([p for p in self.net.parameters() if p.requires_grad])
            return torch.stack(norms).norm().item()

    def state_dict_all(self):
        return {'net': self.net.state_dict()}

    def load_state_dict_all(self, state):
        from tdmpc.model import convert_legacy_state_dict
        self.net.load_state_dict(convert_legacy_state_dict(
            state['net'], self.net.num_q, self.net.num_dyn))
