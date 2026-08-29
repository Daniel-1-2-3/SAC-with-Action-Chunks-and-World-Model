import jax
import numpy as np
import torch

from helpers.interop import jax_to_torch

class ExploreSelector:
    """ Disagreement-driven chunk selection for ONLINE COLLECTION only. The
        single place the world model influences this trainer.

        At each chunk boundary: sample n candidate chunks from the one-step
        policy at the raw observation and execute the argmax of

            score = Q(s, chunk) + beta * z(novelty)

        where novelty is the model's DISAGREEMENT about the chunk's outcome:
        the std, across `draws` stochastic RSSM rollouts, of the latent
        features the chunk ends at, averaged over feature dims. States the
        model has seen resolve consistently (low spread); states it hasn't
        seen produce scattered imaginations (high spread). This signal is
        large and varying exactly where the extrinsic reward is a flat -3
        floor -- which is why it attacks THIS task's bottleneck (zero online
        successes) where reward-based scoring measurably could not.

        Q comes from the online critic on the REAL current observation -- no
        decoder, no imagined reward, no latents in any training loss. Training
        stays bit-for-bit QC-FQL; eval acts with the bare policy, because the
        bonus is a data-collection device, not part of the learned behavior.

        Novelty is z-normalized by running Welford statistics so beta is in
        Q-units. Until `warmup` decisions have been scored the bonus is held
        at zero (the normalizer is not yet trustworthy) while statistics
        accumulate.

        The control ladder this gives you:
            explore_n = 1   -> exact QC-FQL (no candidates, no model)
            explore_beta = 0 -> critic-only best-of-N, i.e. the QC paper's
                                own method, with the model still unused
            explore_beta > 0 -> ours

        The posterior latent is maintained by one encode_step per env step
        (observe); that is scorer plumbing, not a policy input.

        v2 (novelty_mode='rnd'): novelty comes from an RNDNovelty module
        scoring the imagined END feature of each candidate (one rollout per
        candidate, no draws) instead of the std over stochastic draws. RND
        error decays wherever the policy has actually been, so the bonus
        self-anneals per state. Pair it with norm_freeze > 0: a tracking
        normalizer would re-inflate the decaying signal and cancel the
        anneal, so the Welford stats are frozen after that many scored
        values and beta keeps an early-training scale. """

    def __init__(self, bridge, policy, action_dim, chunk_len, n, beta, draws,
                 device, warmup=100, novelty_mode='draws', norm_freeze=0,
                 rnd=None):
        self.bridge = bridge
        self.policy = policy
        self.action_dim = action_dim
        self.chunk_len = chunk_len
        self.n = int(n)
        self.enabled = self.n > 1
        self.beta = float(beta)
        self.draws = int(draws)
        self.device = device
        self.warmup = int(warmup)
        self.novelty_mode = str(novelty_mode)
        self.norm_freeze = int(norm_freeze)
        self.rnd = rnd
        # Welford running stats over raw novelty values
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._stats = {}
        self.reset()

    def reset(self):
        """ Call at every episode start (env.reset). """
        if not self.enabled:
            return
        self.enc, self.dyn = self.bridge.init_encode(1)
        self.prevact = np.zeros((1, self.action_dim), dtype=np.float32)
        self.is_first = np.array([True])

    def observe(self, state_1d):
        """ Call once per env step with the CURRENT raw observation, before
            acting, so select() imagines from an up-to-date latent. """
        if not self.enabled:
            return
        s = np.asarray(state_1d, dtype=np.float32).reshape(1, -1)
        self.enc, self.dyn, inp = self.bridge.encode_step(
            self.enc, self.dyn, s, self.prevact, self.is_first)
        self.is_first = np.array([False])
        if self.rnd is not None and self.beta != 0.0:
            # Visited-state feature -> RND training data. Same feature space
            # the imagined end states are scored in.
            self.rnd.add(jax.device_get(inp))

    def record_action(self, action_1d):
        """ Call after env.step with the action that was EXECUTED. """
        if not self.enabled:
            return
        self.prevact = np.asarray(action_1d, dtype=np.float32).reshape(1, -1)

    def _norm_update(self, values):
        for v in values:
            self._count += 1
            d = v - self._mean
            self._mean += d / self._count
            self._m2 += d * (v - self._mean)

    def _norm_std(self):
        if self._count < 2:
            return 1.0
        return max((self._m2 / self._count) ** 0.5, 1e-6)

    def _acc(self, key, value):
        s, c = self._stats.get(key, (0.0, 0))
        self._stats[key] = (s + float(value), c + 1)

    def pop_stats(self):
        out = {f'explore/{k}': s / c for k, (s, c) in self._stats.items() if c > 0}
        self._stats = {}
        return out

    @torch.no_grad()
    def select(self, state_1d):
        """ state_1d: (obs_dim,) raw observation, the same one observe() just
            saw. Returns (chunk_len, action_dim). """
        if not self.enabled:
            return self.policy.act(np.asarray(state_1d, dtype=np.float32),
                                   eval_mode=False)

        feat = torch.as_tensor(np.asarray(state_1d, dtype=np.float32),
                               device=self.device).reshape(1, -1)
        feat_n = feat.repeat(self.n, 1)
        cands = self.policy.sample_chunk(feat_n) # (n, chunk_len * action_dim)
        q = self.policy._agg(self.policy.critic(feat_n, cands)).squeeze(-1) # (n,)

        novelty = torch.zeros(self.n, device=self.device)
        if self.beta != 0.0 and self.novelty_mode == 'rnd':
            carry_h = {k: np.repeat(np.asarray(jax.device_get(v)), self.n, axis=0)
                       for k, v in self.dyn.items()}
            cands_np = cands.detach().cpu().numpy()
            # One rollout per candidate: RND scores the end feature
            # deterministically, so no draws and no std are needed.
            carry, _, _, _ = self.bridge.img_chunk(
                self.bridge.place_seed(carry_h), cands_np, self.chunk_len)
            novelty = self.rnd.score(
                jax_to_torch(self.bridge.get_feat(carry), self.device))
        elif self.beta != 0.0 and self.draws >= 2:
            carry_h = {k: np.repeat(np.asarray(jax.device_get(v)), self.n, axis=0)
                       for k, v in self.dyn.items()}
            cands_np = cands.detach().cpu().numpy()
            end_feats = []
            for _ in range(self.draws):
                carry, _, _, _ = self.bridge.img_chunk(
                    self.bridge.place_seed(carry_h), cands_np, self.chunk_len)
                end_feats.append(jax_to_torch(self.bridge.get_feat(carry),
                                              self.device))
            # Disagreement: std across stochastic draws, mean over feat dims.
            novelty = torch.stack(end_feats).std(dim=0, correction=0).mean(-1)

        nov_np = novelty.detach().cpu().numpy()
        prev_count = self._count
        # norm_freeze > 0: stop updating the Welford stats after that many
        # scored values. The frozen scale is what lets a decaying RND signal
        # actually shrink the bonus instead of being re-normalized to unit
        # size every step.
        if self.norm_freeze <= 0 or self._count < self.norm_freeze:
            self._norm_update(nov_np.tolist())
        if prev_count >= self.warmup and self.beta != 0.0:
            z = (novelty - self._mean) / self._norm_std()
            score = q + self.beta * z
        else:
            score = q

        idx = int(torch.argmax(score).item())
        idx_q = int(torch.argmax(q).item())
        self._acc('novelty_mean', float(nov_np.mean()))
        self._acc('novelty_picked_z',
                  float((nov_np[idx] - self._mean) / self._norm_std()))
        # THE attribution metrics for this method: how often the bonus changes
        # the pick at all, and how much exploit value (Q) each changed pick
        # pays for the exploration it buys. Both ~0 = the bonus is a no-op and
        # this run is critic best-of-N.
        self._acc('pick_changed', float(idx != idx_q))
        self._acc('q_paid', float((q[idx_q] - q[idx]).item()))
        self._acc('q_std', q.std().item())
        if self.rnd is not None:
            self._acc('rnd_loss', self.rnd.last_loss)

        return cands[idx].detach().cpu().numpy().reshape(
            self.chunk_len, self.action_dim)