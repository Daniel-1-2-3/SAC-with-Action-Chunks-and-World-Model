import numpy as np
import torch


class ChunkSelector:
    """ Best-of-N chunk selection at act time.

        At every chunk boundary: sample n candidate chunks from the QC-FQL
        one-step policy at the raw observation and execute the argmax of
        Q(s, chunk) under the online QC critic (mean over the ensemble). This
        is QC's own best-of-N and needs no model at all: it is the combined
        arm (train_combined.py).

        Training never sees any of this. The critic and actor update on real
        replay chunks with the plain QC-FQL target, so selection's entire
        effect on the run is through WHICH chunks get executed -- i.e. through
        the data that selection collects.

        Failure containment: candidates are i.i.d. samples from the policy, so
        if the scores are uninformative noise the argmax is distributed like a
        single policy sample and the agent degrades to QC-FQL rather than
        below it.

        n <= 1 disables everything here -- select() is exactly policy.act.
        That is the plain QC-FQL run.

        The TD-MPC2 encoder is Markov, so there is no posterior to keep
        filtered between steps and no per-step encode. The selector is
        stateless. """

    def __init__(self, model, policy, action_dim, chunk_len, n, gamma, device,
                 candidate_source='actor'):
        """ model: a TDMPC2Model (wm_combined arm, for the episode-start
            value capture only), or None (combined arm). It never influences
            which chunk is executed.
            candidate_source  'actor'  candidates from the one-step actor
                                       (policy.sample_chunk)
                              'bc'     candidates from the flow BC policy
                                       (policy.compute_flow_actions on
                                       policy.noise(n): Euler, flow_steps,
                                       clipped). Scoring is unchanged. """
        assert candidate_source in ('actor', 'bc'), candidate_source
        self.candidate_source = candidate_source
        self.model = model
        self.policy = policy
        self.action_dim = action_dim
        self.chunk_len = chunk_len
        self.n = int(n)
        self.enabled = self.n > 1
        self.gamma = gamma
        self.device = device
        self._stats = {}
        self._ep_first_pending = False
        self._ep_first_vals = None

    def _acc(self, key, value):
        s, c = self._stats.get(key, (0.0, 0))
        self._stats[key] = (s + float(value), c + 1)

    def begin_episode(self):
        """ Called by the loop at the start of every REAL online episode
            (never for eval). Arms the episode-start value capture: the next
            select() records the critic's (and, if a model is attached, the
            wm's) value estimate of the start state, which
            report_episode_discounted later pairs with the realized
            discounted return -- ground-truth calibration of both value
            functions at episode-start granularity. """
        self._ep_first_pending = True
        self._ep_first_vals = None

    def report_episode_discounted(self, disc_return):
        """ Realized discounted return of the episode whose start-state
            values _ep_first_vals captured. Emits, per finished episode:
              select/critic_value_first, select/critic_optimism
              select/wm_value_first,     select/wm_optimism   (model arms)
            optimism = estimate - realized: persistent positive means the
            value function promises more than episodes deliver. """
        vals = getattr(self, '_ep_first_vals', None)
        if vals is None:
            return
        critic_v, wm_v = vals
        self._ep_first_vals = None
        self._acc('critic_value_first', critic_v)
        self._acc('critic_optimism', critic_v - float(disc_return))
        self._acc('realized_disc_return', float(disc_return))
        if wm_v == wm_v:                       # not NaN
            self._acc('wm_value_first', wm_v)
            self._acc('wm_optimism', wm_v - float(disc_return))

    def pop_stats(self):
        """ Means since the last pop, prefixed select/. Empty when disabled or
            no decisions happened. """
        out = {f'select/{k}': s / c for k, (s, c) in self._stats.items() if c > 0}
        self._stats = {}
        return out

    @torch.no_grad()
    def select(self, state_1d, eval_mode=False):
        """ state_1d: (obs_dim,) raw observation.
            Returns (chunk_len, action_dim). eval_mode only skips the
            training-time bookkeeping; scoring is identical. """
        if not self.enabled:
            return self.policy.act(np.asarray(state_1d, dtype=np.float32),
                                   eval_mode=False)

        feat = torch.as_tensor(np.asarray(state_1d, dtype=np.float32),
                               device=self.device).reshape(1, -1)
        feat_n = feat.repeat(self.n, 1)
        if self.candidate_source == 'bc':
            noises = self.policy.noise(self.n)
            cands = self.policy.compute_flow_actions(feat_n, noises)  # (n, chunk_len * action_dim)
        else:
            cands = self.policy.sample_chunk(feat_n)  # (n, chunk_len * action_dim)

        # QC's own best-of-N: the online critic scores every candidate.
        qs = self.policy.critic(feat_n, cands)          # (ensemble, n, 1)
        critic_score = self.policy._agg(qs).squeeze(-1)  # (n,)

        if getattr(self, '_ep_first_pending', False) and not eval_mode:
            # Episode-start value estimates, paired later with the realized
            # discounted return by report_episode_discounted.
            self._ep_first_pending = False
            wm_v = float('nan')
            if self.model is not None:
                try:
                    z1 = self.model.encode(feat)
                    if getattr(self.model, 'q_mode', 'step') == 'chunk':
                        wm_v = float(self.model.chunk_q(
                            z1, cands[:1], target=True).squeeze())
                    else:
                        wm_v = float(self.model.net.q_subset(
                            z1, self.model.net.pi(z1)[1], reduce='avg').squeeze())
                except Exception:
                    pass
            self._ep_first_vals = (float(critic_score.max()), wm_v)

        idx_t = torch.argmax(critic_score)
        # One GPU->CPU transfer for everything this decision needs.
        stats = torch.stack([
            idx_t.float(),
            critic_score[idx_t] - critic_score.mean(),
            critic_score.std(),
        ]).cpu().tolist()
        idx = int(stats[0])
        if not eval_mode:
            self._acc('score_gap', stats[1])
            self._acc('score_std', stats[2])
        return cands[idx].detach().cpu().numpy().reshape(
            self.chunk_len, self.action_dim)
