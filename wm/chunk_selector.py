import numpy as np
import torch


class ChunkSelector:
    """ Best-of-N chunk selection. The ONLY place the latent model influences
        behavior.

        At every chunk boundary: sample n candidate chunks from the QC-FQL
        one-step policy at the raw observation, then rank them. What does the
        ranking is the experiment:

          score_mode 'model'   TD-MPC2 latent model. Encode the observation
                               once, imagine each candidate chunk in latent
                               space, score it as

                                 sum_t gamma^t * r_model(z_t, a_t)
                                   + gamma^H * Q_model(z_H, pi(z_H))

                               Nothing decodes back to observation space at
                               any point -- the reward head and the Q both
                               read the latent directly. Both terms are
                               symlog two-hot predictions, so they are in the
                               same units as each other and as the QC critic.

          score_mode 'critic'  Q(s, chunk) from the online QC critic. This is
                               the QC paper's own best-of-N and needs no model
                               at all. It is the control arm.

        Training never sees any of this. The critic and actor update on real
        replay chunks with the plain QC-FQL target, so the model's entire
        effect on the run is through WHICH chunks get executed -- i.e. through
        the data that selection collects.

        Failure containment: candidates are i.i.d. samples from the policy, so
        if the scores are uninformative noise the argmax is distributed like a
        single policy sample and the agent degrades to QC-FQL rather than
        below it. The exception is a model that SYSTEMATICALLY prefers chunks
        it predicts well (do-nothing chunks are the usual case); watch
        select/score_gap together with eval/coherence for that.

        n <= 1 disables everything here -- select() is exactly policy.act.
        That is the QC-FQL control run.

        The TD-MPC2 encoder is Markov, so there is no posterior to keep
        filtered between steps and no per-step encode. The selector is
        stateless.

        Attribution: in model mode every decision ALSO scores the same
        candidates with the online critic alone. select/pick_agreement near
        1.0 means the model adds nothing beyond the critic and this run
        reduces to QC. """

    def __init__(self, model, policy, action_dim, chunk_len, n, gamma, device,
                 score_mode='model', rollout_chunks=1, bonus_beta=0.0):
        """ model: a TDMPC2Model, or None in critic mode.
            rollout_chunks (model mode): imagine this many chunks ahead. The
            first is the candidate; each further chunk's actions come from the
            model's policy prior at the imagined latent. Lookahead depth is
            rollout_chunks * chunk_len steps and costs nothing but dynamics
            steps -- no decode, so depth does not compound decoder error the
            way it did with the previous scorer.
            bonus_beta (model mode): weight of the dynamics-ensemble
            disagreement bonus (explore arm). 0 disables it. The bonus is
            divided by a running mean of disagreement so beta is in reward
            units. """
        assert score_mode in ('model', 'critic'), score_mode
        assert score_mode == 'critic' or model is not None
        self.score_mode = score_mode
        self.rollout_chunks = max(1, int(rollout_chunks))
        self.bonus_beta = float(bonus_beta)
        self._dis_scale = None
        self.model = model
        self.policy = policy
        self.action_dim = action_dim
        self.chunk_len = chunk_len
        self.n = int(n)
        self.enabled = self.n > 1
        self.gamma = gamma
        self.device = device
        self._stats = {}

    def _acc(self, key, value):
        s, c = self._stats.get(key, (0.0, 0))
        self._stats[key] = (s + float(value), c + 1)

    def pop_stats(self):
        """ Means since the last pop, prefixed select/. Empty when disabled or
            no decisions happened. """
        out = {f'select/{k}': s / c for k, (s, c) in self._stats.items() if c > 0}
        self._stats = {}
        return out

    @torch.no_grad()
    def select(self, state_1d):
        """ state_1d: (obs_dim,) raw observation.
            Returns (chunk_len, action_dim). """
        if not self.enabled:
            return self.policy.act(np.asarray(state_1d, dtype=np.float32),
                                   eval_mode=False)

        feat = torch.as_tensor(np.asarray(state_1d, dtype=np.float32),
                               device=self.device).reshape(1, -1)
        feat_n = feat.repeat(self.n, 1)
        cands = self.policy.sample_chunk(feat_n)  # (n, chunk_len * action_dim)

        # Critic score of the SAME candidates -- QC's own best-of-N ranking.
        # In critic mode it picks; in model mode it is logged for attribution.
        critic_score = self.policy._agg(
            self.policy.critic(feat_n, cands)).squeeze(-1)  # (n,)

        if self.score_mode == 'critic':
            idx = int(torch.argmax(critic_score).item())
            self._acc('score_gap',
                      (critic_score[idx] - critic_score.mean()).item())
            self._acc('score_std', critic_score.std().item())
            return cands[idx].detach().cpu().numpy().reshape(
                self.chunk_len, self.action_dim)

        z = self.model.encode(feat_n)
        chunk_actions = cands.reshape(self.n, self.chunk_len, self.action_dim)
        z, r_term, disc, dis = self.model.rollout_chunk(z, chunk_actions)
        for _ in range(self.rollout_chunks - 1):
            z, r_j, disc, dis_j = self.model.rollout_pi_chunk(z, self.chunk_len, disc)
            r_term = r_term + r_j
            dis = dis + dis_j
        dis = dis / self.rollout_chunks

        end_v = self.model.terminal_value(z)
        q_term = (disc * end_v).squeeze(-1)  # (n,)
        r_term = r_term.squeeze(-1)          # (n,) pooled imagined reward
        score = r_term + q_term

        if self.bonus_beta > 0.0:
            # Disagreement bonus in reward units: beta per typical
            # disagreement, with "typical" a running mean over decisions.
            d = dis.squeeze(-1)
            batch_mean = float(d.mean().item())
            if self._dis_scale is None:
                self._dis_scale = max(batch_mean, 1e-8)
            else:
                self._dis_scale = 0.99 * self._dis_scale + 0.01 * batch_mean
            bonus = self.bonus_beta * d / max(self._dis_scale, 1e-8)
            score = score + bonus
            self._acc('bonus_std', bonus.std().item())
            self._acc('bonus_absmean', bonus.abs().mean().item())
            self._acc('disagreement_raw', batch_mean)
            # Would the bonus alone pick the same chunk? Near 1.0 means the
            # bonus, not the value, is driving exploration.
            self._acc('bonus_only_agree',
                      float(int(torch.argmax(bonus).item())
                            == int(torch.argmax(score).item())))

        # TERM ATTRIBUTION. The ranking is decided by whichever term VARIES
        # across the n candidates, not by whichever is larger. If r_term_std
        # is ~0 while q_term_std is large, this arm is not "model reward
        # scoring" at all -- it is a latent value function, and the reward
        # head contributes nothing to the ordering. term_r_share is the
        # reward term's share of the total spread; near 0 means model reward
        # is irrelevant to the pick.
        r_std = float(r_term.std().item())
        q_std = float(q_term.std().item())
        self._acc('term_r_std', r_std)
        self._acc('term_q_std', q_std)
        self._acc('term_r_share', r_std / (r_std + q_std + 1e-8))
        self._acc('term_r_absmean', float(r_term.abs().mean().item()))
        self._acc('term_q_absmean', float(q_term.abs().mean().item()))
        # Would ranking by imagined reward ALONE pick the same chunk as the
        # full score? Near 1.0 means the reward term drives the pick.
        self._acc('term_r_only_agree',
                  float(int(torch.argmax(r_term).item())
                        == int(torch.argmax(score).item())))
        # Would ranking by the Q term ALONE pick the same chunk? Near 1.0
        # means the pick is entirely the latent value.
        self._acc('term_q_only_agree',
                  float(int(torch.argmax(q_term).item())
                        == int(torch.argmax(score).item())))

        idx = int(torch.argmax(score).item())
        self._acc('score_gap', (score[idx] - score.mean()).item())
        self._acc('score_std', score.std().item())
        self._acc('end_v_std', end_v.squeeze(-1).std().item())
        self._acc('pick_agreement',
                  float(idx == int(torch.argmax(critic_score).item())))
        # Pearson over the n candidates. correction=0 (population std)
        # because the products are averaged over n -- the default sample std
        # would scale the result by (n-1)/n.
        s_std = score.std(correction=0)
        c_std = critic_score.std(correction=0)
        if s_std.item() > 1e-8 and c_std.item() > 1e-8:
            sn = (score - score.mean()) / s_std
            cn = (critic_score - critic_score.mean()) / c_std
            self._acc('model_critic_corr', (sn * cn).mean().item())

        return cands[idx].detach().cpu().numpy().reshape(
            self.chunk_len, self.action_dim)
