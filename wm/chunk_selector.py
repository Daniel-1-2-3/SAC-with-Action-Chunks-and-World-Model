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
            beta * relu(d / d_data - 1), where d_data is the model's running
            mean disagreement on REAL replay transitions: a candidate whose
            imagined path is no more uncertain than the data gets nothing,
            one that is twice as uncertain gets beta. Both anneal as the
            ensemble converges. beta is in reward units. Training-time only:
            select(..., eval_mode=True) never adds it, so eval measures the
            learned policy, not the exploration policy. """
        assert score_mode in ('model', 'critic'), score_mode
        assert score_mode == 'critic' or model is not None
        assert bonus_beta == 0.0 or model is not None
        self.score_mode = score_mode
        self.rollout_chunks = max(1, int(rollout_chunks))
        self.bonus_beta = float(bonus_beta)
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
    def select(self, state_1d, eval_mode=False):
        """ state_1d: (obs_dim,) raw observation.
            Returns (chunk_len, action_dim). eval_mode disables the
            exploration bonus; scoring is otherwise identical. """
        if not self.enabled:
            return self.policy.act(np.asarray(state_1d, dtype=np.float32),
                                   eval_mode=False)

        feat = torch.as_tensor(np.asarray(state_1d, dtype=np.float32),
                               device=self.device).reshape(1, -1)
        feat_n = feat.repeat(self.n, 1)
        cands = self.policy.sample_chunk(feat_n)  # (n, chunk_len * action_dim)

        # Critic score of the SAME candidates -- QC's own best-of-N ranking.
        # In critic mode it picks; in model mode it is logged for attribution.
        qs = self.policy.critic(feat_n, cands)          # (ensemble, n, 1)
        critic_score = self.policy._agg(qs).squeeze(-1)  # (n,)

        if self.score_mode == 'critic':
            if self.bonus_beta > 0.0 and not eval_mode and self.model is not None:
                return self._select_uncertainty_scaled(feat_n, cands, qs, critic_score)
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

        use_bonus = self.bonus_beta > 0.0 and not eval_mode
        if use_bonus:
            # Novelty relative to the data: how much more uncertain the
            # imagined path is than a typical real transition. In-distribution
            # candidates get ~0, so the bonus does not add constant noise to
            # the ranking and vanishes as the ensemble converges.
            d = dis.squeeze(-1)
            ratio = d / max(self.model.data_disagreement, 1e-10)
            bonus = self.bonus_beta * torch.relu(ratio - 1.0)
            score = score + bonus

        # Every scalar below is computed on the device and moved to the CPU
        # in ONE transfer, instead of one blocking .item() per statistic.
        idx_t = torch.argmax(score)
        s_std = score.std(correction=0)
        c_std = critic_score.std(correction=0)
        sn = (score - score.mean()) / (s_std + 1e-12)
        cn = (critic_score - critic_score.mean()) / (c_std + 1e-12)
        keys = ['idx', 'term_r_std', 'term_q_std', 'term_r_absmean',
                'term_q_absmean', 'r_argmax', 'q_argmax', 'score_gap',
                'score_std', 'end_v_std', 'critic_argmax', 'corr',
                's_std', 'c_std']
        vals = [idx_t.float(), r_term.std(), q_term.std(), r_term.abs().mean(),
                q_term.abs().mean(), torch.argmax(r_term).float(),
                torch.argmax(q_term).float(), score[idx_t] - score.mean(),
                score.std(), end_v.squeeze(-1).std(),
                torch.argmax(critic_score).float(), (sn * cn).mean(),
                s_std, c_std]
        if use_bonus:
            keys += ['bonus_std', 'bonus_absmean', 'disagreement_raw',
                     'disagreement_ratio', 'bonus_argmax']
            vals += [bonus.std(), bonus.abs().mean(), d.mean(), ratio.mean(),
                     torch.argmax(bonus).float()]
        v = dict(zip(keys, torch.stack(vals).cpu().tolist()))
        idx = int(v['idx'])

        if use_bonus:
            self._acc('bonus_std', v['bonus_std'])
            self._acc('bonus_absmean', v['bonus_absmean'])
            self._acc('disagreement_raw', v['disagreement_raw'])
            self._acc('disagreement_ratio', v['disagreement_ratio'])
            # Would the bonus alone pick the same chunk? Near 1.0 means the
            # bonus, not the value, is driving exploration.
            self._acc('bonus_only_agree', float(int(v['bonus_argmax']) == idx))

        # TERM ATTRIBUTION. The ranking is decided by whichever term VARIES
        # across the n candidates, not by whichever is larger. If r_term_std
        # is ~0 while q_term_std is large, this arm is not "model reward
        # scoring" at all -- it is a latent value function, and the reward
        # head contributes nothing to the ordering. term_r_share is the
        # reward term's share of the total spread; near 0 means model reward
        # is irrelevant to the pick.
        r_std, q_std = v['term_r_std'], v['term_q_std']
        self._acc('term_r_std', r_std)
        self._acc('term_q_std', q_std)
        self._acc('term_r_share', r_std / (r_std + q_std + 1e-8))
        self._acc('term_r_absmean', v['term_r_absmean'])
        self._acc('term_q_absmean', v['term_q_absmean'])
        # Would ranking by imagined reward ALONE pick the same chunk as the
        # full score? Near 1.0 means the reward term drives the pick.
        self._acc('term_r_only_agree', float(int(v['r_argmax']) == idx))
        # Would ranking by the Q term ALONE pick the same chunk? Near 1.0
        # means the pick is entirely the latent value.
        self._acc('term_q_only_agree', float(int(v['q_argmax']) == idx))

        self._acc('score_gap', v['score_gap'])
        self._acc('score_std', v['score_std'])
        self._acc('end_v_std', v['end_v_std'])
        self._acc('pick_agreement', float(idx == int(v['critic_argmax'])))
        # Pearson over the n candidates. correction=0 (population std)
        # because the products are averaged over n -- the default sample std
        # would scale the result by (n-1)/n.
        if v['s_std'] > 1e-8 and v['c_std'] > 1e-8:
            self._acc('model_critic_corr', v['corr'])

        return cands[idx].detach().cpu().numpy().reshape(
            self.chunk_len, self.action_dim)

    def _select_uncertainty_scaled(self, feat_n, cands, qs, critic_score):
        """ EXPLORE arm. The critic ranks; novelty may only move the pick by
            as much as the critic is unsure of its own numbers:

                score_i = Q_i + beta * s_i * nu_i

            Q_i   critic's value of candidate i (ensemble mean)
            s_i   critic's uncertainty about candidate i: the std across its
                  ensemble heads, in Q units
            nu_i  model novelty of candidate i, dimensionless in [0, 1]:
                  clamp(disagreement / data_disagreement - 1, 0, 1), so 0 for
                  a path no more uncertain than the data, 1 for one at least
                  twice as uncertain

            Since nu_i <= 1, the bonus is bounded by beta * s_i: novelty can
            never outvote the critic by more than beta times the critic's own
            error bar on that candidate. As the critic converges its heads
            agree, s_i -> 0 and the arm becomes the control -- continuously,
            with no threshold, schedule or step counter. Where the critic
            still disagrees with itself (unseen states, the from-scratch
            regime) s_i is large and novelty decides. beta is dimensionless:
            beta = 1 means "one critic-std of novelty". """
        s_unc = qs.squeeze(-1).std(0, correction=0)  # (n,)
        z = self.model.encode(feat_n)
        chunk_actions = cands.reshape(self.n, self.chunk_len, self.action_dim)
        z, _, disc, dis = self.model.rollout_chunk(z, chunk_actions)
        for _ in range(self.rollout_chunks - 1):
            z, _, disc, dis_j = self.model.rollout_pi_chunk(z, self.chunk_len, disc)
            dis = dis + dis_j
        dis = dis / self.rollout_chunks
        d = dis.squeeze(-1)
        ratio = d / max(self.model.data_disagreement, 1e-10)
        nu = torch.clamp(ratio - 1.0, 0.0, 1.0)
        bonus = self.bonus_beta * s_unc * nu
        score = critic_score + bonus

        idx_t = torch.argmax(score)
        crit_t = torch.argmax(critic_score)
        unc_mean = s_unc.mean()
        gap = critic_score[crit_t] - critic_score.mean()
        # IS THE CRITIC SURE? Number of candidates the critic cannot tell
        # apart from its favourite, i.e. within one error bar of the max.
        # 1/n = sure (only the argmax survives); 1.0 = the critic has no
        # opinion at all. This is the gauge of the anneal, independent of
        # whether the model found anything novel.
        within = (critic_score >= critic_score[crit_t] - unc_mean).float().mean()
        sc = s_unc - s_unc.mean()
        nc = nu - nu.mean()
        unc_nov_corr = (sc * nc).mean() / (sc.std(correction=0) * nc.std(correction=0) + 1e-8)
        v = torch.stack([
            idx_t.float(), crit_t.float(),
            unc_mean, gap, critic_score.std(),
            bonus.abs().mean(), bonus.max(),
            nu.mean(), ratio.mean(), torch.argmax(nu).float(),
            within, s_unc[idx_t], nu[idx_t],
            (nu > 0).float().mean(), (nu >= 1).float().mean(),
            unc_nov_corr,
        ]).cpu().tolist()
        idx, crit_idx = int(v[0]), int(v[1])
        # -- critic side: is it sure, and how sure relative to its margin
        self._acc('critic_unc', v[2])           # mean s_i, Q units
        self._acc('score_gap', v[3])            # critic's own margin
        self._acc('score_std', v[4])
        self._acc('unc_over_gap', v[2] / (abs(v[3]) + 1e-8))
        self._acc('frac_within_unc', v[10])     # -> 1/n when sure
        # -- bonus side: what it could buy and what it did
        self._acc('bonus_absmean', v[5])
        self._acc('bonus_max', v[6])
        # bonus_max / score_gap: how much of the critic's margin novelty
        # could have bought. Falls toward 0 as the critic converges.
        self._acc('bonus_over_gap', v[6] / (abs(v[3]) + 1e-8))
        self._acc('pick_changed', float(idx != crit_idx))
        self._acc('bonus_only_agree', float(int(v[9]) == idx))
        self._acc('picked_unc', v[11])          # s_i of the executed chunk
        self._acc('picked_novelty', v[12])      # nu_i of the executed chunk
        # -- novelty side: alive (some nu > 0), or saturated (nu == 1)
        self._acc('novelty_mean', v[7])
        self._acc('disagreement_ratio', v[8])
        self._acc('novelty_frac_active', v[13])
        self._acc('novelty_frac_saturated', v[14])
        # Do the critic's doubt and the model's novelty point at the same
        # candidates? Positive = the bonus fires where the critic is unsure.
        self._acc('unc_novelty_corr', v[15])
        return cands[idx].detach().cpu().numpy().reshape(
            self.chunk_len, self.action_dim)