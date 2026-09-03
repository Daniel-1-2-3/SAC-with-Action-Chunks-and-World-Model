import numpy as np
import torch


class ChunkSelector:
    """ Best-of-N chunk selection at act time.

        At every chunk boundary: sample n candidate chunks from the QC-FQL
        one-step policy at the raw observation and execute the argmax of
        Q(s, chunk) under the online QC critic (mean over the ensemble). This
        is QC's own best-of-N and needs no model at all: it is the control
        arm (train_qcfql_bon.py).

        The explore arm adds a bounded novelty bonus on top of the critic's
        score, read from a TD-MPC2 dynamics ensemble; see
        _select_uncertainty_scaled. That is the ONLY place the latent model
        influences behavior.

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
        stateless apart from the explore arm's learning-progress gate. """

    def __init__(self, model, policy, action_dim, chunk_len, n, gamma, device,
                 rollout_chunks=1, bonus_beta=0.0,
                 novelty='model', novelty_at='path', nu_cap=1.0,
                 bonus_scale='spread', progress_gate=True, progress_window=20,
                 progress_tau=0.2, use_rel_unc=True, controller='gate',
                 bandit_window=20, bandit_c=1.0):
        """ model: a TDMPC2Model (explore arm), or None (control).
            rollout_chunks (explore arm): imagine this many chunks ahead when
            measuring novelty. The first is the candidate; each further
            chunk's actions come from the model's policy prior at the
            imagined latent. Lookahead depth is rollout_chunks * chunk_len
            steps and costs nothing but dynamics steps -- no decode.
            bonus_beta (explore arm): weight of the novelty bonus. 0 disables
            it and gives exactly the control. Training-time only:
            select(..., eval_mode=True) never adds it, so eval measures the
            learned policy, not the exploration policy. """
        assert bonus_beta == 0.0 or model is not None
        self.rollout_chunks = max(1, int(rollout_chunks))
        self.bonus_beta = float(bonus_beta)
        # explore arm only:
        #   novelty     'model'  nu from dynamics-ensemble disagreement
        #               'none'   nu = 1 for every candidate -- the critic's
        #                        doubt explores alone, no model in the pick
        #                        (attribution arm)
        #   novelty_at  'path'   disagreement averaged over the imagined steps
        #               'end'    disagreement at the final imagined latent
        #                        only (outcome, not motion)
        #   nu_cap      clamp ceiling on nu; bonus <= beta * s_i * nu_cap
        #   bonus_scale 'unc'    v3: bonus = beta * s_i * nu_i (g ignored)
        #               'spread' v4: bonus = beta * g * sigma_Q * s~_i * nu_i
        #   progress_gate / progress_window / progress_tau: the learning-
        #               progress gate g, see report_episode_return.
        #   use_rel_unc (spread mode) multiply by the critic's relative
        #               doubt s~_i; False drops that factor, so the bonus is
        #               beta * g * sigma_Q * nu_i.
        #   controller  'gate'    g is the learning-progress gate above
        #               'bandit'  g is 0 or 1 for a whole episode, chosen at
        #                         begin_episode() by a sliding-window UCB
        #                         over the two arms' real episode returns
        #   bandit_window / bandit_c: the window (episodes) and the UCB
        #               exploration constant.
        assert novelty in ('model', 'none'), novelty
        assert novelty_at in ('path', 'end'), novelty_at
        assert bonus_scale in ('unc', 'spread'), bonus_scale
        self.novelty = novelty
        self.novelty_at = novelty_at
        self.nu_cap = float(nu_cap)
        self.bonus_scale = bonus_scale
        self.progress_gate = bool(progress_gate)
        self.progress_window = int(progress_window)
        self.progress_tau = float(progress_tau)
        self.use_rel_unc = bool(use_rel_unc)
        assert controller in ('gate', 'bandit'), controller
        self.controller = controller
        self.bandit_window = int(bandit_window)
        self.bandit_c = float(bandit_c)
        self._pulls = []            # (arm, return) of the last finished episodes
        self.bandit_arm = 1         # arm in force: 0 exploit (g=0), 1 explore (g=1)
        self._returns = []          # real ONLINE episode returns, in order
        self._g = 1.0               # gate value in use (EMA-smoothed)
        self._g_raw = 1.0
        self._frac = 0.5
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

    def report_episode_return(self, ep_return):
        """ Learning-progress gate. Call once per finished REAL online
            episode (never for eval episodes). With W = progress_window:

                recent = last W returns, older = the W before those
                frac   = fraction of recent strictly above median(older)
                g      = 1 - 2 * max(frac - 0.5, 0)

            Flat or plateaued returns give frac ~ 0.5 -> g = 1 (explore).
            Improving returns give frac -> 1 -> g -> 0 (become the control).
            Fewer than 2W episodes -> g = 1. g is EMA-smoothed with
            progress_tau. With progress_gate off, g is identically 1. """
        self._returns.append(float(ep_return))
        if self.controller == 'bandit':
            self._pulls.append((self.bandit_arm, float(ep_return)))
            return
        if not self.progress_gate:
            return
        W = self.progress_window
        if len(self._returns) < 2 * W:
            return
        recent = np.asarray(self._returns[-W:])
        older = np.asarray(self._returns[-2 * W:-W])
        frac = float((recent > np.median(older)).mean())
        g = 1.0 - 2.0 * max(frac - 0.5, 0.0)
        self._frac = frac
        self._g_raw = g
        self._g = (1.0 - self.progress_tau) * self._g + self.progress_tau * g

    def begin_episode(self):
        """ Called by the loop at the start of every REAL online episode
            (never for eval). Bandit controller only: pick this episode's
            arm by sliding-window UCB over the last bandit_window finished
            episodes,

                score_k = mean_k / range + c * sqrt(2 ln N / n_k)

            mean_k the mean return of the episodes that ran arm k in the
            window, range = max - min of all returns in the window (so c is
            dimensionless), N the window size in use, n_k arm k's pulls in
            it. An arm with no pulls in the window is taken first (exploit,
            then explore). Arm 0 = exploit, g = 0, exactly the control for
            that episode; arm 1 = explore, g = 1, the full bonus.

            Every call logs five select/bandit_* keys: the arm chosen,
            the window size, the two window means (raw return units, only
            when the arm has pulls) and the UCB score gap explore - exploit
            (only when both have). """
        if self.controller != 'bandit':
            return
        window = self._pulls[-self.bandit_window:]
        n_total = len(window)
        means, counts = {}, {}
        for k in (0, 1):
            rs = [r for a, r in window if a == k]
            counts[k] = len(rs)
            if rs:
                means[k] = float(np.mean(rs))
        untried = [k for k in (0, 1) if counts[k] == 0]
        gap = None
        if untried:
            arm = untried[0]
        else:
            rets = np.asarray([r for _, r in window])
            rng_ = max(float(rets.max() - rets.min()), 1e-8)
            score = {k: means[k] / rng_ + self.bandit_c * np.sqrt(2.0 * np.log(n_total) / counts[k])
                     for k in (0, 1)}
            arm = 1 if score[1] >= score[0] else 0
            gap = score[1] - score[0]
        self.bandit_arm = int(arm)
        self._acc('bandit_arm', float(arm))
        self._acc('bandit_n_window', float(n_total))
        if 0 in means:
            self._acc('bandit_mean_exploit', means[0])
        if 1 in means:
            self._acc('bandit_mean_explore', means[1])
        if gap is not None:
            self._acc('bandit_ucb_gap', float(gap))

    @property
    def gate(self):
        if self.controller == 'bandit':
            return float(self.bandit_arm)
        return self._g if self.progress_gate else 1.0

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

        # QC's own best-of-N: the online critic scores every candidate.
        qs = self.policy.critic(feat_n, cands)          # (ensemble, n, 1)
        critic_score = self.policy._agg(qs).squeeze(-1)  # (n,)

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

    def _select_uncertainty_scaled(self, feat_n, cands, qs, critic_score):
        """ EXPLORE arm. The critic orders the candidates; novelty may only
            move the pick by a bounded amount.

            bonus_scale 'spread' (v4):

                score_i = Q_i + beta * g * sigma_Q * s~_i * nu_i

            sigma_Q  population std of Q_i across the n candidates (Q units)
            s~_i     relative critic doubt clamp(s_i / mean_j s_j, 0, 1), or
                     1 with use_rel_unc off (a 2-head ensemble's spread is
                     mostly noise, and it halved the bonus at random)
            g        learning-progress gate in [0, 1], see
                     report_episode_return; 1 while returns are flat, -> 0
                     while they improve (the arm becomes the control)
            nu_i     model novelty in [0, nu_cap] as below

            so bonus_i <= beta * g * sigma_Q * nu_cap: novelty can reorder
            candidates the critic rates within ~beta spreads of each other and
            cannot override a clear critic preference.

            bonus_scale 'unc' (v3, kept for reproducibility):

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
        if self.novelty == 'none':
            d = torch.ones_like(s_unc)
            ratio = torch.full_like(s_unc, 2.0)
            nu = torch.ones_like(s_unc)
        else:
            d = self._candidate_disagreement(feat_n, cands)  # (n,)
            ratio = d / max(self.model.data_disagreement, 1e-10)
            nu = torch.clamp(ratio - 1.0, 0.0, self.nu_cap)
        sigma_q = critic_score.std(correction=0)
        if self.bonus_scale == 'unc':
            scale = self.bonus_beta * s_unc
        else:
            g = self.gate
            if self.use_rel_unc:
                s_rel = torch.clamp(s_unc / (s_unc.mean() + 1e-12), 0.0, 1.0)
            else:
                s_rel = torch.ones_like(s_unc)
            scale = self.bonus_beta * g * sigma_q * s_rel
        bonus = scale * nu
        score = critic_score + bonus

        idx_t = torch.argmax(score)
        crit_t = torch.argmax(critic_score)
        # Counterfactual: the pick with the model removed (nu = 1). Differs
        # from idx_t exactly when the model, not the critic's doubt, decided.
        nomodel_t = torch.argmax(critic_score + scale)
        active = (nu > 0).float().mean()
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
            active, (nu >= self.nu_cap).float().mean(),
            unc_nov_corr, nomodel_t.float(),
            ((active > 0) & (active < 1)).float(),
            sigma_q,
        ]).cpu().tolist()
        idx, crit_idx = int(v[0]), int(v[1])
        if self.bonus_scale == 'spread':
            self._acc('progress_g', self.gate)
            self._acc('progress_frac', self._frac)
            self._acc('sigma_q', v[18])
            # bonus_max / sigma_Q: must stay <= beta * nu_cap by construction
            self._acc('bonus_over_sigma', v[6] / (abs(v[18]) + 1e-8))
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
        # -- novelty side: alive (some nu > 0), or saturated (nu == cap)
        self._acc('novelty_mean', v[7])
        self._acc('disagreement_ratio', v[8])
        self._acc('novelty_frac_active', v[13])
        self._acc('novelty_frac_saturated', v[14])
        # Do the critic's doubt and the model's novelty point at the same
        # candidates? Positive = the bonus fires where the critic is unsure.
        self._acc('unc_novelty_corr', v[15])
        # Did the MODEL change the pick, relative to critic doubt alone?
        self._acc('model_changed', float(idx != int(v[16])))
        # Per decision: did the model separate siblings at this state
        # (some novel, some not), rather than flag the whole state?
        self._acc('novelty_mixed', v[17])
        return cands[idx].detach().cpu().numpy().reshape(
            self.chunk_len, self.action_dim)

    @torch.no_grad()
    def _candidate_disagreement(self, feat_n, cands):
        """ Novelty measure of each candidate's imagined path, (n,). The same
            function measures the data reference (tdmpc.ref_mode=rollout). """
        m = self.model
        acts = cands.reshape(self.n, self.chunk_len, self.action_dim)
        return m.path_disagreement(m.encode(feat_n), acts)
