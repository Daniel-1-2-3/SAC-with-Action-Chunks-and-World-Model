""" Tree selection: the combined method's critic best-of-N, plus a
    world-model tree of imagined futures that can add a bounded
    "unvisited but trusted" bonus to each root candidate during online
    collection.

    At an exploring decision the root set is the 16 actor candidates, a
    few flow behaviour-cloning candidates and the carried plan chunk from
    the previous decision. Each root chunk is imagined one chunk ahead;
    deeper levels grow from the best trusted nodes with chunks from the
    proposer. Every imagined end point is scored by how far it is from
    anywhere actually visited (the visit memory), discounted by depth, and
    zeroed where the dynamics ensemble disagrees more than trust_cap times
    the real-data reference at that depth. Each root candidate's reach is
    the best such score among its descendants, and

        bonus_i = amount * spread of critic values * clamp(reach_i - 1, 0, reach_cap)
        executed = argmax(critic score_i + bonus_i)

    Training, targets, replay and eval are the combined method's. Eval and
    decisions with amount < tree.min_amount take the plain path, which
    draws and scores candidates exactly as the combined arm does and
    consumes exactly the same random numbers.

    TreeSelector sets the explore amount per EPISODE (the bandit's arm
    times tree.beta); TreeSwitchSelector (arms/tree_switch.py uses it)
    sets it per DECISION from the critic's own stuck score. """

import time

import numpy as np
import torch

from wm.chunk_selector import ChunkSelector


class TreeSelector(ChunkSelector):

    def __init__(self, model, policy, action_dim, chunk_len, n, gamma, device,
                 candidate_source='actor', tree=None, proposer=None,
                 memory=None, bandit=None, seed=0):
        """ tree: the config's tree block. proposer: a ChunkProposer.
            memory: a VisitMemory. bandit: an EpisodeBandit, or None when
            tree.bandit is off (every episode explores). seed: the run
            seed; the selector's own generator is seeded from seed + 1 so
            act-time tree sampling doesn't consume the global RNG. """
        super().__init__(model, policy, action_dim, chunk_len, n, gamma,
                         device, candidate_source=candidate_source)
        assert model is not None and tree is not None
        self.tree = tree
        self.proposer = proposer
        self.memory = memory
        self.bandit = bandit
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(int(seed) + 1)
        # Trust reference per depth d = 1..tree.depth (None until the first
        # measurement; a depth with no reference yet trusts nothing, so the
        # tree stays silent rather than trusting an unmeasured model).
        self.reference = [None] * int(tree.depth)
        self.carried = None            # planned next chunk (chunk_dim,) or None
        self._env_steps = 0
        self._ep_reward_sum = 0.0
        self._last_refresh_step = None
        self._tree_count = 0

    # --------------------------------------------------------------- hooks

    def begin_episode(self):
        """ Start of a REAL online episode (also called after an eval
            interruption): arm the value capture, clear the carried plan,
            discard any partial episode-return sum without reporting it,
            reset per-episode state, and let the bandit pick this
            episode's arm. """
        super().begin_episode()
        self.carried = None
        self._ep_reward_sum = 0.0
        self._reset_episode_state()
        if self.bandit is not None:
            self.bandit.choose(self._acc)

    def observe_step(self, reward, done):
        """ Called by the loop after every real online env step. Sums the
            episode return for the bandit; done reports it and resets. """
        self._env_steps += 1
        self._ep_reward_sum += float(reward)
        self._observe_reward(float(reward))
        if done:
            if self.bandit is not None:
                self.bandit.report(self._ep_reward_sum)
            self._ep_reward_sum = 0.0

    def _reset_episode_state(self):
        pass

    def _observe_reward(self, reward):
        pass

    def explore_amount(self, critic_score_actor):
        """ The episode-level amount: tree.beta in an exploring episode
            (bandit arm 1, or no bandit), 0 otherwise. """
        arm = 1 if self.bandit is None else self.bandit.arm
        return float(self.tree.beta) * float(arm)

    # ------------------------------------------------------ trust reference

    @torch.no_grad()
    def update_trust_reference(self, replay, rng, online_frac):
        """ Measure, per depth, the dynamics-ensemble disagreement the model
            shows on REAL action sequences, so imagined disagreement at
            depth d is compared like with like. Also measures drift: the
            distance between the imagined latent at the end of chunk d and
            the encoding of the real observation there. Returns a metric
            dictionary (plain keys; the caller's prefixing makes them
            wm/...). Skipped when fewer than 32 windows stay inside one
            episode. """
        depth = int(self.tree.depth)
        span = depth * self.chunk_len
        w = replay.sample_model_windows(256, span, self.device, rng,
                                        online_frac=online_frac)
        if w is None:
            return {}
        keep = (w['valid'][:, :, 0].min(dim=1).values > 0.5).nonzero().squeeze(-1)
        if len(keep) < 32:
            return {}
        obs, next_obs, act = w['obs'][keep], w['next_obs'][keep], w['action'][keep]
        z = self.model.encode(obs[:, 0])
        rate = float(self.tree.ref_rate)
        metrics = {}
        for d in range(1, depth + 1):
            lo = (d - 1) * self.chunk_len
            z, dis = self.model.imagine_chunk(z, act[:, lo:lo + self.chunk_len])
            z_real = self.model.encode(next_obs[:, d * self.chunk_len - 1])
            drift = (z - z_real).pow(2).sum(-1).sqrt().mean().item()
            disagreement = dis.mean().item()
            old = self.reference[d - 1]
            self.reference[d - 1] = disagreement if old is None else \
                (1.0 - rate) * old + rate * disagreement
            metrics[f'tree_reference_d{d}'] = self.reference[d - 1]
            typical = self.memory.typical_distance if self.memory is not None else None
            if typical:
                metrics[f'tree_drift_ratio_d{d}'] = drift / typical
            else:
                metrics[f'tree_drift_d{d}'] = drift
        return metrics

    # -------------------------------------------------------------- select

    @torch.no_grad()
    def select(self, state_1d, eval_mode=False):
        """ state_1d: (obs_dim,) raw observation.
            Returns (chunk_len, action_dim). """
        if not self.enabled:
            return self.policy.act(np.asarray(state_1d, dtype=np.float32),
                                   eval_mode=False)

        feat = torch.as_tensor(np.asarray(state_1d, dtype=np.float32),
                               device=self.device).reshape(1, -1)
        # Always the combined arm's draw and scoring first, so the plain
        # path consumes exactly the same random numbers it would.
        cands, qs, critic_score = self.score_candidates(feat)
        self._value_capture(feat, cands, critic_score, eval_mode)

        amount = 0.0 if eval_mode else self.explore_amount(critic_score)
        if not eval_mode:
            self._acc('explore_amount', amount)
            self._maybe_refresh_memory()

        if eval_mode or amount < float(self.tree.min_amount):
            if eval_mode:
                self._acc('eval_tree_grown', 0.0)
            else:
                self._acc('tree_grown_frac', 0.0)
                self.carried = None
                self.memory.add_online(np.asarray(state_1d, dtype=np.float32),
                                       self.model.encode(feat)[0])
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

        return self._select_tree(state_1d, feat, cands, critic_score, amount)

    def _value_capture(self, feat, cands, critic_score, eval_mode):
        """ The parent's episode-start value capture, with the global torch
            RNG saved and restored around the model's value estimate (pi and
            q_subset draw random numbers), so the plain path's RNG stream
            stays identical to the combined arm's. """
        if not getattr(self, '_ep_first_pending', False) or eval_mode:
            return
        self._ep_first_pending = False
        wm_v = float('nan')
        if self.model is not None:
            rng_state = torch.get_rng_state()
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
            torch.set_rng_state(rng_state)
        self._ep_first_vals = (float(critic_score.max()), wm_v)

    def _maybe_refresh_memory(self):
        due = (self._last_refresh_step is None or
               self._env_steps - self._last_refresh_step >=
               int(self.tree.memory_refresh_every))
        if due and self.memory is not None:
            self.memory.refresh(self.model.encode, self.generator)
            self._last_refresh_step = self._env_steps

    # ------------------------------------------------------------ the tree

    @torch.no_grad()
    def _select_tree(self, state_1d, feat, cands, critic_score, amount):
        tree = self.tree
        self._tree_count += 1
        timing = (self._tree_count % 50 == 1)
        if timing:
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()

        z0 = self.model.encode(feat)                      # (1, latent)

        # Step 1: the root set -- actor candidates, flow behaviour-cloning
        # candidates, and the carried plan chunk from the last decision.
        root_chunks = [cands]
        root_crit = [critic_score]
        sources = ['actor'] * cands.shape[0]
        root_bc = int(tree.root_bc)
        if root_bc > 0:
            bc_cands, _, bc_crit = self.score_candidates(feat, n=root_bc,
                                                         source='bc')
            root_chunks.append(bc_cands)
            root_crit.append(bc_crit)
            sources += ['bc'] * root_bc
        if self.carried is not None:
            carried = self.carried.reshape(1, -1)
            c_qs = self.policy.critic(feat, carried)
            root_chunks.append(carried)
            root_crit.append(self.policy._agg(c_qs).squeeze(-1))
            sources += ['carried']
        cands_all = torch.cat(root_chunks, dim=0)         # (R, chunk_dim)
        crit_all = torch.cat(root_crit, dim=0)            # (R,)
        R = cands_all.shape[0]

        # Step 2: level 1 -- imagine every root chunk from the real state.
        z1, dis1 = self.model.imagine_chunk(
            z0.repeat(R, 1), cands_all.reshape(R, self.chunk_len, self.action_dim))
        trusted1 = self._trusted(dis1, 1)
        unv1 = self.memory.unvisitedness(z1)
        score1 = unv1 * trusted1.float()                  # gamma^(chunk_len * 0) = 1
        levels = [{'z': z1, 'score': score1, 'trusted': trusted1,
                   'root': torch.arange(R, device=self.device),
                   'parent': None, 'chunk': None}]

        # Step 3: levels 2..depth, beam of the best trusted nodes, branch
        # proposer chunks each, one imagination batch per level.
        for d in range(2, int(tree.depth) + 1):
            prev = levels[-1]
            trusted_rows = prev['trusted'].nonzero().squeeze(-1)
            if trusted_rows.numel() == 0:
                break
            scores_np = prev['score'][trusted_rows].cpu().numpy()
            order = np.argsort(-scores_np, kind='stable')  # ties: lower index
            chosen = trusted_rows[torch.as_tensor(
                order[:int(tree.beam)].copy(), device=self.device)]
            parents = prev['z'][chosen]
            rep = parents.repeat_interleave(int(tree.branch), dim=0)
            chunks = self.proposer.propose(rep, self.generator)
            zc, disc = self.model.imagine_chunk(
                rep, chunks.reshape(-1, self.chunk_len, self.action_dim))
            trusted_c = self._trusted(disc, d)             # parents all trusted
            unv = self.memory.unvisitedness(zc)
            score = (self.gamma ** (self.chunk_len * (d - 1))) * unv * trusted_c.float()
            parent_rows = chosen.repeat_interleave(int(tree.branch))
            levels.append({'z': zc, 'score': score, 'trusted': trusted_c,
                           'root': prev['root'][parent_rows],
                           'parent': parent_rows, 'chunk': chunks})

        # Step 4: back up -- each root's reach is the best node score among
        # its descendants, its own level-1 node included.
        reach = levels[0]['score'].clone()
        for lvl in levels[1:]:
            reach.scatter_reduce_(0, lvl['root'], lvl['score'], reduce='amax')

        # Step 5: final score and pick.
        sigma = crit_all.std(correction=0)
        bonus = amount * sigma * torch.clamp(reach - 1.0, 0.0, float(tree.reach_cap))
        final = crit_all + bonus
        exec_idx = int(torch.argmax(final))

        # Step 6: plan carry -- the depth-2 chunk on the path to the best
        # node under the executed candidate, when that node is deep enough.
        best_level, best_row, best_score = 0, exec_idx, float(levels[0]['score'][exec_idx])
        for li in range(1, len(levels)):
            mask = levels[li]['root'] == exec_idx
            if mask.any():
                sc = torch.where(mask, levels[li]['score'],
                                 torch.full_like(levels[li]['score'], -1.0))
                row = int(torch.argmax(sc))
                if float(sc[row]) > best_score:
                    best_level, best_row, best_score = li, row, float(sc[row])
        if best_level >= 1:                                # depth >= 2
            li, row = best_level, best_row
            while li > 1:
                row = int(levels[li]['parent'][row])
                li -= 1
            self.carried = levels[1]['chunk'][row].detach().clone()
        else:
            self.carried = None

        # Step 7: this real state is now visited.
        self.memory.add_online(np.asarray(state_1d, dtype=np.float32), z0[0])

        self._tree_diagnostics(levels, reach, bonus, sigma, crit_all,
                               exec_idx, sources, unv1, best_level)
        if timing:
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            self._acc('tree_milliseconds', (time.perf_counter() - t0) * 1000.0)
        return cands_all[exec_idx].detach().cpu().numpy().reshape(
            self.chunk_len, self.action_dim)

    def _trusted(self, disagreement, depth):
        """ A node is trusted while its disagreement stays within trust_cap
            times the real-data reference at its depth; with no reference
            measured yet, nothing at that depth is trusted. """
        ref = self.reference[depth - 1]
        if ref is None:
            return torch.zeros_like(disagreement, dtype=torch.bool)
        return disagreement <= float(self.tree.trust_cap) * ref

    def _tree_diagnostics(self, levels, reach, bonus, sigma, crit_all,
                          exec_idx, sources, unv1, best_level):
        self._acc('tree_grown_frac', 1.0)
        self._acc('reach_picked', float(reach[exec_idx]))
        self._acc('reach_max', float(reach.max()))
        self._acc('reach_mean', float(reach.mean()))
        self._acc('unvisitedness_level1_mean', float(unv1.mean()))
        self._acc('unvisitedness_deepest_mean', self._deepest_unv(levels))
        for li, lvl in enumerate(levels):
            self._acc(f'trusted_frac_d{li + 1}', float(lvl['trusted'].float().mean()))
        self._acc('best_depth_picked', float(best_level + 1))
        crit_arg = int(torch.argmax(crit_all))
        self._acc('pick_changed', float(exec_idx != crit_arg))
        src = sources[exec_idx]
        for name in ('actor', 'bc', 'carried'):
            self._acc(f'picked_source_{name}', float(src == name))
        self._acc('bonus_over_spread', float(bonus.max()) / (abs(float(sigma)) + 1e-8))

    def _deepest_unv(self, levels):
        lvl = levels[-1]
        depth = len(levels)
        disc = self.gamma ** (self.chunk_len * (depth - 1))
        # score = disc * unvisitedness * trusted; recover the mean
        # unvisitedness of the deepest level from its trusted nodes when
        # any exist, else 0.
        t = lvl['trusted']
        if t.any():
            return float((lvl['score'][t] / disc).mean())
        return 0.0


class TreeSwitchSelector(TreeSelector):
    """ The same tree; only the explore amount differs. In a switch-enabled
        episode (bandit arm 1) the amount is recomputed at every chunk
        boundary from a stuck score read off the critic:

            current value     = best critic score among the ACTOR candidates
            no-progress value = current reward / (1 - gamma)
            expected progress = current value - no-progress value
            broken promise    = max(0, value promised lookback_chunks
                                 decisions ago - what was delivered since)

        both ranked against recent history; stuck in [0, 1];
        amount = tree.beta * stuck. Bandit arm 0 gives amount 0 for the
        whole episode (pure combined method, and clean statistics). """

    def __init__(self, *args, switch=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert switch is not None
        self.switch = switch
        assert str(switch.signal) in ('both', 'progress', 'promise', 'constant'), \
            switch.signal
        self._progress_window = []
        self._promise_window = []
        self._reset_episode_state()

    def _reset_episode_state(self):
        self._last_reward = None
        self._ep_rewards = []
        self._decisions = []        # (decision value, step index) per decision

    def _observe_reward(self, reward):
        self._last_reward = reward
        self._ep_rewards.append(reward)

    def explore_amount(self, critic_score_actor):
        sw = self.switch
        current_value = float(critic_score_actor.max())
        arm = 1 if self.bandit is None else self.bandit.arm

        if self._last_reward is None:
            # First decision of the episode: no reward observed yet.
            self._decisions.append((current_value, len(self._ep_rewards)))
            return 0.0

        no_progress_value = self._last_reward / (1.0 - self.gamma)
        expected_progress = current_value - no_progress_value

        broken_promise = 0.0
        if len(self._decisions) >= int(sw.lookback_chunks):
            promised, step_idx = self._decisions[-int(sw.lookback_chunks)]
            since = self._ep_rewards[step_idx:]
            delivered = sum((self.gamma ** j) * r for j, r in enumerate(since))
            delivered += (self.gamma ** len(since)) * current_value
            broken_promise = max(0.0, promised - delivered)

        # Ranks against recent history (window excludes this decision).
        low_progress_rank = (float(np.mean(
            [v > expected_progress for v in self._progress_window]))
            if self._progress_window else 0.0)
        broken_promise_rank = (float(np.mean(
            [v < broken_promise for v in self._promise_window]))
            if self._promise_window else 0.0)

        self._progress_window.append(expected_progress)
        self._promise_window.append(broken_promise)
        cap = int(sw.rank_window)
        if len(self._progress_window) > cap:
            del self._progress_window[:len(self._progress_window) - cap]
        if len(self._promise_window) > cap:
            del self._promise_window[:len(self._promise_window) - cap]
        self._decisions.append((current_value, len(self._ep_rewards)))

        signal = str(sw.signal)
        if signal == 'both':
            stuck = max(low_progress_rank, broken_promise_rank)
        elif signal == 'progress':
            stuck = low_progress_rank
        elif signal == 'promise':
            stuck = broken_promise_rank
        else:
            stuck = float(sw.constant_stuck)

        self._acc('switch_stuck', stuck)
        self._acc('switch_expected_progress', expected_progress)
        self._acc('switch_broken_promise', broken_promise)
        self._acc('switch_low_progress_rank', low_progress_rank)
        self._acc('switch_broken_promise_rank', broken_promise_rank)
        self._acc('switch_negative_progress_frac', float(expected_progress < 0))

        if arm == 0:
            return 0.0
        if len(self._progress_window) < int(sw.min_history):
            return 0.0
        return float(self.tree.beta) * stuck
