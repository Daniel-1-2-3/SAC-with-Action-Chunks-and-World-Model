""" Minimal episode replay for the SEAR/P2E stack (jax-free; the legacy
    OnlineReplay stays untouched for the QC baseline). Stores dreamer-format
    episodes; also serves flat (obs, act, next_obs) pairs for the ensemble.
"""
import numpy as np

from helpers.common import sample_sequences


class EpisodeReplay:
    def __init__(self, obs_key='state', action_key='action',
                 max_episodes=2000):
        self.obs_key, self.action_key = obs_key, action_key
        self.max_episodes = max_episodes
        self.episodes = []
        self._obs, self._act, self._rew, self._term = [], [], [], []
        self._ach = []
        self._goal = None
        self.best_online_reward = -np.inf
        self.total_steps = 0

    def add_step(self, obs, action, reward, next_obs, terminated, truncated,
                 achieved=None, next_achieved=None):
        if not self._obs:
            self._obs.append(np.asarray(obs, np.float32))
            if achieved is not None:
                self._ach.append(np.asarray(achieved, np.float32))
        self._act.append(np.asarray(action, np.float32))
        self._rew.append(float(reward))
        self._term.append(bool(terminated))
        self._obs.append(np.asarray(next_obs, np.float32))
        if next_achieved is not None:
            self._ach.append(np.asarray(next_achieved, np.float32))
        self.total_steps += 1
        if terminated or truncated:
            self.end_episode()

    def end_episode(self):
        if not self._act:
            return
        ep = self._build(self._obs, self._act, self._rew, self._term)
        if self._ach:
            ep['achieved'] = np.asarray(self._ach, np.float32)
        if self._goal is not None:
            ep['goal'] = np.tile(self._goal,
                                 (len(ep[self.obs_key]), 1)).astype(
                np.float32)
        self.episodes.append(ep)
        if len(self.episodes) > self.max_episodes:
            self.episodes.pop(0)
        r = ep['reward'][1:]
        if len(r):
            self.best_online_reward = max(self.best_online_reward,
                                          float(r.max()))
        self._obs, self._act, self._rew, self._term = [], [], [], []
        self._ach = []

    def _build(self, obs, act, rew, term):
        T = len(obs)                                     # steps + 1
        A = len(act[0])
        ep = {
            self.obs_key: np.asarray(obs, np.float32),
            self.action_key: np.concatenate(
                [np.zeros((1, A), np.float32),
                 np.asarray(act, np.float32)], 0),
            'reward': np.concatenate(
                [np.zeros(1, np.float32), np.asarray(rew, np.float32)], 0),
            'is_first': np.zeros(T, bool),
            'is_last': np.zeros(T, bool),
            'is_terminal': np.zeros(T, bool),
        }
        ep['is_first'][0] = True
        ep['is_last'][-1] = True
        ep['is_terminal'][-1] = bool(term[-1])
        ep['cont'] = (~ep['is_terminal']).astype(np.float32)
        return ep

    def set_goal(self, goal):
        """ Record the active task goal; stored with the episode so its
            windows are labeled with THEIR goal, not a later one. """
        self._goal = None if goal is None else np.asarray(goal, np.float32)

    # ---------- sampling ----------
    def ready(self, seq_len):
        return any(len(e[self.obs_key]) >= seq_len for e in self.episodes)

    def sample_seqs(self, batch, seq_len, rng, reward_frac=0.0,
                    reward_thresh=-1.5):
        """ Uniform sequence sampling, with an optional fraction of the
            batch drawn from reward-bearing episodes, start-biased so the
            sequence contains a reward hit. Biased sampling without
            importance correction (labeled): the fraction is kept small so
            the bias stays modest. """
        eps = [e for e in self.episodes if len(e[self.obs_key]) >= seq_len]
        extra = [k for k in ('achieved', 'goal')
                 if eps and k in eps[0]]
        n_r = int(round(batch * reward_frac))
        out = None
        if n_r > 0:
            rich = [e for e in eps
                    if len(e['reward']) > 1
                    and e['reward'][1:].max() > reward_thresh]
            if rich:
                keys = [self.obs_key, self.action_key, 'reward', 'is_first',
                        'is_terminal', 'cont'] + extra
                out = {k: [] for k in keys}
                for _ in range(n_r):
                    ep = rich[rng.integers(0, len(rich))]
                    hits = 1 + np.flatnonzero(
                        ep['reward'][1:] > reward_thresh)
                    h = int(hits[rng.integers(0, len(hits))])
                    lo = max(0, h - seq_len + 1)
                    hi = min(h, len(ep[self.obs_key]) - seq_len)
                    start = int(rng.integers(lo, hi + 1)) if hi >= lo else 0
                    for k in keys:
                        out[k].append(ep[k][start:start + seq_len])
            else:
                n_r = 0
        uni = sample_sequences(eps, batch - n_r, seq_len, self.obs_key,
                               self.action_key, rng, extra_keys=extra)
        if out is None:
            return uni
        return {k: np.concatenate([np.stack(out[k]), uni[k]])
                for k in uni}

    def sample_pairs(self, batch, rng):
        """ Flat (obs, act, next_obs) transitions for the ensemble. """
        eps = [e for e in self.episodes if len(e[self.obs_key]) >= 2]
        o, a, no = [], [], []
        for _ in range(batch):
            ep = eps[rng.integers(0, len(eps))]
            t = rng.integers(1, len(ep[self.obs_key]))
            o.append(ep[self.obs_key][t - 1])
            a.append(ep[self.action_key][t])
            no.append(ep[self.obs_key][t])
        return np.stack(o), np.stack(a), np.stack(no)

    def sample_start_obs(self, batch, rng, recent=200):
        eps = self.episodes[-recent:]
        out = []
        for _ in range(batch):
            ep = eps[rng.integers(0, len(eps))]
            out.append(ep[self.obs_key][
                rng.integers(0, len(ep[self.obs_key]))])
        return np.stack(out)

    def success_stats(self, thresh=-0.5):
        n = len(self.episodes)
        succ = sum(1 for e in self.episodes
                   if len(e['reward']) > 1 and e['reward'][1:].max() > thresh)
        return {'episodes': n, 'success_episodes': succ,
                'success_frac': succ / max(n, 1),
                'best_reward': self.best_online_reward}
