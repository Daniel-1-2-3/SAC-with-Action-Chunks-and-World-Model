""" Minimal episode replay for the v8 stack (jax-free; the legacy
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
        self.best_online_reward = -np.inf
        self.total_steps = 0

    def add_step(self, obs, action, reward, next_obs, terminated, truncated):
        if not self._obs:
            self._obs.append(np.asarray(obs, np.float32))
        self._act.append(np.asarray(action, np.float32))
        self._rew.append(float(reward))
        self._term.append(bool(terminated))
        self._obs.append(np.asarray(next_obs, np.float32))
        self.total_steps += 1
        if terminated or truncated:
            self.end_episode()

    def end_episode(self):
        if not self._act:
            return
        ep = self._build(self._obs, self._act, self._rew, self._term)
        self.episodes.append(ep)
        if len(self.episodes) > self.max_episodes:
            self.episodes.pop(0)
        r = ep['reward'][1:]
        if len(r):
            self.best_online_reward = max(self.best_online_reward,
                                          float(r.max()))
        self._obs, self._act, self._rew, self._term = [], [], [], []

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

    # ---------- sampling ----------
    def ready(self, seq_len):
        return any(len(e[self.obs_key]) >= seq_len for e in self.episodes)

    def sample_seqs(self, batch, seq_len, rng):
        eps = [e for e in self.episodes if len(e[self.obs_key]) >= seq_len]
        return sample_sequences(eps, batch, seq_len, self.obs_key,
                                self.action_key, rng)

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
