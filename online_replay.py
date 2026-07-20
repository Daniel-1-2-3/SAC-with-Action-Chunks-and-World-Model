import collections
import numpy as np
from ogbench_methods import OGBenchMethods

class OnlineReplay:
    def __init__(self, obs_key='state', action_key='action', max_episodes=2000):
        self.obs_key = obs_key
        self.action_key = action_key
        self.max_episodes = max_episodes
        self.dreamer_episodes = []
        self._raw = collections.defaultdict(list)
        self.total_transitions = 0

    def __len__(self):
        return self.total_transitions

    def add_step(self, obs, action, reward, next_obs, terminated, truncated):
        self._raw['observations'].append(np.asarray(obs, dtype=np.float32))
        self._raw['actions'].append(np.asarray(action, dtype=np.float32))
        self._raw['rewards'].append(np.float32(reward))
        self._raw['next_observations'].append(np.asarray(next_obs, dtype=np.float32))
        done = bool(terminated or truncated)
        self._raw['terminals'].append(done)
        self._raw['masks'].append(0.0 if terminated else 1.0)
        self.total_transitions += 1
        if done:
            self._finalize_episode()

    # Converts a real episode to a format the DreamerV3 can be trained on
    def _finalize_episode(self):
        ep = {k: np.stack(v, axis=0) for k, v in self._raw.items()}
        self._raw = collections.defaultdict(list)
        dreamer_ep = OGBenchMethods.ogbench_to_dreamer_episode(
            ep, obs_key=self.obs_key, action_key=self.action_key)
        self.dreamer_episodes.append(dreamer_ep)
        if len(self.dreamer_episodes) > self.max_episodes:
            self.dreamer_episodes.pop(0) # Drop the oldest episode, FIFO

    # Warm start from the static OGBench dataset, put some dataset episodes into replay at start
    def seed_from_offline(self, dreamer_episodes, n=None, rng=None):
        eps = dreamer_episodes
        if n is not None and n < len(eps):
            rng = rng or np.random.default_rng()
            idx = rng.choice(len(eps), size=n, replace=False)
            eps = [eps[i] for i in idx]
        self.dreamer_episodes.extend(eps)

    # Check if has enough data to sample a training batch
    def ready(self, seq_len, min_episodes=1):
        usable = [e for e in self.dreamer_episodes if len(e[self.obs_key]) >= seq_len]
        return len(usable) >= min_episodes

    def sample_batch(self, batch_size, seq_len, rng=None):
        usable = [e for e in self.dreamer_episodes if len(e[self.obs_key]) >= seq_len]
        if not usable:
            raise RuntimeError(
                f'No episodes long enough (need >= {seq_len} steps) to sample yet. '
                f'Check replay.ready(seq_len) before calling sample_batch.')
        return OGBenchMethods.sample_dreamer_batch(
            usable, batch_size, seq_len,
            obs_key=self.obs_key, action_key=self.action_key, rng=rng)
