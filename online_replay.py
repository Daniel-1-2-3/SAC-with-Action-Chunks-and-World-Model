import collections
import numpy as np
from ogbench_methods import OGBenchMethods

class OnlineReplay:
    def __init__(self, obs_key='state', action_key='action', max_episodes=2000,
                 max_success_episodes=200, success_frac=0.25, success_reward_thresh=-1.0):
        self.obs_key = obs_key
        self.action_key = action_key
        self.max_episodes = max_episodes
        # success_episodes has its own small cap and is never touched by the
        # online FIFO eviction above -- this is what keeps reward signal in
        # the training batches once online episodes outnumber it.
        self.max_success_episodes = max_success_episodes
        self.success_frac = success_frac
        # -1.0 is this task's sparse "no progress" baseline; any reward
        # above it means a subtask succeeded at some step in the episode.
        self.success_reward_thresh = success_reward_thresh

        self.offline_episodes = []
        self.online_episodes = []
        self.success_episodes = []
        self._raw = collections.defaultdict(list)
        self.total_transitions = 0

    def __len__(self):
        return self.total_transitions

    @property
    def dreamer_episodes(self):
        # success_episodes are copies of entries already counted here, so
        # they're excluded from this combined view to avoid double-counting
        # (keeps diagnosis/replay_episodes meaningful).
        return self.offline_episodes + self.online_episodes

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

        self.online_episodes.append(dreamer_ep)
        if len(self.online_episodes) > self.max_episodes:
            self.online_episodes.pop(0) # Drop the oldest episode, FIFO

        self._maybe_add_success(dreamer_ep)

    def _maybe_add_success(self, dreamer_ep):
        if np.any(dreamer_ep['reward'] > self.success_reward_thresh):
            self.success_episodes.append(dreamer_ep)
            if len(self.success_episodes) > self.max_success_episodes:
                self.success_episodes.pop(0)

    # Warm start from the static OGBench dataset, put some dataset episodes into replay at start
    def seed_from_offline(self, dreamer_episodes, n=None, rng=None):
        eps = dreamer_episodes
        if n is not None and n < len(eps):
            rng = rng or np.random.default_rng()
            idx = rng.choice(len(eps), size=n, replace=False)
            eps = [eps[i] for i in idx]
        self.offline_episodes.extend(eps)
        for ep in eps:
            self._maybe_add_success(ep)

    # Check if has enough data to sample a training batch
    def ready(self, seq_len, min_episodes=1):
        usable = [e for e in self.dreamer_episodes if len(e[self.obs_key]) >= seq_len]
        return len(usable) >= min_episodes

    def sample_batch(self, batch_size, seq_len, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        usable = [e for e in self.dreamer_episodes if len(e[self.obs_key]) >= seq_len]
        if not usable:
            raise RuntimeError(
                f'No episodes long enough (need >= {seq_len} steps) to sample yet. '
                f'Check replay.ready(seq_len) before calling sample_batch.')

        usable_success = [e for e in self.success_episodes if len(e[self.obs_key]) >= seq_len]
        n_success = min(int(round(batch_size * self.success_frac)), batch_size) if usable_success else 0

        if n_success == 0:
            return OGBenchMethods.sample_dreamer_batch(
                usable, batch_size, seq_len,
                obs_key=self.obs_key, action_key=self.action_key, rng=rng)

        # bias_start_to_reward=True only here: this pool is already
        # reward-selected at the episode level, so it's also where making
        # sure the reward step actually lands inside the sampled window
        # matters. regular_batch below is untouched -- still uniform.
        success_batch = OGBenchMethods.sample_dreamer_batch(
            usable_success, n_success, seq_len,
            obs_key=self.obs_key, action_key=self.action_key, rng=rng,
            bias_start_to_reward=True)
        regular_batch = OGBenchMethods.sample_dreamer_batch(
            usable, batch_size - n_success, seq_len,
            obs_key=self.obs_key, action_key=self.action_key, rng=rng)
        return {k: np.concatenate([success_batch[k], regular_batch[k]], axis=0) for k in regular_batch}