import collections
import numpy as np
from helpers.ogbench_methods import OGBenchMethods

class OnlineReplay:
    def __init__(self, obs_key='state', action_key='action', max_episodes=2000,
                 success_reward_thresh=-1.0):
        self.obs_key = obs_key
        self.action_key = action_key
        self.max_episodes = max_episodes
        # -1.0 is this task's sparse "no progress" baseline; any reward above
        # it means a subtask succeeded at some step in the episode.
        self.success_reward_thresh = success_reward_thresh
        # Offline demonstrations are seeded once at startup and never evicted
        # -- the online FIFO below only ever pops from online_episodes.
        self.offline_episodes = []
        self.online_episodes = []
        # Success bookkeeping for reporting only -- it does NOT affect
        # sampling (batches are uniform). Flags are cached at insert time so
        # the log path never rescans every reward array. offline never
        # evicts, so a single count suffices there; online_success_flags is
        # kept parallel to online_episodes and popped in lockstep.
        self.offline_success_count = 0
        self.online_success_flags = []
        # Per-episode max reward past the fabricated index 0, kept in
        # lockstep with the episode lists. Lets sample_reward_batch filter
        # to reward-bearing episodes at ANY threshold without rescanning
        # reward arrays.
        self.offline_max_rewards = []
        self.online_max_rewards = []
        self._raw = collections.defaultdict(list)
        self.total_transitions = 0

    def __len__(self):
        return self.total_transitions

    @property
    def dreamer_episodes(self):
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
        self.online_success_flags.append(self._is_success(dreamer_ep))
        self.online_max_rewards.append(self._max_reward(dreamer_ep))
        if len(self.online_episodes) > self.max_episodes:
            self.online_episodes.pop(0) # Drop the oldest episode, FIFO
            self.online_success_flags.pop(0) # kept in lockstep
            self.online_max_rewards.pop(0) # kept in lockstep

    # Warm start from the static OGBench dataset, put some dataset episodes into replay at start
    def seed_from_offline(self, dreamer_episodes, n=None, rng=None):
        eps = dreamer_episodes
        if n is not None and n < len(eps):
            rng = rng or np.random.default_rng()
            idx = rng.choice(len(eps), size=n, replace=False)
            eps = [eps[i] for i in idx]
        self.offline_episodes.extend(eps)
        self.offline_success_count += sum(self._is_success(ep) for ep in eps)
        self.offline_max_rewards.extend(self._max_reward(ep) for ep in eps)

    def _max_reward(self, dreamer_ep):
        # Index 0 is skipped for the same reason as in _is_success below.
        r = dreamer_ep['reward'][1:]
        return float(r.max()) if len(r) else -np.inf

    def _is_success(self, dreamer_ep):
        # reward[0] is the fabricated 0.0 that ogbench_to_dreamer_episode
        # prepends for Dreamer's "reward for arriving at state[t]" convention.
        # It clears this task's -1.0 baseline for EVERY episode, so index 0
        # must be skipped or every episode reads as a success.
        return bool(np.any(dreamer_ep['reward'][1:] > self.success_reward_thresh))

    @property
    def success_stats(self):
        """ Fraction of buffered episodes containing at least one
            above-baseline reward step. Split offline/online because the
            offline number is a fixed reference line while the online number
            is the one that should climb if the policy is actually learning. """
        n_off, n_on = len(self.offline_episodes), len(self.online_episodes)
        off_succ, on_succ = self.offline_success_count, sum(self.online_success_flags)
        return {
            'offline_episodes': n_off,
            'online_episodes': n_on,
            'offline_success': off_succ,
            'online_success': on_succ,
            'offline_frac': off_succ / max(n_off, 1),
            'online_frac': on_succ / max(n_on, 1),
            'total_frac': (off_succ + on_succ) / max(n_off + n_on, 1),
        }

    # Check if has enough data to sample a training batch
    def ready(self, seq_len, min_episodes=1):
        usable = [e for e in self.dreamer_episodes if len(e[self.obs_key]) >= seq_len]
        return len(usable) >= min_episodes

    def sample_reward_batch(self, batch_size, seq_len, reward_thresh,
                            rng=None):
        """ v2 self-imitation sampling: like sample_batch with
            bias_start_to_reward, but ALSO filtered to episodes whose max
            reward clears reward_thresh. sample_dreamer_batch picks episodes
            uniformly and only biases the window start WITHIN an episode, so
            in a buffer where reward-bearing episodes are rare the bias alone
            still returns mostly floor windows; the episode filter is what
            makes the batch actually reward-dense. Returns None while no
            qualifying episode exists -- the caller falls back to uniform,
            so this is inert until the first partial success. Intended ONLY
            for the actor's flow-BC batch; critic and world-model training
            batches stay uniform (see sample_batch's warning below). """
        if rng is None:
            rng = np.random.default_rng()
        pairs = list(zip(self.offline_episodes, self.offline_max_rewards)) + \
            list(zip(self.online_episodes, self.online_max_rewards))
        usable = [e for e, m in pairs
                  if m > reward_thresh and len(e[self.obs_key]) >= seq_len]
        if not usable:
            return None
        return OGBenchMethods.sample_dreamer_batch(
            usable, batch_size, seq_len,
            obs_key=self.obs_key, action_key=self.action_key, rng=rng,
            bias_start_to_reward=True, reward_thresh=reward_thresh)

    def sample_batch(self, batch_size, seq_len, rng=None,
                     bias_start_to_reward=False, bias_reward_thresh=None):
        """ bias_start_to_reward is for DIAGNOSTICS and Dyna SEED SELECTION
            only. Training batches for the critic/actor and the world model
            must stay uniform over start states; biasing them would change the
            data distribution they are fit on.

            bias_reward_thresh: reward level that counts as a "hit" when
            biasing. Defaults to success_reward_thresh (-1.0, full success
            only), which on cube-triple's play data almost never fires and
            silently degrades the bias to uniform sampling. Pass the partial-
            progress level (e.g. -2.5: anything above the -3 floor) when the
            consumer needs reward VARIATION rather than full successes. """
        if rng is None:
            rng = np.random.default_rng()

        usable = [e for e in self.dreamer_episodes if len(e[self.obs_key]) >= seq_len]
        if not usable:
            raise RuntimeError(
                f'No episodes long enough (need >= {seq_len} steps) to sample yet. '
                f'Check replay.ready(seq_len) before calling sample_batch.')

        thresh = self.success_reward_thresh if bias_reward_thresh is None \
            else bias_reward_thresh
        return OGBenchMethods.sample_dreamer_batch(
            usable, batch_size, seq_len,
            obs_key=self.obs_key, action_key=self.action_key, rng=rng,
            bias_start_to_reward=bias_start_to_reward,
            reward_thresh=thresh)