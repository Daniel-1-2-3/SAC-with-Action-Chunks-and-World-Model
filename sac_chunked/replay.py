""" One replay buffer for every arm.

    A flat ring buffer of single-step transitions, the same structure QC's
    utils/datasets.py uses. All training rows -- QC-FQL chunk transitions AND
    the latent model's short windows -- are cut out of it with uniform start
    indices and boundary MASKING rather than rejection. Every arm therefore
    trains on exactly the same data distribution; what differs between arms is
    only what they compute from it.

    (Before this, the control arm sampled from a flat buffer while the model
    arm sampled fixed-length sequences from a 2000-episode FIFO. That was a
    second difference between arms on top of the scorer, and it is gone.) """

import numpy as np
import torch

from sac_chunked.chunk_utils import pool_chunk_np, step_valid_np


class ChunkTransitionReplay:
    """ Flat ring buffer that serves chunk-level transitions and latent-model
        windows.

        Windows are sampled uniformly and masked, not rejected -- see
        sample_chunks. This mirrors utils/datasets.py sample_sequence in the
        reference implementation. Rejection would bias sampling away from
        episode ends, which on a cube task is exactly where completions are. """

    def __init__(self, obs_dim, action_dim, chunk_len, capacity=1_000_000,
                 success_reward_thresh=-1.0):
        self.capacity = int(capacity)
        self.action_dim = action_dim
        self.chunk_len = chunk_len
        # -1.0 is the sparse "no progress" baseline on the cube tasks; any
        # reward above it means a subtask succeeded at that step.
        self.success_reward_thresh = success_reward_thresh
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.mask = np.zeros((self.capacity, 1), dtype=np.float32)
        self.terminal = np.zeros((self.capacity, 1), dtype=np.float32)
        self.idx = 0
        self.full = False
        self.offline_episodes = 0
        self.offline_success = 0
        self.online_episodes = 0
        self.online_success = 0
        self._ep_success = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, terminated, truncated=False):
        i = self.idx
        self.obs[i] = obs
        self.action[i] = action
        self.reward[i] = reward
        self.next_obs[i] = next_obs
        self.mask[i] = 0.0 if terminated else 1.0
        # A truncation ends the window for `valid` purposes even though it
        # does not zero the bootstrap mask.
        self.terminal[i] = 1.0 if (terminated or truncated) else 0.0
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

        if reward > self.success_reward_thresh:
            self._ep_success = True
        if terminated or truncated:
            self.online_episodes += 1
            self.online_success += int(self._ep_success)
            self._ep_success = False

    def seed_from_offline(self, dataset):
        obs = np.asarray(dataset['observations'], dtype=np.float32)
        act = np.asarray(dataset['actions'], dtype=np.float32)
        rew = np.asarray(dataset['rewards'], dtype=np.float32).reshape(-1, 1)
        nobs = np.asarray(dataset['next_observations'], dtype=np.float32)
        term = np.asarray(dataset['terminals']).reshape(-1).astype(bool)
        if 'masks' in dataset:
            mk = np.asarray(dataset['masks'], dtype=np.float32).reshape(-1, 1)
        else:
            mk = (~term).astype(np.float32).reshape(-1, 1)

        n = min(len(obs), self.capacity)
        sl = slice(0, n)
        self.obs[sl] = obs[:n]
        self.action[sl] = act[:n]
        self.reward[sl] = rew[:n]
        self.next_obs[sl] = nobs[:n]
        self.mask[sl] = mk[:n]
        self.terminal[sl] = term[:n].astype(np.float32).reshape(-1, 1)
        self.idx = n % self.capacity
        if n >= self.capacity:
            self.full = True

        ep_ok = False
        for t in range(n):
            if rew[t, 0] > self.success_reward_thresh:
                ep_ok = True
            if term[t]:
                self.offline_episodes += 1
                self.offline_success += int(ep_ok)
                ep_ok = False
        return n

    # ------------------------------------------------------------ QC-FQL rows

    def sample_chunks(self, batch_size, device, rng, gamma):
        """ utils/datasets.py sample_sequence. Start indices are uniform over
            the buffer with no episode-boundary rejection; windows that cross
            a terminal are kept and masked. Returns the per-position validity
            too, for the BC flow term. """
        h = self.chunk_len
        size = len(self)
        if size <= h:
            return None
        starts = rng.integers(0, size - h + 1, size=batch_size)
        window = starts[:, None] + np.arange(h)[None, :]

        rewards = self.reward[window, 0]
        masks = self.mask[window, 0]
        terminals = self.terminal[window, 0]
        chunk_reward, chunk_mask, valid = pool_chunk_np(rewards, masks, terminals, gamma)
        step_valid = step_valid_np(terminals)
        chunks = self.action[window].reshape(batch_size, h * self.action_dim)

        to = lambda x: torch.as_tensor(x, device=device).float()
        return (to(self.obs[starts]), to(chunks), to(chunk_reward), to(chunk_mask),
                to(valid), to(step_valid), to(self.next_obs[window[:, -1]]))

    # ------------------------------------------------------- latent-model rows

    def _windows(self, starts, horizon, device):
        window = starts[:, None] + np.arange(horizon)[None, :]
        terminals = self.terminal[window, 0]
        to = lambda x: torch.as_tensor(x, device=device).float()
        return {
            'obs': to(self.obs[window]),                 # (B, H, obs_dim)
            'next_obs': to(self.next_obs[window]),       # (B, H, obs_dim)
            'action': to(self.action[window]),           # (B, H, action_dim)
            'reward': to(self.reward[window]),           # (B, H, 1)
            'mask': to(self.mask[window]),               # (B, H, 1)
            'valid': to(step_valid_np(terminals))[..., None],  # (B, H, 1)
        }

    def sample_model_windows(self, batch_size, horizon, device, rng):
        """ Consecutive single-step transitions for the latent model:
            obs_t, next_obs_t, a_t, r_t for t in [start, start+horizon).

            next_obs is each transition's OWN next observation, so the
            consistency target is right even at an episode boundary (the
            steps after it are masked by `valid`). Same uniform-start,
            mask-don't-reject rule as sample_chunks. """
        size = len(self)
        if size < horizon:
            return None
        starts = rng.integers(0, size - horizon + 1, size=batch_size)
        return self._windows(starts, horizon, device)

    def sample_reward_windows(self, batch_size, horizon, device, rng):
        """ DIAGNOSTICS ONLY. Windows guaranteed to contain an above-baseline
            reward step, so the ground truth has variance and correlations
            are defined. Training must never use this -- it changes the data
            distribution the model and critic are fit on.

            Returns None when the buffer holds no above-baseline reward yet. """
        size = len(self)
        if size < horizon:
            return None
        hits = np.flatnonzero(self.reward[:size, 0] > self.success_reward_thresh)
        if len(hits) == 0:
            return None
        r = hits[rng.integers(0, len(hits), size=batch_size)]
        lo = np.maximum(0, r - horizon + 1)
        hi = np.minimum(r, size - horizon)
        starts = (lo + rng.random(batch_size) * (hi - lo + 1)).astype(np.int64)
        starts = np.clip(starts, 0, size - horizon)
        return self._windows(starts, horizon, device)

    @property
    def success_stats(self):
        n_off, n_on = self.offline_episodes, self.online_episodes
        off, on = self.offline_success, self.online_success
        return {
            'offline_frac': off / max(n_off, 1),
            'online_frac': on / max(n_on, 1),
            'total_frac': (off + on) / max(n_off + n_on, 1),
            'offline_success': off,
            'online_success': on,
        }
