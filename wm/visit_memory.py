""" Visit memory: raw observations the robot has actually seen, plus an
    encoded copy, answering "how far is this latent from anywhere visited?"

    Contents: a fixed sample of observations from the offline region of the
    replay (filled once at attach_replay time) and a first-in-first-out
    store of online observations, one per online chunk boundary.

    The encoded copy lives on the device. Online observations are encoded
    at insertion with the current encoder (the tree's root latent is
    reused); because the encoder keeps training, every stored observation
    is re-encoded periodically (refresh), which also re-measures the
    typical distance.

    unvisitedness(latent) = mean distance to the k nearest stored latents
                            / typical_distance

    typical_distance: at each refresh, the median over 2048 random stored
    latents of their own k-nearest distance to the memory, excluding the
    latent itself. So unvisitedness ~ 1 means "as far from the data as the
    data is from itself", and > 1 means genuinely outside it. """

import numpy as np
import torch


class VisitMemory:

    def __init__(self, obs_dim, device, offline_size=25000, online_size=25000,
                 k=5, block=8192):
        self.obs_dim = int(obs_dim)
        self.device = device
        self.offline_size = int(offline_size)
        self.online_size = int(online_size)
        self.k = int(k)
        self.block = int(block)
        self.offline_obs = np.zeros((0, self.obs_dim), dtype=np.float32)
        self.online_obs = np.zeros((self.online_size, self.obs_dim), dtype=np.float32)
        self.online_count = 0          # filled rows (saturates at online_size)
        self.online_ptr = 0            # next row to write (first-in-first-out)
        self.latents = None            # (offline + filled online, latent_dim)
        self.typical_distance = None   # set by the first refresh

    def __len__(self):
        return len(self.offline_obs) + min(self.online_count, self.online_size)

    def fill_offline(self, replay, rng):
        """ Sample offline_size observations uniformly from the offline
            region of the replay (indices < replay.offline_size). """
        n_off = int(getattr(replay, 'offline_size', 0))
        if n_off <= 0:
            return 0
        take = min(self.offline_size, n_off)
        idx = rng.integers(0, n_off, size=take)
        self.offline_obs = replay.obs[idx].astype(np.float32).copy()
        return take

    def add_online(self, obs, latent):
        """ One real online observation and its latent (encoded by the
            caller with the current encoder). First-in-first-out. """
        row = self.online_ptr
        self.online_obs[row] = np.asarray(obs, dtype=np.float32).reshape(-1)
        self.online_ptr = (self.online_ptr + 1) % self.online_size
        self.online_count = min(self.online_count + 1, self.online_size)
        if self.latents is not None:
            flat = latent.detach().reshape(-1)
            pos = len(self.offline_obs) + row
            if pos < self.latents.shape[0]:
                self.latents[pos] = flat
            else:
                self.latents = torch.cat([self.latents, flat[None]], dim=0)

    def _stored_obs(self):
        n_on = min(self.online_count, self.online_size)
        if n_on == 0:
            return self.offline_obs
        return np.concatenate([self.offline_obs, self.online_obs[:n_on]], axis=0)

    @torch.no_grad()
    def refresh(self, encode, generator=None):
        """ Re-encode every stored observation in batches (the encoder keeps
            training) and re-measure typical_distance. generator: the
            caller's torch.Generator, so the measurement's random sample
            never consumes the global RNG. """
        obs = self._stored_obs()
        if len(obs) == 0:
            return
        outs = []
        for start in range(0, len(obs), 4096):
            batch = torch.as_tensor(obs[start:start + 4096], device=self.device)
            outs.append(encode(batch))
        self.latents = torch.cat(outs, dim=0)
        self._measure_typical(generator)

    @torch.no_grad()
    def _measure_typical(self, generator=None):
        n = self.latents.shape[0]
        if n < 2:
            self.typical_distance = None
            return
        take = min(2048, n)
        idx = torch.randperm(n, device=self.device, generator=generator)[:take]
        q = self.latents[idx]
        # k+1 nearest including the query itself, then drop the self match
        # (its distance is exactly 0) and average the remaining k.
        k_plus = min(self.k + 1, n)
        dists = self._knn(q, k_plus)                      # (take, k_plus)
        mean_k = dists[:, 1:].mean(dim=1) if k_plus > 1 else dists[:, 0]
        self.typical_distance = float(mean_k.median().clamp_min(1e-8))

    @torch.no_grad()
    def _knn(self, q, k):
        """ Smallest k Euclidean distances from each query row to the
            memory, (B, k). Blocked over the memory so peak memory stays
            bounded whatever the store size. """
        best = None
        n = self.latents.shape[0]
        for start in range(0, n, self.block):
            d = torch.cdist(q, self.latents[start:start + self.block])
            cat = d if best is None else torch.cat([best, d], dim=1)
            kk = min(k, cat.shape[1])
            best = cat.topk(kk, dim=1, largest=False).values
        return best

    @torch.no_grad()
    def unvisitedness(self, latents):
        """ latents (B, latent_dim) -> (B,): mean distance to the k nearest
            stored latents, divided by typical_distance. Requires a refresh
            to have run (the tree refreshes at the first online decision). """
        assert self.latents is not None and self.typical_distance is not None, \
            'visit memory queried before its first refresh'
        k = min(self.k, self.latents.shape[0])
        return self._knn(latents, k).mean(dim=1) / self.typical_distance
