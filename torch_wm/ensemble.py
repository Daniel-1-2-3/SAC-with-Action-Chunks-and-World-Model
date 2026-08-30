""" Plan2Explore's exploration signal, ported to observation space.

    K one-step forward models f_k(obs_t, a_t) -> obs_{t+1}, trained on real
    replay transitions. The intrinsic reward is P2E's Eq. 10: the empirical
    variance of the ensemble MEANS,
        D(s, a) = 1/(K-1) * sum_k (mu_k(s,a) - mu')^2,  mu' = mean_k mu_k,
    averaged over output dims. It is epistemic by construction: all heads
    converge to the conditional mean where data is plentiful, so D -> 0 even
    under stochastic dynamics -- the self-annealing property the project's
    earlier draws/RND signals lacked.

    Labeled deviation from P2E: they predict next latent embeddings inside
    the Dreamer graph; we predict next observations. With 37-dim state obs
    this is lossless in spirit, trains straight off replay, and keeps the
    v3 finding available: `obs_slice` restricts the disagreement to object
    dims so novelty means 'objects predicted to move somewhere new'.

    Implementation: all K heads live in stacked parameter tensors and every
    forward runs the whole ensemble in fused einsum passes -- one kernel
    per layer for all heads, no per-head loop.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DisagreementEnsemble(nn.Module):
    def __init__(self, obs_dim, act_dim, k=5, hidden=256, layers=2,
                 lr=3e-4, obs_slice=None, device='cpu'):
        super().__init__()
        assert layers == 2, 'stacked implementation is fixed at 2 layers'
        self.k = k
        self.obs_slice = (slice(obs_slice[0], obs_slice[1])
                          if obs_slice else None)
        out_dim = (obs_slice[1] - obs_slice[0]) if obs_slice else obs_dim
        d_in = obs_dim + act_dim

        def w(fan_in, *shape):
            t = torch.empty(*shape)
            nn.init.uniform_(t, -1 / math.sqrt(fan_in), 1 / math.sqrt(fan_in))
            return nn.Parameter(t)

        self.w1, self.b1 = w(d_in, k, d_in, hidden), w(d_in, k, hidden)
        self.w2, self.b2 = w(hidden, k, hidden, hidden), w(hidden, k, hidden)
        self.w3, self.b3 = w(hidden, k, hidden, out_dim), w(hidden, k, out_dim)
        self.to(device)
        self.device = device
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

    def _forward_all(self, x):
        """ x (B, d_in) -> predictions (K, B, out_dim), all heads fused. """
        h = F.silu(torch.einsum('bi,kih->kbh', x, self.w1)
                   + self.b1.unsqueeze(1))
        h = F.silu(torch.einsum('kbh,khj->kbj', h, self.w2)
                   + self.b2.unsqueeze(1))
        return torch.einsum('kbj,kjo->kbo', h, self.w3) + self.b3.unsqueeze(1)

    def _targets(self, next_obs):
        return next_obs[:, self.obs_slice] if self.obs_slice else next_obs

    def train_batch(self, obs, act, next_obs):
        to = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32,
                                       device=self.device)
        obs, act, next_obs = to(obs), to(act), to(next_obs)
        x = torch.cat([obs, act], -1)
        tgt = self._targets(next_obs).unsqueeze(0)            # (1, B, D)
        preds = self._forward_all(x)                          # (K, B, D)
        # independent bootstrap masks per head so the heads disagree from
        # data noise, not only from initialization
        mask = (torch.rand(self.k, len(x), 1,
                           device=self.device) < 0.8).float()
        loss = (mask * (preds - tgt) ** 2).mean() * self.k
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return {'ensemble_loss': float(loss.item()) / self.k}

    @torch.no_grad()
    def disagreement(self, obs, act):
        """ obs (B, obs_dim), act (B, act_dim) tensors -> (B,) intrinsic
            reward. Variance across ensemble means, averaged over dims. """
        x = torch.cat([obs, act], -1)
        preds = self._forward_all(x)                          # (K, B, D)
        return preds.var(dim=0, unbiased=True).mean(-1)       # (B,)