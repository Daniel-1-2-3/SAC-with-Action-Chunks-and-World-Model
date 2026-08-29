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
"""
import numpy as np
import torch
import torch.nn as nn


class DisagreementEnsemble(nn.Module):
    def __init__(self, obs_dim, act_dim, k=5, hidden=256, layers=2,
                 lr=3e-4, obs_slice=None, device='cpu'):
        super().__init__()
        self.k = k
        self.obs_slice = (slice(obs_slice[0], obs_slice[1])
                          if obs_slice else None)
        out_dim = (obs_slice[1] - obs_slice[0]) if obs_slice else obs_dim

        def head():
            seq, d = [], obs_dim + act_dim
            for _ in range(layers):
                seq += [nn.Linear(d, hidden), nn.SiLU()]
                d = hidden
            seq += [nn.Linear(d, out_dim)]
            return nn.Sequential(*seq)

        self.heads = nn.ModuleList([head() for _ in range(self.k)])
        self.to(device)
        self.device = device
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

    def _targets(self, next_obs):
        return next_obs[:, self.obs_slice] if self.obs_slice else next_obs

    def train_batch(self, obs, act, next_obs):
        to = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32,
                                       device=self.device)
        obs, act, next_obs = to(obs), to(act), to(next_obs)
        x = torch.cat([obs, act], -1)
        tgt = self._targets(next_obs)
        # independent bootstrap masks per head so the heads disagree from
        # data noise, not only from initialization
        loss = 0.0
        for h in self.heads:
            mask = (torch.rand(len(x), 1, device=self.device) < 0.8).float()
            loss = loss + (mask * (h(x) - tgt) ** 2).mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return {'ensemble_loss': float(loss.item()) / self.k}

    @torch.no_grad()
    def disagreement(self, obs, act):
        """ obs (B, obs_dim), act (B, act_dim) tensors -> (B,) intrinsic
            reward. Variance across ensemble means, averaged over dims. """
        x = torch.cat([obs, act], -1)
        preds = torch.stack([h(x) for h in self.heads])       # (K, B, D)
        return preds.var(dim=0, unbiased=True).mean(-1)       # (B,)
