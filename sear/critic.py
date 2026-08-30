""" SEAR's causal transformer critic (their Figure 1a, adapted from
    TOP-ERL): input tokens are [state, a_1, ..., a_N] with a causal mask;
    the head over action token i emits Q^(i)(s, a_{1:i}) -- one Q per chunk
    prefix, so a single forward pass yields all N multi-horizon predictions
    and the k-step prediction provably cannot see actions beyond k.

    Appendix C fidelity: transformer hidden dim 512, 16 heads, 2 blocks;
    the critic is DISTRIBUTIONAL -- 101 bins over a fixed value range,
    trained with two-hot cross-entropy, Q = expectation over bins. Their
    range is [0, 1000] for Metaworld's positive returns; ours must cover
    OGBench's negative Q scale and is a config value (vmin/vmax).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalChunkCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, chunk_len, d_model=512, layers=2,
                 heads=16, bins=101, vmin=-260.0, vmax=10.0):
        super().__init__()
        self.chunk_len = chunk_len
        self.state_in = nn.Linear(obs_dim, d_model)
        self.act_in = nn.Linear(act_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, chunk_len + 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, heads, dim_feedforward=4 * d_model, batch_first=True,
            norm_first=True, activation='gelu', dropout=0.0)
        self.tf = nn.TransformerEncoder(layer, layers)
        self.q_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.SiLU(),
            nn.Linear(d_model, bins))
        mask = torch.triu(torch.ones(chunk_len + 1, chunk_len + 1), 1).bool()
        self.register_buffer('causal_mask', mask)
        self.register_buffer('bin_values', torch.linspace(vmin, vmax, bins))

    def logits(self, obs, actions):
        """ -> (B, N, bins): distributional logits per prefix. """
        tok = torch.cat([self.state_in(obs).unsqueeze(1),
                         self.act_in(actions)], 1) + self.pos
        h = self.tf(tok, mask=self.causal_mask)
        return self.q_head(h[:, 1:])

    def forward(self, obs, actions):
        """ -> (B, N): expected Q per prefix. """
        p = F.softmax(self.logits(obs, actions), -1)
        return (p * self.bin_values).sum(-1)

    def two_hot(self, y):
        """ y (...,) scalar targets -> (..., bins) two-hot distribution. """
        y = y.clamp(self.bin_values[0], self.bin_values[-1])
        bins = self.bin_values
        idx = torch.searchsorted(bins, y.detach().contiguous()).clamp(
            1, len(bins) - 1)
        lo, hi = bins[idx - 1], bins[idx]
        w_hi = ((y - lo) / (hi - lo + 1e-8)).clamp(0, 1)
        out = torch.zeros(*y.shape, len(bins), device=y.device)
        out.scatter_(-1, (idx - 1).unsqueeze(-1), (1 - w_hi).unsqueeze(-1))
        out.scatter_(-1, idx.unsqueeze(-1), w_hi.unsqueeze(-1))
        return out


class TwinCritic(nn.Module):
    """ Two independent transformer critics (SEAR's j in {1,2} min), plus
        frozen target copies with soft updates. """

    def __init__(self, obs_dim, act_dim, chunk_len, **kw):
        super().__init__()
        self.q1 = CausalChunkCritic(obs_dim, act_dim, chunk_len, **kw)
        self.q2 = CausalChunkCritic(obs_dim, act_dim, chunk_len, **kw)
        self.t1 = CausalChunkCritic(obs_dim, act_dim, chunk_len, **kw)
        self.t2 = CausalChunkCritic(obs_dim, act_dim, chunk_len, **kw)
        self.t1.load_state_dict(self.q1.state_dict())
        self.t2.load_state_dict(self.q2.state_dict())
        for p in list(self.t1.parameters()) + list(self.t2.parameters()):
            p.requires_grad_(False)

    def online(self, obs, actions):
        return self.q1(obs, actions), self.q2(obs, actions)

    def online_logits(self, obs, actions):
        return self.q1.logits(obs, actions), self.q2.logits(obs, actions)

    @torch.no_grad()
    def target_min(self, obs, actions):
        return torch.minimum(self.t1(obs, actions), self.t2(obs, actions))

    @torch.no_grad()
    def soft_update(self, tau):
        for online, target in [(self.q1, self.t1), (self.q2, self.t2)]:
            for p, tp in zip(online.parameters(), target.parameters()):
                tp.mul_(1 - tau).add_(tau * p)
