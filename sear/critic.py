""" SEAR's causal transformer critic (their Figure 1a, adapted from
    TOP-ERL): input tokens are [state, a_1, ..., a_N] with a causal mask;
    the head over action token i emits Q^(i)(s, a_{1:i}) -- one Q per chunk
    prefix, so a single forward pass yields all N multi-horizon predictions
    and the k-step prediction provably cannot see actions beyond k.

    Sizes (d_model 256, 3 layers, 4 heads) are assumptions: the SEAR
    preprint's appendix values were not fully extracted. Marked for tuning.
"""
import torch
import torch.nn as nn


class CausalChunkCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, chunk_len, d_model=256, layers=3,
                 heads=4):
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
            nn.Linear(d_model, 1))
        mask = torch.triu(torch.ones(chunk_len + 1, chunk_len + 1), 1).bool()
        self.register_buffer('causal_mask', mask)

    def forward(self, obs, actions):
        """ obs (B, obs_dim), actions (B, N, act_dim) ->
            q (B, N): q[:, k-1] = Q^(k)(s, a_{1:k}). """
        tok = torch.cat([self.state_in(obs).unsqueeze(1),
                         self.act_in(actions)], 1) + self.pos
        h = self.tf(tok, mask=self.causal_mask)
        return self.q_head(h[:, 1:]).squeeze(-1)


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

    @torch.no_grad()
    def target_min(self, obs, actions):
        return torch.minimum(self.t1(obs, actions), self.t2(obs, actions))

    @torch.no_grad()
    def soft_update(self, tau):
        for online, target in [(self.q1, self.t1), (self.q2, self.t2)]:
            for p, tp in zip(online.parameters(), target.parameters()):
                tp.mul_(1 - tau).add_(tau * p)
