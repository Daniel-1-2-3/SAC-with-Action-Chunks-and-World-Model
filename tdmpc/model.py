""" TD-MPC2 networks and the symlog two-hot machinery, in PyTorch.

    Ported from github.com/nicklashansen/tdmpc2 (common/layers.py,
    common/math.py, common/world_model.py), state-observation variant.

    NOTHING here decodes back to observation space. The encoder maps a raw
    observation to a latent, the dynamics stay in that latent, and the reward
    head, the Q ensemble and the policy prior all read the latent directly.
    That is the whole reason for the swap: the previous scorer had to decode
    an imagined latent before the observation-space critic could look at it,
    so every score carried decoder error on top of dynamics error.

    Reward and value are predicted as two-hot distributions over symlog-spaced
    bins, matching the Dreamer configuration this replaces, so the reward term
    and the value term of a chunk score stay in comparable units. """

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# math (common/math.py)
# --------------------------------------------------------------------------

def symlog(x):
    return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def two_hot(x, vmin, vmax, num_bins):
    """ Raw value -> two-hot target over symlog-spaced bins.

        x: (..., 1). Returns (..., num_bins). The value is symlogged first, so
        the bin grid is uniform in symlog space and covers a huge dynamic
        range with few bins -- symexp(10) is ~22000, which is why the default
        [-10, 10] holds any return this task can produce. """
    bin_size = (vmax - vmin) / (num_bins - 1)
    x = torch.clamp(symlog(x), vmin, vmax).squeeze(-1)
    bin_idx = torch.floor((x - vmin) / bin_size)
    bin_offset = ((x - vmin) / bin_size - bin_idx).unsqueeze(-1)
    bin_idx = bin_idx.long().unsqueeze(-1).clamp(0, num_bins - 1)
    soft = torch.zeros(*x.shape, num_bins, device=x.device, dtype=x.dtype)
    soft.scatter_(-1, bin_idx, 1 - bin_offset)
    soft.scatter_(-1, (bin_idx + 1).clamp_max(num_bins - 1), bin_offset)
    return soft


def from_two_hot(logits, bins):
    """ Two-hot logits -> raw value. Expectation under the softmax, then
        symexp back out of symlog space. Returns (..., 1). """
    probs = F.softmax(logits, dim=-1)
    return symexp((probs * bins).sum(-1, keepdim=True))


def soft_ce(logits, target, vmin, vmax, num_bins):
    """ Cross-entropy of a two-hot prediction against a raw-valued target.
        Returns (..., 1) so callers can mask it per element. """
    pred = F.log_softmax(logits, dim=-1)
    tgt = two_hot(target, vmin, vmax, num_bins)
    return -(tgt * pred).sum(-1, keepdim=True)


def gaussian_logprob(eps, log_std):
    residual = (-0.5 * eps.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * eps.shape[-1] * torch.log(
        torch.tensor(2 * torch.pi, device=eps.device, dtype=eps.dtype))


def squash(mu, pi, log_pi):
    """ tanh squashing with the matching log-density correction. """
    mu = torch.tanh(mu)
    pi = torch.tanh(pi)
    log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


# --------------------------------------------------------------------------
# layers (common/layers.py)
# --------------------------------------------------------------------------

class SimNorm(nn.Module):
    """ Simplicial normalization: split the latent into groups of `dim` and
        softmax each group. This is what keeps TD-MPC2's latent bounded
        without a reconstruction loss -- the latent cannot blow up or collapse
        to a constant scale, so the consistency loss alone is enough to keep
        it meaningful. """

    def __init__(self, dim=8):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        shape = x.shape
        x = x.view(*shape[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shape)


class NormedLinear(nn.Module):
    """ Linear -> LayerNorm -> activation, TD-MPC2's basic block. """

    def __init__(self, in_dim, out_dim, act=None, dropout=0.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)
        self.act = act if act is not None else nn.Mish(inplace=False)
        self.dropout = nn.Dropout(dropout, inplace=False) if dropout else None

    def forward(self, x):
        x = self.linear(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.act(self.ln(x))


def mlp(in_dim, hidden_dim, out_dim, num_layers, out_act=None, dropout=0.0):
    layers, d = [], in_dim
    for i in range(num_layers):
        layers.append(NormedLinear(d, hidden_dim, dropout=dropout if i == 0 else 0.0))
        d = hidden_dim
    if out_act is None:
        layers.append(nn.Linear(d, out_dim))
    else:
        layers.append(NormedLinear(d, out_dim, act=out_act))
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------
# world model (common/world_model.py)
# --------------------------------------------------------------------------

class TDMPC2Nets(nn.Module):
    """ Encoder, latent dynamics, reward head, Q ensemble and policy prior.

        Every head takes a LATENT. The policy prior is what lets the score
        look past the candidate chunk: the follow-on actions are sampled at
        the imagined latent, so extending the horizon costs one dynamics step
        per action and no decode at all. """

    def __init__(self, obs_dim, action_dim, latent_dim=512, mlp_dim=512,
                 enc_layers=2, simnorm_dim=8, num_q=5, dropout=0.01,
                 num_bins=101, vmin=-10.0, vmax=10.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.num_q = num_q
        self.num_bins = num_bins
        self.vmin = vmin
        self.vmax = vmax
        self.log_std_min, self.log_std_max = -10.0, 2.0

        self.encoder = mlp(obs_dim, mlp_dim, latent_dim, enc_layers,
                           out_act=SimNorm(simnorm_dim))
        self.dynamics = mlp(latent_dim + action_dim, mlp_dim, latent_dim, 2,
                            out_act=SimNorm(simnorm_dim))
        self.reward = mlp(latent_dim + action_dim, mlp_dim, num_bins, 2)
        self.pi_net = mlp(latent_dim, mlp_dim, 2 * action_dim, 2)
        self.Qs = nn.ModuleList([
            mlp(latent_dim + action_dim, mlp_dim, num_bins, 2, dropout=dropout)
            for _ in range(num_q)])
        # Target copy lives here rather than on the trainer so it travels with
        # state_dict(). Updated by soft_update_target_q, never by the
        # optimizer.
        self.Qs_target = deepcopy(self.Qs)
        for p in self.Qs_target.parameters():
            p.requires_grad_(False)
        self.register_buffer(
            'bins', torch.linspace(vmin, vmax, num_bins), persistent=False)

    def encode(self, obs):
        return self.encoder(obs)

    def next(self, z, action):
        return self.dynamics(torch.cat([z, action], dim=-1))

    def reward_pred(self, z, action):
        """ Raw predicted reward, (B, 1). """
        return from_two_hot(self.reward_logits(z, action), self.bins)

    def reward_logits(self, z, action):
        return self.reward(torch.cat([z, action], dim=-1))

    def pi(self, z):
        """ Returns (mu, action, log_prob), all tanh-squashed to [-1, 1].
            `mu` is the deterministic mean action. """
        mu, log_std = self.pi_net(z).chunk(2, dim=-1)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * \
            (torch.tanh(log_std) + 1)
        eps = torch.randn_like(mu)
        log_prob = gaussian_logprob(eps, log_std)
        mu, action, log_prob = squash(mu, mu + eps * log_std.exp(), log_prob)
        return mu, action, log_prob

    def q_logits(self, z, action, target=False):
        """ (num_q, B, num_bins). """
        heads = self.Qs_target if target else self.Qs
        h = torch.cat([z, action], dim=-1)
        return torch.stack([q(h) for q in heads], dim=0)

    def q_values(self, z, action, target=False):
        """ Raw Q per ensemble member, (num_q, B, 1). """
        return from_two_hot(self.q_logits(z, action, target), self.bins)
