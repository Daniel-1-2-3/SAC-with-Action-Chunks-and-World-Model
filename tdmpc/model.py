""" TD-MPC2 networks and the symlog two-hot machinery, in PyTorch.

    Ported from github.com/nicklashansen/tdmpc2 (common/layers.py,
    common/math.py, common/init.py, common/world_model.py), state-observation
    variant, checked module by module against that source.

    NOTHING here decodes back to observation space. The encoder maps a raw
    observation to a latent, the dynamics stay in that latent, and the reward
    head, the Q ensemble and the policy prior all read the latent directly.

    Reward and value are predicted as two-hot distributions over symlog-spaced
    bins, matching the Dreamer configuration this replaced, so the reward term
    and the value term of a chunk score stay in comparable units.

    Two additions over the reference, both off by default:
      num_dyn > 1   an ENSEMBLE of dynamics heads. The rollout uses their mean;
                    their spread is the disagreement bonus of the explore arm
                    (Pathak et al. 2019, "Self-Supervised Exploration via
                    Disagreement").
      categorical   the SimNorm latent read as G independent categoricals, so
                    a next-latent has a log-likelihood. The optimistic arm's
                    RBMLE loss needs that (Mete et al. 2026, "Optimistic World
                    Models"). """

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
# layers (common/layers.py, common/init.py)
# --------------------------------------------------------------------------

def weight_init(m):
    """ TD-MPC2's init: truncated normal (std 0.02) on every Linear, zero
        bias. """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class SimNorm(nn.Module):
    """ Simplicial normalization: split the latent into groups of `dim` and
        softmax each group. This is what keeps TD-MPC2's latent bounded
        without a reconstruction loss -- the latent cannot blow up or collapse
        to a constant scale, so the consistency loss alone is enough to keep
        it meaningful. Each group is also, read literally, a categorical
        distribution over `dim` classes; the optimistic loss uses that. """

    def __init__(self, dim=8):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        shape = x.shape
        x = x.view(*shape[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shape)


class NormedLinear(nn.Module):
    """ Linear -> (dropout) -> LayerNorm -> activation, TD-MPC2's block. """

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
    """ layers.mlp: num_layers NormedLinear hidden layers (dropout on the
        first only), then either a plain Linear or a NormedLinear with the
        given output activation. """
    layers, d = [], in_dim
    for i in range(num_layers):
        layers.append(NormedLinear(d, hidden_dim, dropout=dropout if i == 0 else 0.0))
        d = hidden_dim
    if out_act is None:
        layers.append(nn.Linear(d, out_dim))
    else:
        layers.append(NormedLinear(d, out_dim, act=out_act))
    return nn.Sequential(*layers)


class EnsembleMLP(nn.Module):
    """ E copies of `mlp(...)` with their weights stacked along a leading
        member axis, so all E members run in ONE batched matmul per layer
        (TD-MPC2's own Ensemble/vmap trick). Same architecture as mlp():
        num_layers NormedLinear blocks (Linear -> dropout on the first only
        -> LayerNorm -> Mish), then a plain Linear or a NormedLinear with
        out_act. Same init: trunc-normal 0.02, zero bias, LayerNorm at 1/0.

        forward(x): x is (B, in) (shared input for every member) or
        (E, B, in). Returns (E, B, out). `idx` restricts the forward to a
        subset of members, (len(idx), B, out) -- the others cost nothing. """

    def __init__(self, num, in_dim, hidden_dim, out_dim, num_layers,
                 out_act=None, dropout=0.0):
        super().__init__()
        self.num = num
        self.out_act = out_act
        self.dropout = dropout
        dims = [in_dim] + [hidden_dim] * num_layers + [out_dim]
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.ln_w = nn.ParameterList()
        self.ln_b = nn.ParameterList()
        self.normed = []
        for i in range(len(dims) - 1):
            w = torch.empty(num, dims[i], dims[i + 1])
            for e in range(num):
                nn.init.trunc_normal_(w[e], std=0.02)
            self.weights.append(nn.Parameter(w))
            self.biases.append(nn.Parameter(torch.zeros(num, 1, dims[i + 1])))
            normed = (i < num_layers) or (out_act is not None)
            self.normed.append(normed)
            if normed:
                self.ln_w.append(nn.Parameter(torch.ones(num, 1, dims[i + 1])))
                self.ln_b.append(nn.Parameter(torch.zeros(num, 1, dims[i + 1])))
        self.num_layers = num_layers

    def zero_output(self):
        nn.init.zeros_(self.weights[-1])

    def forward(self, x, idx=None):
        if x.dim() == 2:
            n = self.num if idx is None else len(idx)
            x = x.unsqueeze(0).expand(n, -1, -1)
        j = 0
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            if idx is not None:
                w, b = w[idx], b[idx]
            x = torch.baddbmm(b, x, w)
            if self.normed[i]:
                if i == 0 and self.dropout and self.training:
                    x = F.dropout(x, self.dropout, training=True)
                lw, lb = self.ln_w[j], self.ln_b[j]
                if idx is not None:
                    lw, lb = lw[idx], lb[idx]
                mu = x.mean(-1, keepdim=True)
                var = x.var(-1, unbiased=False, keepdim=True)
                x = (x - mu) * torch.rsqrt(var + 1e-5) * lw + lb
                x = F.mish(x) if (i < self.num_layers) else self.out_act(x)
                j += 1
        return x


# --------------------------------------------------------------------------
# world model (common/world_model.py)
# --------------------------------------------------------------------------

class TDMPC2Nets(nn.Module):
    """ Encoder, latent dynamics, reward head, Q ensemble and policy prior.

        Every head takes a LATENT. The policy prior is what lets a score look
        past the candidate chunk: the follow-on actions are sampled at the
        imagined latent, so extending the horizon costs one dynamics step per
        action and no decode at all. """

    def __init__(self, obs_dim, action_dim, latent_dim=512, mlp_dim=512,
                 enc_dim=256, enc_layers=2, simnorm_dim=8, num_q=5,
                 dropout=0.01, num_bins=101, vmin=-10.0, vmax=10.0, num_dyn=1):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.simnorm_dim = simnorm_dim
        self.num_q = num_q
        self.num_dyn = max(1, int(num_dyn))
        self.num_bins = num_bins
        self.vmin = vmin
        self.vmax = vmax
        self.log_std_min, self.log_std_max = -10.0, 2.0

        # layers.enc: max(num_enc_layers - 1, 1) hidden layers of enc_dim,
        # then a NormedLinear to the latent with SimNorm as its activation.
        self.encoder = mlp(obs_dim, enc_dim, latent_dim, max(enc_layers - 1, 1),
                           out_act=SimNorm(simnorm_dim))
        # Dynamics heads and Q heads are stacked ensembles: one batched
        # matmul per layer for all members instead of a Python loop of
        # separate MLPs. Identical math and init; only the launch count
        # changes.
        self.dynamics = EnsembleMLP(self.num_dyn, latent_dim + action_dim,
                                    mlp_dim, latent_dim, 2,
                                    out_act=SimNorm(simnorm_dim))
        self.reward = mlp(latent_dim + action_dim, mlp_dim, num_bins, 2)
        self.pi_net = mlp(latent_dim, mlp_dim, 2 * action_dim, 2)
        self.Qs = EnsembleMLP(num_q, latent_dim + action_dim, mlp_dim,
                              num_bins, 2, dropout=dropout)
        self.register_buffer(
            'bins', torch.linspace(vmin, vmax, num_bins), persistent=False)

        # world_model.py: trunc-normal init everywhere, then the OUTPUT layer
        # of the reward head and of every Q head is zeroed, so both start out
        # predicting the middle bin (0) instead of noise.
        self.apply(weight_init)
        nn.init.zeros_(self.reward[-1].weight)
        self.Qs.zero_output()

        # Target copy lives here rather than on the trainer so it travels with
        # state_dict(). Updated by soft_update_target_q, never by the
        # optimizer. Created AFTER init so it inherits the zeroed outputs.
        self.Qs_target = deepcopy(self.Qs)
        for p in self.Qs_target.parameters():
            p.requires_grad_(False)

    def train(self, mode=True):
        """ world_model.py overrides train() so the TARGET Q heads stay in
            eval mode -- their dropout must never be active, even mid-update,
            or the TD target picks up dropout noise. """
        super().train(mode)
        self.Qs_target.train(False)
        return self

    # --------------------------------------------------------------- heads

    def encode(self, obs):
        return self.encoder(obs)

    def next_all(self, z, action):
        """ Every dynamics head's prediction, (num_dyn, B, latent_dim). """
        return self.dynamics(torch.cat([z, action], dim=-1))

    def next(self, z, action):
        """ Next latent. With one head this is that head; with an ensemble it
            is the mean, which stays on the SimNorm simplices because they
            are convex. """
        preds = self.next_all(z, action)
        return preds[0] if self.num_dyn == 1 else preds.mean(0)

    def reward_logits(self, z, action):
        return self.reward(torch.cat([z, action], dim=-1))

    def reward_pred(self, z, action):
        """ Raw predicted reward, (B, 1). """
        return from_two_hot(self.reward_logits(z, action), self.bins)

    def pi(self, z):
        """ Returns (mu, action, log_prob), all tanh-squashed to [-1, 1].
            `mu` is the deterministic mean action.

            world_model.py additionally reports a "scaled entropy", which is
            -log_prob multiplied by action_dim (its entropy_scale reduces to
            exactly that outside the multitask path). pi_loss uses the scaled
            one; it is recoverable here as -log_prob * action_dim. """
        mu, log_std = self.pi_net(z).chunk(2, dim=-1)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * \
            (torch.tanh(log_std) + 1)
        eps = torch.randn_like(mu)
        log_prob = gaussian_logprob(eps, log_std)
        mu, action, log_prob = squash(mu, mu + eps * log_std.exp(), log_prob)
        return mu, action, log_prob

    def q_logits(self, z, action, target=False, idx=None):
        """ (num_q, B, num_bins), or (len(idx), B, num_bins) for a subset of
            heads -- the unselected heads are not computed. """
        heads = self.Qs_target if target else self.Qs
        return heads(torch.cat([z, action], dim=-1), idx=idx)

    def q_values(self, z, action, target=False, idx=None):
        """ Raw Q per ensemble member, (num_q, B, 1). """
        return from_two_hot(self.q_logits(z, action, target, idx), self.bins)

    def q_subset(self, z, action, reduce='min', target=False):
        """ world_model.py Q(return_type='min'|'avg'): two RANDOM heads,
            reduced. 'min' is the pessimistic TD target; 'avg' is what the
            policy-prior loss and TD-MPC2's own planner use. (B, 1). """
        idx = torch.randperm(self.num_q, device=z.device)[:2]
        q = self.q_values(z, action, target, idx=idx)
        return q.min(0).values if reduce == 'min' else q.mean(0)

    # ------------------------------------------- categorical view of a latent

    def as_categorical(self, z):
        """ (B, latent_dim) on the SimNorm simplices -> (B, G, simnorm_dim)
            per-group class probabilities. No computation: it is a reshape. """
        return z.view(*z.shape[:-1], -1, self.simnorm_dim)

    def sample_latent(self, z_probs):
        """ Draw one class per group and return the one-hot latent with a
            straight-through gradient to the probabilities (Dreamer's trick),
            plus the log-likelihood of the draw, summed over groups.

            Used only by the optimistic loss, which needs a SAMPLED next
            latent whose likelihood the dynamics can be pushed on. Scoring
            and the consistency loss stay deterministic. """
        p = self.as_categorical(z_probs)
        idx = torch.multinomial(p.reshape(-1, self.simnorm_dim), 1).view(*p.shape[:-1])
        onehot = F.one_hot(idx, self.simnorm_dim).to(p.dtype)
        sample = onehot + p - p.detach()
        logp = torch.log(p.clamp_min(1e-8)).gather(-1, idx.unsqueeze(-1)).squeeze(-1).sum(-1, keepdim=True)
        return sample.view(*z_probs.shape), logp

    def latent_entropy(self, z_probs):
        """ Sum over groups of the categorical entropy, (B, 1). """
        p = self.as_categorical(z_probs)
        return -(p * torch.log(p.clamp_min(1e-8))).sum(-1).sum(-1, keepdim=True)


def convert_legacy_state_dict(sd, num_q, num_dyn):
    """ Map a checkpoint saved by the per-head ModuleList layout
        (Qs.{i}.{layer}.linear.weight, dynamics.{i}...) onto the stacked
        EnsembleMLP layout. Already-stacked checkpoints pass through. """
    if not any(k.startswith('Qs.0.') for k in sd):
        return sd
    out = {k: v for k, v in sd.items()
           if not k.startswith(('Qs.', 'Qs_target.', 'dynamics.'))}
    for name, num in (('Qs', num_q), ('Qs_target', num_q), ('dynamics', num_dyn)):
        layers = sorted({int(k.split('.')[2]) for k in sd if k.startswith(name + '.0.')})
        j = 0
        for li, layer in enumerate(layers):
            pre = lambda e: f'{name}.{e}.{layer}.'
            has_ln = pre(0) + 'ln.weight' in sd
            lin = 'linear.' if has_ln else ''
            out[f'{name}.weights.{li}'] = torch.stack(
                [sd[pre(e) + lin + 'weight'].t() for e in range(num)])
            out[f'{name}.biases.{li}'] = torch.stack(
                [sd[pre(e) + lin + 'bias'][None] for e in range(num)])
            if has_ln:
                out[f'{name}.ln_w.{j}'] = torch.stack(
                    [sd[pre(e) + 'ln.weight'][None] for e in range(num)])
                out[f'{name}.ln_b.{j}'] = torch.stack(
                    [sd[pre(e) + 'ln.bias'][None] for e in range(num)])
                j += 1
    return out
