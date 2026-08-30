""" PyTorch world model: a faithful port of the JAX DreamerV3-style RSSM the
    project previously used through dreamer/wm_bridge.py, specialized to
    state observations (37-dim vectors, no images).

    Mirrored exactly from the old configs.yaml agent block:
      deter=512, hidden=512, stoch=32 categoricals x 16 classes,
      unimix=0.01, free_nats=1.0, SiLU activations, RMSNorm,
      symlog on encoder input and decoder/reward targets,
      loss scales: rec 1.0, rew 1.0, con 1.0, dyn 1.0, rep 0.1.

    Labeled deviations from the JAX original (each pragmatic, none load-
    bearing for the disagreement/imagination roles this model plays here):
      - Standard GRUCell instead of the 8-block block-diagonal GRU.
      - Reward head is symlog-MSE instead of two-hot discretized regression.
      - AdamW(lr=1e-4, clip=100) instead of LaProp with AGC and warmup.

    Performance notes: the GRU recurrence is inherently sequential, but
    everything hoistable is hoisted out of the time loop -- the encoder runs
    once over (B*T), and all KL math is vectorized over (B, T) after the
    loop, so the loop body is just the recurrent linears. use_compile=True
    additionally fuses the whole train step with torch.compile (falls back
    silently if unsupported); first call pays a one-off compilation delay.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def mlp(inp, units, layers, out=None):
    seq = []
    d = inp
    for _ in range(layers):
        seq += [nn.Linear(d, units), nn.RMSNorm(units), nn.SiLU()]
        d = units
    if out is not None:
        seq += [nn.Linear(d, out)]
    return nn.Sequential(*seq)


class WorldModel(nn.Module):
    def __init__(self, obs_dim, act_dim, deter=512, hidden=512, stoch=32,
                 classes=16, units=512, enc_layers=3, dec_layers=3,
                 unimix=0.01, free_nats=1.0, lr=1e-4, device='cpu',
                 scales=None, use_compile=False):
        super().__init__()
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.deter, self.stoch, self.classes = deter, stoch, classes
        self.unimix, self.free_nats = unimix, free_nats
        self.feat_dim = deter + stoch * classes
        self.scales = scales or dict(rec=1.0, rew=1.0, con=1.0, dyn=1.0, rep=0.1)

        self.encoder = mlp(obs_dim, units, enc_layers, out=None)
        self.embed_dim = units
        # posterior: (deter, embed) -> stoch logits; prior: deter -> logits
        self.obs_logits = mlp(deter + units, hidden, 1, out=stoch * classes)
        self.img_logits = mlp(deter, hidden, 1, out=stoch * classes)
        self.pre_gru = mlp(stoch * classes + act_dim, hidden, 1, out=None)
        self.gru = nn.GRUCell(hidden, deter)
        self.decoder = mlp(self.feat_dim, units, dec_layers, out=obs_dim)
        self.reward_head = mlp(self.feat_dim, units, 2, out=1)
        self.cont_head = mlp(self.feat_dim, units, 2, out=1)
        self.to(device)
        self.device = device
        self.opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        self.last_feats = self.last_embeds = self.last_actions = None
        self._train_fn = self._train_impl
        if use_compile and hasattr(torch, 'compile'):
            try:
                self._train_fn = torch.compile(self._train_impl)
            except Exception as e:
                print(f'torch.compile unavailable ({e}); running eager')

    # ---------- distribution helpers ----------
    def _dist_probs(self, logits):
        logits = logits.reshape(*logits.shape[:-1], self.stoch, self.classes)
        probs = F.softmax(logits, -1)
        probs = (1 - self.unimix) * probs + self.unimix / self.classes
        return probs

    def _sample(self, probs):
        # straight-through one-hot sample
        idx = torch.distributions.Categorical(probs=probs).sample()
        onehot = F.one_hot(idx, self.classes).float()
        return onehot + probs - probs.detach()

    def _kl(self, p_probs, q_probs):
        # KL(p || q), summed over categoricals
        kl = (p_probs * (torch.log(p_probs + 1e-8) -
                         torch.log(q_probs + 1e-8))).sum(-1)
        return kl.sum(-1)

    # ---------- carries ----------
    def init(self, batch):
        return dict(
            deter=torch.zeros(batch, self.deter, device=self.device),
            stoch=torch.zeros(batch, self.stoch, self.classes,
                              device=self.device))

    def feat(self, carry):
        return torch.cat(
            [carry['deter'], carry['stoch'].flatten(1)], -1)

    def _core(self, carry, action):
        x = torch.cat([carry['stoch'].flatten(1), action], -1)
        deter = self.gru(self.pre_gru(x), carry['deter'])
        return deter

    def obs_step(self, carry, action, obs, is_first):
        """ One posterior step. obs: (B, obs_dim) raw; is_first: (B,) bool. """
        mask = (~is_first).float().unsqueeze(-1)
        carry = dict(deter=carry['deter'] * mask,
                     stoch=carry['stoch'] * mask.unsqueeze(-1))
        action = action * mask
        deter = self._core(carry, action)
        embed = self.encoder(symlog(obs))
        post_probs = self._dist_probs(self.obs_logits(
            torch.cat([deter, embed], -1)))
        stoch = self._sample(post_probs)
        prior_probs = self._dist_probs(self.img_logits(deter))
        return dict(deter=deter, stoch=stoch), post_probs, prior_probs

    def img_step(self, carry, action):
        deter = self._core(carry, action)
        prior_probs = self._dist_probs(self.img_logits(deter))
        stoch = self._sample(prior_probs)
        return dict(deter=deter, stoch=stoch)

    # ---------- public API (mirrors the old wm_bridge) ----------
    @torch.no_grad()
    def encode_step(self, carry, obs_np, prev_action_np, is_first_np):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        act = torch.as_tensor(prev_action_np, dtype=torch.float32,
                              device=self.device)
        first = torch.as_tensor(np.asarray(is_first_np, bool),
                                device=self.device)
        carry, _, _ = self.obs_step(carry, act, obs, first)
        return carry, self.feat(carry)

    def img_step_grad(self, carry, action):
        """ img_step WITHOUT no_grad -- used by the explorer's imagination
            training so gradients flow through the dynamics (Dreamer-style
            pathwise; the categorical sample is straight-through). """
        return self.img_step(carry, action)

    @torch.no_grad()
    def imagine(self, carry, actions):
        """ actions: (B, T, act_dim) tensor. Rolls the prior. Returns feats
            (B, T, feat_dim) and the final carry. """
        feats = []
        for t in range(actions.shape[1]):
            carry = self.img_step(carry, actions[:, t])
            feats.append(self.feat(carry))
        return carry, torch.stack(feats, 1)

    @torch.no_grad()
    def decode(self, feats):
        return symexp(self.decoder(feats))

    @torch.no_grad()
    def pred_reward(self, feats):
        return symexp(self.reward_head(feats)).squeeze(-1)

    # ---------- training ----------
    def train_batch(self, batch):
        """ batch: dict of numpy (B, T, ...): obs 'state', 'action',
            'reward', 'is_first', 'cont'. Returns metrics dict. """
        to = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32,
                                       device=self.device)
        obs, action = to(batch['state']), to(batch['action'])
        reward, cont = to(batch['reward']), to(batch['cont'])
        is_first = torch.as_tensor(np.asarray(batch['is_first'], bool),
                                   device=self.device)
        out, feats, embeds = self._train_fn(obs, action, reward, cont,
                                            is_first)
        # cached (detached) for the latent disagreement ensemble: inputs
        # (feat_t, action_{t+1}) -> target embed_{t+1}, P2E's training pairs
        self.last_feats, self.last_embeds = feats, embeds
        self.last_actions = action
        return {k: float(v.item()) for k, v in out.items()}

    def _train_impl(self, obs, action, reward, cont, is_first):
        B, T = obs.shape[:2]
        # hoisted: encode every timestep in one pass
        embeds = self.encoder(symlog(obs.reshape(B * T, -1))
                              ).reshape(B, T, -1)
        mask = (~is_first).float()                          # (B, T)
        deter = torch.zeros(B, self.deter, device=obs.device)
        stoch = torch.zeros(B, self.stoch * self.classes, device=obs.device)
        feats, post_l, prior_l = [], [], []
        for t in range(T):                # recurrence: irreducibly sequential
            m = mask[:, t:t + 1]
            deter = self.gru(self.pre_gru(
                torch.cat([stoch * m, action[:, t] * m], -1)), deter * m)
            pl = self.obs_logits(torch.cat([deter, embeds[:, t]], -1))
            post_l.append(pl)
            prior_l.append(self.img_logits(deter))
            stoch = self._sample(self._dist_probs(pl)).flatten(1)
            feats.append(torch.cat([deter, stoch], -1))
        feats = torch.stack(feats, 1)
        # hoisted: all KL math vectorized over (B, T)
        post_p = self._dist_probs(torch.stack(post_l, 1))
        prior_p = self._dist_probs(torch.stack(prior_l, 1))
        dyn = torch.clamp(self._kl(post_p.detach(), prior_p),
                          min=self.free_nats).mean()
        rep = torch.clamp(self._kl(post_p, prior_p.detach()),
                          min=self.free_nats).mean()
        rec = ((self.decoder(feats) - symlog(obs)) ** 2).mean()
        rew = ((self.reward_head(feats).squeeze(-1) - symlog(reward)) ** 2)
        # index 0 carries the fabricated pre-step reward; mask it out of the
        # reward loss exactly as the legacy pipeline did after the
        # fabricated-reward bug was found.
        rew = rew[:, 1:].mean()
        con = F.binary_cross_entropy_with_logits(
            self.cont_head(feats).squeeze(-1), cont)
        s = self.scales
        loss = (s['rec'] * rec + s['rew'] * rew + s['con'] * con +
                s['dyn'] * dyn + s['rep'] * rep)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 100.0)
        self.opt.step()
        return ({'loss/state': rec.detach(), 'loss/rew': rew.detach(),
                 'loss/con': con.detach(), 'loss/dyn': dyn.detach(),
                 'loss/rep': rep.detach()},
                feats.detach(), embeds.detach())