"""TD-MPC2 latent world model in JAX/flax -- port of this repo's PyTorch
tdmpc/model.py + tdmpc/agent.py (themselves ported from
nicklashansen/tdmpc2), restricted to exactly the paths the wm_explore arm
runs (its v5 configuration):

  PORTED
    nets        encoder (NormedLinear blocks + SimNorm latent), dynamics
                ENSEMBLE (num_dyn heads, vmapped), reward head and Q
                ensemble as symlog two-hot heads (zero-initialised output
                layers), max-entropy policy prior. trunc-normal(0.02)
                init, zero biases, LayerNorm eps 1e-5, Mish.
    update      the three losses on one latent rollout (consistency /
                reward / value), rho-weighted, valid-masked, /horizon;
                TD target = reward + gamma * mask * min of TWO RANDOM
                TARGET Q heads at a SAMPLED prior action; grad clip 20;
                Adam with enc_lr_scale on the encoder; then the prior's
                own update (Adam eps 1e-5, Q parameters held constant,
                running trimmed scale on Q, rho weights, valid_z) and the
                soft target-Q update.
    novelty     path_disagreement: per-step dynamics-ensemble variance
                (mean over latent dims = novelty 'mean') along an imagined
                path rolled through the ensemble MEAN, extended by
                rollout_chunks-1 prior chunks (mean action); 'path' or
                'end' reduction. data_disagreement EMA reference,
                ref_mode 'rollout' (real window actions, like for like)
                or 'step'.

  NOT PORTED (unused by wm_explore, or diagnostics only)
    q_mode='chunk' and everything under it, novelty='reward' weighting,
    tdmpc/diagnostics.py model_report, legacy checkpoint conversion.

get_config() carries the repo's tdmpc + wm_explore blocks (configs.yaml
values) so main.py can expose them as one --wm config flag.
"""

from functools import partial
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import nonpytree_field


# --------------------------------------------------------------------------
# math (tdmpc/model.py: symlog two-hot)
# --------------------------------------------------------------------------

def symlog(x):
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x):
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1.0)


def two_hot(x, vmin, vmax, num_bins):
    """Raw value (..., 1) -> two-hot target over symlog-spaced bins (..., bins)."""
    bin_size = (vmax - vmin) / (num_bins - 1)
    x = jnp.clip(symlog(x), vmin, vmax)[..., 0]
    pos = (x - vmin) / bin_size
    bin_idx = jnp.floor(pos)
    bin_offset = pos - bin_idx
    idx = jnp.clip(bin_idx.astype(jnp.int32), 0, num_bins - 1)
    lo = jax.nn.one_hot(idx, num_bins) * (1.0 - bin_offset)[..., None]
    hi = jax.nn.one_hot((idx + 1) % num_bins, num_bins) * bin_offset[..., None]
    return lo + hi


def from_two_hot(logits, bins):
    """Two-hot logits -> raw value (..., 1)."""
    probs = jax.nn.softmax(logits, axis=-1)
    return symexp(jnp.sum(probs * bins, axis=-1, keepdims=True))


def soft_ce(logits, target, vmin, vmax, num_bins):
    """Cross-entropy of a two-hot prediction against a raw target, (..., 1)."""
    pred = jax.nn.log_softmax(logits, axis=-1)
    tgt = two_hot(target, vmin, vmax, num_bins)
    return -jnp.sum(tgt * pred, axis=-1, keepdims=True)


def gaussian_logprob(eps, log_std):
    residual = jnp.sum(-0.5 * eps ** 2 - log_std, axis=-1, keepdims=True)
    return residual - 0.5 * eps.shape[-1] * jnp.log(2 * jnp.pi)


def squash(mu, pi, log_pi):
    mu = jnp.tanh(mu)
    pi = jnp.tanh(pi)
    log_pi = log_pi - jnp.sum(
        jnp.log(jax.nn.relu(1.0 - pi ** 2) + 1e-6), axis=-1, keepdims=True)
    return mu, pi, log_pi


# --------------------------------------------------------------------------
# modules (tdmpc/model.py layers)
# --------------------------------------------------------------------------

def mish(x):
    return x * jnp.tanh(jax.nn.softplus(x))


trunc_init = nn.initializers.truncated_normal(stddev=0.02)


class SimNorm(nn.Module):
    dim: int = 8

    def __call__(self, x):
        shape = x.shape
        x = x.reshape(*shape[:-1], -1, self.dim)
        x = jax.nn.softmax(x, axis=-1)
        return x.reshape(shape)


class NormedLinear(nn.Module):
    """Linear -> (dropout) -> LayerNorm(eps 1e-5) -> activation."""
    features: int
    act: Any = mish
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, train=False):
        x = nn.Dense(self.features, kernel_init=trunc_init)(x)
        if self.dropout > 0.0:
            x = nn.Dropout(self.dropout, deterministic=not train)(x)
        x = nn.LayerNorm(epsilon=1e-5)(x)
        return self.act(x)


class MLP(nn.Module):
    """tdmpc mlp(): num_layers NormedLinear hidden blocks (dropout on the
    first only), then a plain Linear or a NormedLinear with out_act.
    zero_out zeroes the output layer (reward head, Q heads)."""
    hidden_dim: int
    out_dim: int
    num_layers: int
    out_act: Any = None            # None = plain final Linear
    dropout: float = 0.0
    zero_out: bool = False

    @nn.compact
    def __call__(self, x, train=False):
        for i in range(self.num_layers):
            x = NormedLinear(self.hidden_dim,
                             dropout=self.dropout if i == 0 else 0.0)(x, train)
        if self.out_act is None:
            kinit = nn.initializers.zeros if self.zero_out else trunc_init
            return nn.Dense(self.out_dim, kernel_init=kinit)(x)
        return NormedLinear(self.out_dim, act=self.out_act)(x, train)


def ensemble(module_cls, num, **kwargs):
    """Stacked ensemble: E copies, shared input, output (E, ..., out)."""
    return nn.vmap(
        module_cls,
        variable_axes={'params': 0},
        split_rngs={'params': True, 'dropout': True},
        in_axes=None, out_axes=0, axis_size=num,
    )(**kwargs)


# --------------------------------------------------------------------------
# config: the repo's tdmpc + wm_explore blocks (configs.yaml defaults)
# --------------------------------------------------------------------------

def get_config():
    return ml_collections.ConfigDict(dict(
        # --- tdmpc block
        train_every=2,
        batch_size=256,
        horizon=5,
        latent_dim=512,
        mlp_dim=512,
        enc_dim=256,
        enc_layers=2,
        simnorm_dim=8,
        num_q=5,
        dropout=0.01,
        online_frac=0.5,       # balanced sampling for the MODEL's windows
        ref_mode='rollout',    # 'rollout' | 'step'
        num_bins=101,
        vmin=-10.0,
        vmax=10.0,
        lr=3e-4,
        enc_lr_scale=0.3,
        tau=0.01,
        rho=0.5,
        consistency_coef=20.0,
        reward_coef=0.1,
        value_coef=0.1,
        entropy_coef=1e-4,
        grad_clip_norm=20.0,
        # --- wm_explore block
        beta=1.0,
        bonus_scale='spread',  # 'spread' | 'unc'
        novelty='model',       # 'model' | 'none'
        novelty_at='path',     # 'path' | 'end'
        nu_cap=1.0,
        rollout_chunks=1,
        num_dyn=4,
        controller='bandit',   # 'bandit' | 'gate'
        bandit_window=40,
        bandit_c=1.0,
        progress_gate=True,
        progress_window=20,
        progress_tau=0.2,
        use_rel_unc=False,
    ))


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------

class WorldModel(flax.struct.PyTreeNode):
    """Owns the nets, both optimizers, the running Q scale and the
    data-disagreement reference. All jitted; static config in cfg."""
    rng: Any
    params: Any                # {'encoder','dynamics','reward','pi','Qs'}
    target_q_params: Any
    model_opt_state: Any
    pi_opt_state: Any
    scale: Any                 # running trimmed scale of Q (scalar)
    data_disagreement: Any     # EMA of disagreement on real data (scalar)
    cfg: Any = nonpytree_field()
    defs: Any = nonpytree_field()
    action_dim: int = nonpytree_field()
    chunk_len: int = nonpytree_field()
    gamma: float = nonpytree_field()

    # ------------------------------------------------------------- heads

    @property
    def bins(self):
        c = self.cfg
        return jnp.linspace(c['vmin'], c['vmax'], c['num_bins'])

    def encode(self, obs, params=None):
        p = self.params if params is None else params
        return self.defs['encoder'].apply({'params': p['encoder']}, obs)

    def next_all(self, z, action, params=None):
        """Every dynamics head, (num_dyn, B, latent)."""
        p = self.params if params is None else params
        return self.defs['dynamics'].apply(
            {'params': p['dynamics']}, jnp.concatenate([z, action], -1))

    def reward_logits(self, z, action, params=None):
        p = self.params if params is None else params
        return self.defs['reward'].apply(
            {'params': p['reward']}, jnp.concatenate([z, action], -1))

    def reward_pred(self, z, action, params=None):
        return from_two_hot(self.reward_logits(z, action, params), self.bins)

    def pi(self, z, rng, params=None):
        """(mu, action, log_prob, scaled_entropy), the reference's exact
        entropy_scale formula (tdmpc/model.py pi())."""
        p = self.params if params is None else params
        out = self.defs['pi'].apply({'params': p['pi']}, z)
        mu, log_std = jnp.split(out, 2, axis=-1)
        log_std_min, log_std_max = -10.0, 2.0
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (jnp.tanh(log_std) + 1)
        eps = jax.random.normal(rng, mu.shape)
        log_prob = gaussian_logprob(eps, log_std)
        scaled_log_prob = log_prob * self.action_dim
        mu, action, log_prob = squash(mu, mu + eps * jnp.exp(log_std), log_prob)
        entropy_scale = scaled_log_prob / (log_prob + 1e-8)
        return mu, action, log_prob, -log_prob * entropy_scale

    def q_logits(self, z, action, params=None, target=False, train=False, rng=None):
        p = (self.target_q_params if target
             else (self.params if params is None else params)['Qs'])
        x = jnp.concatenate([z, action], -1)
        if train:
            # train passed POSITIONALLY: flax's lifted vmap drops kwargs,
            # which would silently disable the Q heads' dropout
            return self.defs['Qs'].apply({'params': p}, x, True,
                                         rngs={'dropout': rng})
        return self.defs['Qs'].apply({'params': p}, x)

    def q_values(self, z, action, params=None, target=False):
        return from_two_hot(self.q_logits(z, action, params, target), self.bins)

    def q_subset(self, z, action, rng, reduce='min', target=False):
        """Two RANDOM heads reduced ('min' = TD target, 'avg' = prior loss)."""
        idx = jax.random.permutation(rng, self.cfg['num_q'])[:2]
        q = self.q_values(z, action, target=target)[idx]
        return q.min(0) if reduce == 'min' else q.mean(0)

    # ------------------------------------------------------------ novelty

    def path_disagreement(self, z, actions, rng):
        """(B,) disagreement of an imagined path: per-step ensemble variance
        (mean over latent dims), rolled through the ensemble MEAN, then
        rollout_chunks-1 prior chunks of MEAN actions. 'path' = mean over
        steps, 'end' = last step."""
        steps = []
        for k in range(actions.shape[1]):
            a = actions[:, k]
            preds = self.next_all(z, a)
            steps.append(preds.var(0).mean(-1))
            z = preds.mean(0)
        for _ in range(self.cfg['rollout_chunks'] - 1):
            for _ in range(self.chunk_len):
                rng, key = jax.random.split(rng)
                a = self.pi(z, key)[0]           # mean action
                preds = self.next_all(z, a)
                steps.append(preds.var(0).mean(-1))
                z = preds.mean(0)
        if self.cfg['novelty_at'] == 'end':
            return steps[-1]
        return jnp.stack(steps, 0).mean(0)

    # ----------------------------------------------------------- training

    def _td_target(self, batch, rng):
        """reward + gamma * mask * min of two random TARGET heads at a
        SAMPLED prior action, whole (B, H) batch, one head pair."""
        B, H = batch['reward'].shape[:2]
        next_z = self.encode(batch['next_obs'])
        flat_z = next_z.reshape(B * H, -1)
        rng, pi_key, q_key = jax.random.split(rng, 3)
        a = self.pi(flat_z, pi_key)[1]
        q = self.q_subset(flat_z, a, q_key, reduce='min', target=True)
        return batch['reward'] + self.gamma * batch['mask'] * q.reshape(B, H, 1), next_z

    def _losses(self, params, batch, next_z_real, td_targets, rng):
        """consistency + reward + value on one latent rollout, rho-weighted,
        valid-masked, each /horizon (tdmpc/agent.py _losses)."""
        c = self.cfg
        action, reward, valid = batch['action'], batch['reward'], batch['valid']
        horizon = action.shape[1]
        z = self.encode(batch['obs'][:, 0], params)
        zs = [z]
        consistency_loss = reward_loss = value_loss = 0.0
        rho = 1.0
        dis0 = None
        for t in range(horizon):
            w = valid[:, t]
            reward_loss = reward_loss + rho * jnp.mean(w * soft_ce(
                self.reward_logits(z, action[:, t], params), reward[:, t],
                c['vmin'], c['vmax'], c['num_bins']))
            rng, dkey = jax.random.split(rng)
            q_logits = self.q_logits(z, action[:, t], params, train=True, rng=dkey)
            value_loss = value_loss + rho * jnp.mean(w * soft_ce(
                q_logits, td_targets[:, t], c['vmin'], c['vmax'], c['num_bins']))
            preds = self.next_all(z, action[:, t], params)
            if t == 0 and preds.shape[0] > 1:
                dis0 = jax.lax.stop_gradient(preds).var(0)   # (B, latent)
            consistency_loss = consistency_loss + rho * jnp.mean(
                w * jnp.mean((preds - next_z_real[:, t]) ** 2, -1, keepdims=True))
            z = preds.mean(0)
            zs.append(z)
            rho = rho * c['rho']
        consistency_loss = consistency_loss / horizon
        reward_loss = reward_loss / horizon
        value_loss = value_loss / horizon
        total = (c['consistency_coef'] * consistency_loss
                 + c['reward_coef'] * reward_loss
                 + c['value_coef'] * value_loss)
        return total, (consistency_loss, reward_loss, value_loss,
                       jnp.stack(zs, 1), dis0)

    def _pi_loss(self, pi_params, zs, valid_z, rng):
        """Max-entropy prior on detached latents; Q read through the STORED
        params (held constant), average of two random heads, divided by the
        running scale (tdmpc/agent.py _update_pi)."""
        c = self.cfg
        params = dict(self.params)
        params['pi'] = pi_params
        flat = lambda x: x.reshape(-1, x.shape[-1])
        rng, pi_key, q_key = jax.random.split(rng, 3)
        _, action, _, scaled_entropy = self.pi(flat(zs), pi_key, params)
        q = self.q_subset(flat(zs), action, q_key, reduce='avg')
        q = q.reshape(*zs.shape[:-1], 1)
        scaled_entropy = scaled_entropy.reshape(*zs.shape[:-1], 1)
        new_scale = self._new_scale(q[:, 0])
        q = q / new_scale
        rho = c['rho'] ** jnp.arange(zs.shape[1])
        per_step = -(c['entropy_coef'] * scaled_entropy + q) * valid_z
        pi_loss = jnp.mean(per_step.mean(axis=(0, 2)) * rho)
        return pi_loss, (new_scale, q)

    def _new_scale(self, q0):
        """RunningScale: EMA of the clamped 5th-95th percentile span."""
        x = jax.lax.stop_gradient(q0).flatten()
        lo, hi = jnp.quantile(x, jnp.array([0.05, 0.95]))
        span = jnp.clip(hi - lo, 1.0, None)
        new = (1 - self.cfg['tau']) * self.scale + self.cfg['tau'] * span
        return jnp.where(jnp.isfinite(span), new, self.scale)

    def _data_ref(self, zs, batch, dis0, rng):
        """data_disagreement EMA (tdmpc/agent.py update_novelty_reference).
        'rollout': path measure over the first chunk_len real actions of
        windows that stay in one episode (unchanged when none does);
        'step': step-0 variance reduced."""
        c = self.cfg
        if c['ref_mode'] == 'rollout':
            T = min(self.chunk_len, batch['action'].shape[1])
            keep = (batch['valid'][:, :T, 0].min(axis=1) > 0.5)
            d_all = self.path_disagreement(
                jax.lax.stop_gradient(zs[:, 0]), batch['action'][:, :T], rng)
            n = keep.sum()
            d = jnp.where(n > 0, (d_all * keep).sum() / jnp.maximum(n, 1), 0.0)
            has = n > 0
        else:
            if dis0 is None:      # single dynamics head: no reference to track
                return self.data_disagreement
            d = dis0.mean()
            has = jnp.array(True)
        ema = jnp.where(self.data_disagreement > 1e-8,
                        0.99 * self.data_disagreement + 0.01 * d, d)
        return jnp.where(has, ema, self.data_disagreement)

    @jax.jit
    def update(self, batch):
        """One TD-MPC2 joint update (tdmpc/agent.py update, step q_mode):
        model losses -> clip -> Adam (enc at lr * enc_lr_scale), then the
        prior's update, the target-Q soft update and the novelty-reference
        EMA. batch: obs/next_obs (B,H,obs), action (B,H,A),
        reward/mask/valid (B,H,1)."""
        c = self.cfg
        rng, td_key, loss_key, pi_key, ref_key = jax.random.split(self.rng, 5)
        td_targets, next_z_real = self._td_target(batch, td_key)

        model_params = {k: self.params[k] for k in
                        ('encoder', 'dynamics', 'reward', 'Qs')}

        def loss_fn(mp):
            p = dict(mp)
            p['pi'] = self.params['pi']
            return self._losses(p, batch, next_z_real, td_targets, loss_key)

        (total, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(model_params)
        consistency_loss, reward_loss, value_loss, zs, dis0 = aux
        updates, new_model_opt = self.model_tx().update(
            grads, self.model_opt_state, model_params)
        model_params = optax.apply_updates(model_params, updates)
        new_params = dict(model_params)
        new_params['pi'] = self.params['pi']
        after_model = self.replace(params=new_params)

        # policy prior on the detached rollout latents; latent t+1 is valid
        # when step t was inside the episode, latent 0 always is
        valid_z = jnp.concatenate(
            [jnp.ones_like(batch['valid'][:, :1]), batch['valid']], axis=1)
        zs = jax.lax.stop_gradient(zs)

        def pi_loss_fn(pp):
            return after_model._pi_loss(pp, zs, valid_z, pi_key)

        (pi_loss, (new_scale, q_scaled)), pi_grads = jax.value_and_grad(
            pi_loss_fn, has_aux=True)(new_params['pi'])
        pi_updates, new_pi_opt = self.pi_tx().update(
            pi_grads, self.pi_opt_state, new_params['pi'])
        new_params = dict(new_params)
        new_params['pi'] = optax.apply_updates(new_params['pi'], pi_updates)

        new_target = optax.incremental_update(
            new_params['Qs'], self.target_q_params, c['tau'])
        # measured with the PRE-update parameters, as the reference does
        # (update_novelty_reference runs before opt.step there)
        new_ref = self._data_ref(zs, batch, dis0, ref_key)

        info = {
            'loss_total': total,
            'loss_consistency': consistency_loss,
            'loss_reward': reward_loss,
            'loss_value': value_loss,
            'loss_pi': pi_loss,
            'td_target_mean': td_targets.mean(),
            'q_scale': new_scale,
            'data_disagreement': new_ref,
        }
        return self.replace(
            rng=rng, params=new_params, target_q_params=new_target,
            model_opt_state=new_model_opt, pi_opt_state=new_pi_opt,
            scale=new_scale, data_disagreement=new_ref), info

    # -------------------------------------------------------- optimizers

    def model_tx(self):
        c = self.cfg
        labels = lambda params: {
            k: jax.tree.map(lambda _: 'enc' if k == 'encoder' else 'rest', v)
            for k, v in params.items()}
        return optax.chain(
            optax.clip_by_global_norm(c['grad_clip_norm']),
            optax.multi_transform(
                {'enc': optax.adam(c['lr'] * c['enc_lr_scale']),
                 'rest': optax.adam(c['lr'])},
                labels))

    def pi_tx(self):
        c = self.cfg
        return optax.chain(
            optax.clip_by_global_norm(c['grad_clip_norm']),
            optax.adam(c['lr'], eps=1e-5))

    # ------------------------------------------------------------- create

    @classmethod
    def create(cls, seed, obs_dim, action_dim, chunk_len, gamma, config):
        cfg = dict(config)
        rng = jax.random.PRNGKey(seed)
        rng, e_key, d_key, r_key, p_key, q_key = jax.random.split(rng, 6)

        latent, mlp_dim = cfg['latent_dim'], cfg['mlp_dim']
        simnorm = SimNorm(cfg['simnorm_dim'])
        defs = {
            # layers.enc: max(enc_layers - 1, 1) hidden layers of enc_dim,
            # then a NormedLinear to the latent with SimNorm
            'encoder': MLP(cfg['enc_dim'], latent,
                           max(cfg['enc_layers'] - 1, 1), out_act=simnorm),
            'dynamics': ensemble(MLP, cfg['num_dyn'], hidden_dim=mlp_dim,
                                 out_dim=latent, num_layers=2, out_act=simnorm),
            'reward': MLP(mlp_dim, cfg['num_bins'], 2, zero_out=True),
            'pi': MLP(mlp_dim, 2 * action_dim, 2),
            'Qs': ensemble(MLP, cfg['num_q'], hidden_dim=mlp_dim,
                           out_dim=cfg['num_bins'], num_layers=2,
                           dropout=cfg['dropout'], zero_out=True),
        }
        ex_z = jnp.zeros((1, latent))
        ex_za = jnp.zeros((1, latent + action_dim))
        params = {
            'encoder': defs['encoder'].init(e_key, jnp.zeros((1, obs_dim)))['params'],
            'dynamics': defs['dynamics'].init(d_key, ex_za)['params'],
            'reward': defs['reward'].init(r_key, ex_za)['params'],
            'pi': defs['pi'].init(p_key, ex_z)['params'],
            'Qs': defs['Qs'].init(q_key, ex_za)['params'],
        }
        model_params = {k: params[k] for k in ('encoder', 'dynamics', 'reward', 'Qs')}
        self0 = cls(
            rng=rng, params=params, target_q_params=params['Qs'],
            model_opt_state=None, pi_opt_state=None,
            scale=jnp.ones(()), data_disagreement=jnp.asarray(1e-8),
            cfg=flax.core.FrozenDict(cfg), defs=flax.core.FrozenDict(defs),
            action_dim=action_dim, chunk_len=chunk_len, gamma=gamma)
        return self0.replace(
            model_opt_state=self0.model_tx().init(model_params),
            pi_opt_state=self0.pi_tx().init(params['pi']))
