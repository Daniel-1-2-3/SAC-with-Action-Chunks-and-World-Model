import re
import elements
import embodied.embodied.jax
import embodied.embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import dreamer.rssm as rssm

f32 = jnp.float32
sg = lambda xs, skip=False: xs if skip else jax.lax.stop_gradient(xs)
isimage = lambda s: s.dtype == np.uint8 and len(s.shape) == 3

class WorldModelAgent(embodied.embodied.jax.Agent): # From Dreamer, policy code excluded

    def __init__(self, obs_space, act_space, config):
        self.obs_space = obs_space
        self.act_space = act_space
        self.config = config

        exclude = ('is_first', 'is_last', 'is_terminal', 'reward')
        enc_space = {k: v for k, v in obs_space.items() if k not in exclude}
        self.dec_space = {k: v for k, v in obs_space.items() if k not in exclude}

        self.enc = {'simple': rssm.Encoder}[config.enc.typ](enc_space, **config.enc[config.enc.typ], name='enc')
        self.dyn = {'rssm': rssm.RSSM}[config.dyn.typ](act_space, **config.dyn[config.dyn.typ], name='dyn')
        self.dec = {'simple': rssm.Decoder}[config.dec.typ](self.dec_space, **config.dec[config.dec.typ], name='dec')

        self.feat2tensor = lambda x: jnp.concatenate([
            nn.cast(x['deter']),
            nn.cast(x['stoch'].reshape((*x['stoch'].shape[:-2], -1)))], -1)

        scalar = elements.Space(np.float32, ())
        binary = elements.Space(bool, (), 0, 2)
        self.rew = embodied.embodied.jax.MLPHead(scalar, **config.rewhead, name='rew')
        self.con = embodied.embodied.jax.MLPHead(binary, **config.conhead, name='con')

        self.modules = [self.dyn, self.enc, self.dec, self.rew, self.con]
        self.opt = embodied.embodied.jax.Optimizer(
            self.modules, 
            self._make_opt(**config.opt), 
            summary_depth=1,
            name='opt'
        )

        scales = self.config.loss_scales.copy()
        rec = scales.pop('rec')
        self.scales = {k: scales[k] for k in ('dyn', 'rep', 'rew', 'con')}
        self.scales.update({k: rec for k in self.dec_space})

    @property
    def ext_space(self):
        return {
            'consec': elements.Space(np.int32),
            'stepid': elements.Space(np.uint8, 20),
        }

    @property
    def policy_keys(self):
        return '^(enc|dyn|dec|rew|con)/'

    def init_train(self, batch_size):
        zeros = lambda x: jnp.zeros((batch_size, *x.shape), x.dtype)
        return (
            self.enc.initial(batch_size),
            self.dyn.initial(batch_size),
            self.dec.initial(batch_size),
            jax.tree.map(zeros, self.act_space))

    def init_report(self, batch_size):
        return self.init_train(batch_size)

    def _shift_actions(self, prevact_carry, data): # Format transitions timing for Dreamer
        prepend = lambda x, y: jnp.concatenate([x[:, None], y[:, :-1]], 1)
        return {k: prepend(prevact_carry[k], data[k]) for k in self.act_space}

    def train(self, carry, data):
        enc_carry, dyn_carry, dec_carry, prevact_carry = carry
        obs = {k: data[k] for k in self.obs_space}
        prevact = self._shift_actions(prevact_carry, data)

        metrics, (new_carry, outs, mets) = self.opt(
            self.loss, (enc_carry, dyn_carry, dec_carry), obs, prevact,
            training=True, has_aux=True)
        metrics.update(mets)

        carry = (*new_carry, {k: data[k][:, -1] for k in self.act_space})
        return carry, outs, metrics

    def report(self, carry, data):
        enc_carry, dyn_carry, dec_carry, prevact_carry = carry
        obs = {k: data[k] for k in self.obs_space}
        prevact = self._shift_actions(prevact_carry, data)

        loss, (new_carry, outs, metrics) = self.loss(
            (enc_carry, dyn_carry, dec_carry), obs, prevact, training=False)
        metrics['loss/total'] = loss

        carry = (*new_carry, {k: data[k][:, -1] for k in self.act_space})
        return carry, metrics
    
    def init_imagine(self, batch_size):
        return self.dyn.initial(batch_size)

    def encode_posterior(self, batch_size, data):
        enc_carry, dyn_carry, dec_carry, prevact_carry = self.init_train(batch_size)
        obs = {k: data[k] for k in self.obs_space}
        prevact = self._shift_actions(prevact_carry, data)
        reset = obs['is_first']

        enc_carry, enc_entries, tokens = self.enc(enc_carry, obs, reset, training=False)
        dyn_carry, dyn_entries, feat = self.dyn.observe(
            dyn_carry, tokens, prevact, reset, training=False)

        # Public contract: always float32 out, regardless of internal compute dtype.
        dyn_entries = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), dyn_entries)
        return dyn_entries

    def imagine_step(self, dyn_carry, action):
        dyn_carry, action = nn.cast((dyn_carry, action))
        next_carry, (feat, _) = self.dyn.imagine(
            dyn_carry, action, 1, training=False, single=True)
        inp = self.feat2tensor(feat)
        reward = self.rew(inp, 1).pred()
        cont = self.con(inp, 1).pred()

        next_carry = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), next_carry)
        inp = inp.astype(jnp.float32)
        reward = reward.astype(jnp.float32)
        cont = cont.astype(jnp.float32)
        return next_carry, inp, reward, cont
    
    def init_encode(self, batch_size):
        return self.enc.initial(batch_size), self.dyn.initial(batch_size)

    def encode_step(self, enc_carry, dyn_carry, obs, prevact, is_first):
        enc_carry, enc_entries, tokens = self.enc(enc_carry, obs, is_first, training=False, single=True)
        dyn_carry, entry, feat = self.dyn.observe(dyn_carry, tokens, prevact, is_first, training=False, single=True)
        inp = self.feat2tensor(dyn_carry)
        enc_carry = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), enc_carry)
        dyn_carry = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), dyn_carry)
        inp = inp.astype(jnp.float32)
        return enc_carry, dyn_carry, inp

    def loss(self, carry, obs, prevact, training):
        enc_carry, dyn_carry, dec_carry = carry
        reset = obs['is_first']
        losses, metrics = {}, {}

        enc_carry, enc_entries, tokens = self.enc(enc_carry, obs, reset, training)
        dyn_carry, dyn_entries, los, repfeat, mets = self.dyn.loss(
            dyn_carry, tokens, prevact, reset, training)
        losses.update(los)
        metrics.update(mets)

        dec_carry, dec_entries, recons = self.dec(
            dec_carry, repfeat, reset, training)

        inp = self.feat2tensor(repfeat)
        rew_dist = self.rew(inp, 2)
        losses['rew'] = rew_dist.loss(obs['reward'])
        metrics['pred/rew'] = rew_dist.pred()

        con = f32(~obs['is_terminal'])
        if self.config.contdisc:
            con *= 1 - 1 / self.config.horizon
        con_dist = self.con(inp, 2)
        losses['con'] = con_dist.loss(con)
        metrics['pred/con'] = con_dist.pred()

        for key, recon in recons.items():
            space, value = self.obs_space[key], obs[key]
            target = f32(value) / 255 if isimage(space) else value
            losses[key] = recon.loss(sg(target))

        B, T = reset.shape
        shapes = {k: v.shape for k, v in losses.items()}
        assert all(x == (B, T) for x in shapes.values()), ((B, T), shapes)
        assert set(losses.keys()) == set(self.scales.keys()), (
            sorted(losses.keys()), sorted(self.scales.keys()))

        metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})
        loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])

        carry = (enc_carry, dyn_carry, dec_carry)
        outs = {'tokens': tokens, 'repfeat': repfeat, 'losses': losses}
        return loss, (carry, outs, metrics)

    def _make_opt(
        self,
        lr: float = 4e-5,
        agc: float = 0.3,
        eps: float = 1e-20,
        beta1: float = 0.9,
        beta2: float = 0.999,
        momentum: bool = True,
        nesterov: bool = False,
        wd: float = 0.0,
        wdregex: str = r'/kernel$',
        schedule: str = 'const',
        warmup: int = 1000,
        anneal: int = 0,
    ):
        chain = []
        chain.append(embodied.embodied.jax.opt.clip_by_agc(agc))
        chain.append(embodied.embodied.jax.opt.scale_by_rms(beta2, eps))
        chain.append(embodied.embodied.jax.opt.scale_by_momentum(beta1, nesterov))
        if wd:
            assert not wdregex[0].isnumeric(), wdregex
            pattern = re.compile(wdregex)
            wdmask = lambda params: {k: bool(pattern.search(k)) for k in params}
            chain.append(optax.add_decayed_weights(wd, wdmask))
        assert anneal > 0 or schedule == 'const'
        if schedule == 'const':
            sched = optax.constant_schedule(lr)
        elif schedule == 'linear':
            sched = optax.linear_schedule(lr, 0.1 * lr, anneal - warmup)
        elif schedule == 'cosine':
            sched = optax.cosine_decay_schedule(lr, anneal - warmup, 0.1 * lr)
        else:
            raise NotImplementedError(schedule)
        if warmup:
            ramp = optax.linear_schedule(0.0, lr, warmup)
            sched = optax.join_schedules([ramp, sched], [warmup])
        chain.append(optax.scale_by_learning_rate(sched))
        return optax.chain(*chain)