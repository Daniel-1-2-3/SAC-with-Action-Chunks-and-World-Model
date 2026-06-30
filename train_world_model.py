"""

ASSUMPTIONS YOU SHOULD VERIFY AGAINST YOUR ENVIRONMENT:
  - embodied.jax.Agent provides the same nj.pure/jit wrapping around methods
    named train/report/init_* that the real Agent subclass relies on. Since
    WorldModelAgent subclasses embodied.jax.Agent the same way, this should
    carry over, but if your installed `embodied` differs, the wrapping
    mechanism may differ too.
  - agent.save() / agent.load(...) exist on the base class for checkpointing
    (standard pattern in this codebase family). If not, swap the two
    TODO-marked lines for whatever your `embodied`/`elements` version uses.
"""

import pathlib
import re

import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import ruamel.yaml as yaml

from rssm import Encoder, Decoder, RSSM
from ogbench_dataset_methods import DatasetMethods

f32 = jnp.float32
sg = lambda xs, skip=False: xs if skip else jax.lax.stop_gradient(xs)
isimage = lambda s: s.dtype == np.uint8 and len(s.shape) == 3


# ---------------------------------------------------------------------------
# WorldModelAgent: world-model-only slice of Agent (agent.py)
# ---------------------------------------------------------------------------

class WorldModelAgent(embodied.jax.Agent):

  def __init__(self, obs_space, act_space, config):
    self.obs_space = obs_space
    self.act_space = act_space
    self.config = config

    exclude = ('is_first', 'is_last', 'is_terminal', 'reward')
    enc_space = {k: v for k, v in obs_space.items() if k not in exclude}
    dec_space = {k: v for k, v in obs_space.items() if k not in exclude}

    self.enc = {
        'simple': Encoder,
    }[config.enc.typ](enc_space, **config.enc[config.enc.typ], name='enc')
    self.dyn = {
        'rssm': RSSM,
    }[config.dyn.typ](act_space, **config.dyn[config.dyn.typ], name='dyn')
    self.dec = {
        'simple': Decoder,
    }[config.dec.typ](dec_space, **config.dec[config.dec.typ], name='dec')

    self.feat2tensor = lambda x: jnp.concatenate([
        nn.cast(x['deter']),
        nn.cast(x['stoch'].reshape((*x['stoch'].shape[:-2], -1)))], -1)

    scalar = elements.Space(np.float32, ())
    binary = elements.Space(bool, (), 0, 2)
    self.rew = embodied.jax.MLPHead(scalar, **config.rewhead, name='rew')
    self.con = embodied.jax.MLPHead(binary, **config.conhead, name='con')

    self.modules = [self.dyn, self.enc, self.dec, self.rew, self.con]
    self.opt = embodied.jax.Optimizer(
        self.modules, self._make_opt(**config.opt), summary_depth=1,
        name='opt')

    # Only keep scales for losses this trimmed agent actually computes:
    # dyn, rep, rew, con, and one reconstruction term per decoded obs key.
    # The full config.loss_scales also has policy/value/repval entries that
    # don't apply here.
    all_scales = dict(self.config.loss_scales)
    rec = all_scales.pop('rec')
    scales = {k: all_scales[k] for k in ('dyn', 'rep', 'rew', 'con')}
    scales.update({k: rec for k in dec_space})
    self.scales = scales

  def init_train(self, batch_size):
    zeros = lambda x: jnp.zeros((batch_size, *x.shape), x.dtype)
    return (
        self.enc.initial(batch_size),
        self.dyn.initial(batch_size),
        self.dec.initial(batch_size),
        jax.tree.map(zeros, self.act_space))

  def init_report(self, batch_size):
    return self.init_train(batch_size)

  def _shift_actions(self, prevact_carry, data):
    """Mirror Agent._apply_replay_context's no-replay-context branch."""
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
    losses['rew'] = self.rew(inp, 2).loss(obs['reward'])

    con = f32(~obs['is_terminal'])
    if self.config.contdisc:
      con *= 1 - 1 / self.config.horizon
    losses['con'] = self.con(inp, 2).loss(con)

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
    chain.append(embodied.jax.opt.clip_by_agc(agc))
    chain.append(embodied.jax.opt.scale_by_rms(beta2, eps))
    chain.append(embodied.jax.opt.scale_by_momentum(beta1, nesterov))
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


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def load_config(folder, preset=None):
  configs_txt = elements.Path(folder / 'configs.yaml').read()
  configs = yaml.YAML(typ='safe').load(configs_txt)
  config = elements.Config(configs['defaults'])
  if preset:
    config = config.update(configs[preset])
  return config


def build_agent_config(config, batch_size, seq_len, logdir):
  return elements.Config(
      **config.agent,
      logdir=str(logdir),
      seed=config.seed,
      jax=config.jax,
      batch_size=batch_size,
      batch_length=seq_len,
      replay_context=0,
      report_length=seq_len,
      replica=0,
      replicas=1,
  )


def main(
    env_name: str,
    obs_key: str = 'state',
    action_key: str = 'action',
    seq_len: int = 64,
    batch_size: int = 16,
    train_steps: int = 50_000,
    log_every: int = 100,
    eval_every: int = 1_000,
    ckpt_every: int = 5_000,
    preset: str = None,
    seed: int = 0,
):
  folder = pathlib.Path(__file__).parent
  logdir = folder / 'logs_world_model'
  logdir.mkdir(parents=True, exist_ok=True)

  config = load_config(folder, preset)

  print(f'Loading OGBench dataset: {env_name}')
  env, train_dataset, val_dataset = DatasetMethods.load_ogbench(env_name)
  obs_dim = train_dataset['observations'].shape[-1]
  action_dim = train_dataset['actions'].shape[-1]
  env.close()

  train_episodes = DatasetMethods.make_dreamer_episodes(
      train_dataset, min_length=seq_len, obs_key=obs_key, action_key=action_key)
  val_episodes = DatasetMethods.make_dreamer_episodes(
      val_dataset, min_length=seq_len, obs_key=obs_key, action_key=action_key)
  DatasetMethods.print_summary(train_dataset, train_episodes, obs_key=obs_key)

  obs_space, act_space = DatasetMethods.make_spaces(
      obs_dim, action_dim, obs_key, action_key)

  agent_config = build_agent_config(config, batch_size, seq_len, logdir)
  agent = WorldModelAgent(obs_space, act_space, agent_config)

  train_carry = agent.init_train(batch_size)
  eval_carry = agent.init_report(batch_size)
  rng = np.random.default_rng(seed)
  eval_rng = np.random.default_rng(seed + 1)

  def fmt(metrics, prefix='loss/'):
    items = {k[len(prefix):]: float(v) for k, v in metrics.items()
              if k.startswith(prefix)}
    return '  '.join(f'{k}={v:.4f}' for k, v in sorted(items.items()))

  print(f'Starting world model training for {train_steps} steps.')
  for step in range(1, train_steps + 1):
    batch = DatasetMethods.sample_jax_dreamer_batch(
        train_episodes, batch_size, seq_len, obs_key, action_key, rng=rng)
    train_carry, outs, metrics = agent.train(train_carry, batch)

    if step % log_every == 0:
      print(f'step {step:>7}  {fmt(metrics)}')

    if step % eval_every == 0:
      val_batch = DatasetMethods.sample_jax_dreamer_batch(
          val_episodes, batch_size, seq_len, obs_key, action_key, rng=eval_rng)
      eval_carry, eval_metrics = agent.report(eval_carry, val_batch)
      print(f'  [eval] step {step:>7}  {fmt(eval_metrics)}')

    if step % ckpt_every == 0:
      ckpt_path = logdir / f'checkpoint_{step}.npz'
      # TODO: verify against your embodied/elements version. agent.save()
      # is the standard pattern in this codebase family for extracting the
      # ninjax parameter pytree; swap this if your version differs.
      state = agent.save()
      np.savez(ckpt_path, **{k: np.asarray(v) for k, v in state.items()})
      print(f'  saved checkpoint: {ckpt_path}')

  print('done')


if __name__ == '__main__':
  import sys
  # Minimal arg handling -- replace with elements.Flags if you want this to
  # accept the same --flag style as main.py.
  env_name = sys.argv[1] if len(sys.argv) > 1 else 'pointmaze-medium-navigate-singletask-v0'
  main(env_name)