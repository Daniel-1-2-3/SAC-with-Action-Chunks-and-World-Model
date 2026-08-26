"""Scaffolding shared by the three method trainers (train_wm_mve.py,
train_wm_select.py, train_wm_dyna.py). Everything here is method-agnostic and
was byte-identical across the three standalone versions; the train loops and
agent updates stay in their own files on purpose -- those ARE the experiments.
"""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false') # Don't let JAX hog the GPU before torch inits
os.environ.setdefault('MUJOCO_GL', 'egl') # Headless for video rendering

import elements
import jax
import numpy as np
import ruamel.yaml as yaml
import ogbench

from helpers.ogbench_methods import OGBenchMethods

OBS_KEY = 'state'
ACTION_KEY = 'action'
ENV_ACTION_LOW = -1.0
ENV_ACTION_HIGH = 1.0

def load_config(folder, argv=None):
    configs_txt = elements.Path(folder / 'configs.yaml').read()
    configs = yaml.YAML(typ='safe').load(configs_txt)
    parsed, other = elements.Flags(configs=['defaults']).parse_known(argv)
    config = elements.Config(configs['defaults'])
    for name in parsed.configs:
        config = config.update(configs[name])
    config = elements.Flags(config).parse(other)
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

def build_real_env(env_name, load_offline_dataset):
    if load_offline_dataset:
        return OGBenchMethods.load_ogbench(env_name)
    env = ogbench.make_env_and_datasets(env_name, env_only=True)
    return env, None, None

def param_norm(params):
    """ Norm over FLOATING-POINT leaves only. tree_leaves(params) also returns
        integer counters, and squaring those made the reported norm move for
        reasons unrelated to the weights. """
    leaves = [x for x in jax.tree_util.tree_leaves(params)
              if hasattr(x, 'dtype')
              and jax.numpy.issubdtype(x.dtype, jax.numpy.floating)]
    if not leaves:
        return float('nan')
    squares = [jax.numpy.sum(jax.numpy.square(x)) for x in leaves]
    total = jax.numpy.sum(jax.numpy.stack(squares))
    return float(jax.device_get(total)) ** 0.5

def prefixed(d, default_prefix):
    return {k if '/' in k else f'{default_prefix}/{k}': v for k, v in d.items()}

def wm_update(wm_agent, replay, batch_size, seq_len, rng, global_step):
    """ DreamerV3 world-model update on real replay sequences. Each trainer's
        header comment names the ONLY consumer of this model in that build;
        the critic/actor training path never touches it in any of them
        (MVE consumes it inside target computation, which is torch.no_grad).
        Co-trained so the model keeps tracking the state distribution the
        policy is currently visiting. """
    batch_np = replay.sample_batch(batch_size, seq_len, rng=rng)
    batch = OGBenchMethods.to_jax(batch_np)
    batch.pop('discount', None)
    batch['seed'] = wm_agent._seeds(global_step, wm_agent.train_mirrored)
    wm_carry = wm_agent.init_train(batch_size)
    wm_carry, outs, wm_mets = wm_agent.train(wm_carry, batch)
    return wm_mets