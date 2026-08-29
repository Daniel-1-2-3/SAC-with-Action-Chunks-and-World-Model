""" Shared utilities for the v8 (SEAR + Plan2Explore) stack.

    Jax-free on purpose: the legacy QC baseline keeps its own helpers
    (trainer_common / ogbench_methods / interop, which import jax); nothing
    in the v8 stack imports those. The functions here are ports of the
    pieces v8 needs, taken from the legacy helpers with jax removed.
"""
import os
import random

import elements
import numpy as np
import ruamel.yaml as yaml
import torch


def load_config(name, path='configs.yaml', argv=None):
    configs = yaml.YAML(typ='safe').load(open(path))
    merged = {**configs['defaults'], **configs.get(name, {})}
    config = elements.Config(merged)
    flags = elements.Flags(config)
    return flags.parse(argv)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def prefixed(metrics, prefix):
    return {f'{prefix}/{k}': v for k, v in metrics.items()}


def symlog_np(x):
    return np.sign(x) * np.log1p(np.abs(x))


def ogbench_to_dreamer_episode(obs_list, act_list, rew_list, term_list,
                               obs_key='state', action_key='action'):
    """ Same layout the legacy pipeline used: index 0 is a fabricated
        pre-step (zero action, zero reward, is_first) so that action[t]
        is the action LEADING INTO obs[t]. Downstream consumers must skip
        index 0's reward (it is not a real environment reward). """
    T = len(obs_list)
    ep = {
        obs_key: np.asarray(obs_list, dtype=np.float32),
        action_key: np.concatenate(
            [np.zeros((1, len(act_list[0])), np.float32),
             np.asarray(act_list[:-1] if len(act_list) == T else act_list,
                        dtype=np.float32)], axis=0)[:T],
        'reward': np.concatenate(
            [np.zeros(1, np.float32),
             np.asarray(rew_list[:-1] if len(rew_list) == T else rew_list,
                        dtype=np.float32)], axis=0)[:T],
        'is_first': np.zeros(T, bool),
        'is_last': np.zeros(T, bool),
        'is_terminal': np.zeros(T, bool),
    }
    ep['is_first'][0] = True
    ep['is_last'][-1] = True
    ep['is_terminal'][-1] = bool(term_list[-1]) if len(term_list) else False
    ep['cont'] = (~ep['is_terminal']).astype(np.float32)
    return ep


def sample_sequences(episodes, batch_size, seq_len, obs_key='state',
                     action_key='action', rng=None):
    """ Uniform sequence sampling from a list of dreamer-format episodes.
        Returns dict of (B, T, ...) arrays. Episodes shorter than seq_len
        are skipped by the caller (filter before calling). """
    if rng is None:
        rng = np.random.default_rng()
    keys = [obs_key, action_key, 'reward', 'is_first', 'is_terminal', 'cont']
    batch = {k: [] for k in keys}
    for _ in range(batch_size):
        ep = episodes[rng.integers(0, len(episodes))]
        start = rng.integers(0, len(ep[obs_key]) - seq_len + 1)
        for k in keys:
            batch[k].append(ep[k][start:start + seq_len])
    return {k: np.stack(v) for k, v in batch.items()}


def temporal_coherence(positions, stride=5):
    """ QC's action-coherency proxy (their Figure 4, right): mean L2 distance
        between end-effector positions `stride` steps apart. Jitter and pauses
        drive it down, committed motion drives it up. Higher is better. """
    pos = np.asarray(positions, dtype=np.float32)
    if len(pos) <= stride:
        return 0.0
    deltas = pos[stride:] - pos[:-stride]
    return float(np.linalg.norm(deltas, axis=-1).mean())
