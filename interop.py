import jax
import jax.numpy as jnp
import numpy as np
import torch

def jax_to_torch(x, device):
    """ Copies a JAX array out of device memory before wrapping it as a
        torch tensor. Always cast to float32 first """
    x = jnp.asarray(x).astype(jnp.float32)
    return torch.as_tensor(jax.device_get(x).copy(), device=device).float()

def unwrap(v):
    if isinstance(v, np.ndarray) and v.dtype == object and v.shape == ():
        return v.item()
    return v

def numeric_metrics(metrics, prefix=''):
    out = {}
    for k, v in metrics.items():
        try:
            out[f'{prefix}{k}'] = float(v)
        except (TypeError, ValueError):
            continue
    return out

def flatten_leading_two_dims_np(tree):
    return {k: jax.device_get(v).reshape((-1,) + v.shape[2:]) for k, v in tree.items()}

def subsample_tree_np(tree, n, rng):
    total = next(iter(tree.values())).shape[0]
    idx = rng.choice(total, size=min(n, total), replace=False)
    return {k: v[idx] for k, v in tree.items()}

def extract_state(obs, obs_key):
    if isinstance(obs, dict):
        return np.asarray(obs[obs_key], dtype=np.float32).reshape(1, -1)
    return np.asarray(obs, dtype=np.float32).reshape(1, -1)
