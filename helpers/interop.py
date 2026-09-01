import numpy as np

def numeric_metrics(metrics, prefix=''):
    out = {}
    for k, v in metrics.items():
        try:
            out[f'{prefix}{k}'] = float(v)
        except (TypeError, ValueError):
            continue
    return out

def extract_state(obs, obs_key):
    if isinstance(obs, dict):
        return np.asarray(obs[obs_key], dtype=np.float32).reshape(1, -1)
    return np.asarray(obs, dtype=np.float32).reshape(1, -1)
