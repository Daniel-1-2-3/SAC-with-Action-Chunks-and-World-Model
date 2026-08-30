""" Builds SEAR's multi-horizon training windows from dreamer-format
    episodes (the replay layout: index 0 fabricated, action[t] leads INTO
    obs[t], reward[t] is received AT obs[t]).

    For a window starting at t, the agent needs:
      obs      = obs[t]
      actions  = action[t+1 .. t+N]      (the N actions taken from obs[t])
      rewards  = reward[t+1 .. t+N]      (their step rewards)
      next_obs[k-1] = obs[t+k]           (state after the k-step prefix)
      boot[k-1]     = cont[t+k]          (0 cuts the bootstrap at terminals)
      valid[k-1]    = 1                  (full windows only; per-prefix
                                          validity is inherent: every prefix
                                          of a full window is in-episode)

    Fully vectorized: windows for all (b, t) are materialized with numpy
    sliding views and boolean-compressed in one shot -- no Python loops.
"""
import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view as swv


def build_windows(seqs, chunk_len, obs_key='state', action_key='action',
                  take=None, rng=None, device='cpu'):
    """ seqs: dict of (B, T, ...) numpy from helpers.common.sample_sequences.
        Returns dict of torch tensors, M = number of windows kept. """
    if rng is None:
        rng = np.random.default_rng()
    obs = np.ascontiguousarray(seqs[obs_key])
    act = np.ascontiguousarray(seqs[action_key])
    rew = np.ascontiguousarray(seqs['reward'])
    cont = np.ascontiguousarray(seqs['cont'])
    is_first = np.ascontiguousarray(seqs['is_first'])
    B, T = rew.shape
    N = chunk_len
    W = T - N                                  # window starts t = 0..W-1
    if W <= 0:
        return None
    # exclude windows crossing an episode boundary: any is_first inside
    # (t, t+N]. sliding view over is_first gives (B, W, N+1) covering
    # indices t..t+N; positions 1: are the exclusion zone.
    bad = swv(is_first, N + 1, axis=1)[:, :W, 1:].any(-1)     # (B, W)
    keep = ~bad
    o = obs[:, :W][keep]                                       # (M, D)
    # windows over the shifted-by-one tail arrays: index t+1..t+N
    a_w = swv(act[:, 1:], N, axis=1)                           # (B, W', N?, A)
    a_w = np.moveaxis(a_w, -1, 2)[:, :W][keep]                 # (M, N, A)
    r_w = swv(rew[:, 1:], N, axis=1)[:, :W][keep]              # (M, N)
    no_w = swv(obs[:, 1:], N, axis=1)                          # (B, W', D?, N)
    no_w = np.moveaxis(no_w, -1, 2)[:, :W][keep]               # (M, N, D)
    b_w = swv(cont[:, 1:], N, axis=1)[:, :W][keep]             # (M, N)
    M = len(o)
    if M == 0:
        return None
    if take is not None and M > take:
        idx = rng.choice(M, size=take, replace=False)
        o, a_w, r_w, no_w, b_w = (x[idx] for x in (o, a_w, r_w, no_w, b_w))
        M = take
    to = lambda x: torch.as_tensor(np.ascontiguousarray(x, dtype=np.float32),
                                   device=device)
    return {'obs': to(o), 'actions': to(a_w), 'rewards': to(r_w),
            'next_obs': to(no_w), 'boot': to(b_w),
            'valid': torch.ones(M, N, device=device)}
