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
"""
import numpy as np
import torch


def build_windows(seqs, chunk_len, obs_key='state', action_key='action',
                  take=None, rng=None, device='cpu'):
    """ seqs: dict of (B, T, ...) numpy from helpers.common.sample_sequences.
        Returns dict of torch tensors, M = number of windows kept. """
    if rng is None:
        rng = np.random.default_rng()
    obs, act = seqs[obs_key], seqs[action_key]
    rew, cont = seqs['reward'], seqs['cont']
    is_first = seqs['is_first']
    B, T = rew.shape
    N = chunk_len
    starts = np.arange(0, T - N)                      # t, needs t+N <= T-1
    o_list, a_list, r_list, no_list, b_list = [], [], [], [], []
    for b in range(B):
        # exclude windows that cross an episode boundary (an is_first
        # inside (t, t+N]) -- boundaries only occur at sequence start in
        # this replay, but imagined episodes keep the same invariant.
        firsts = np.flatnonzero(is_first[b])
        for t in starts:
            if np.any((firsts > t) & (firsts <= t + N)):
                continue
            o_list.append(obs[b, t])
            a_list.append(act[b, t + 1: t + N + 1])
            r_list.append(rew[b, t + 1: t + N + 1])
            no_list.append(obs[b, t + 1: t + N + 1])
            b_list.append(cont[b, t + 1: t + N + 1])
    M = len(o_list)
    if M == 0:
        return None
    if take is not None and M > take:
        idx = rng.choice(M, size=take, replace=False)
    else:
        idx = np.arange(M)
    to = lambda x: torch.as_tensor(
        np.asarray(x, dtype=np.float32)[idx], device=device)
    return {'obs': to(o_list), 'actions': to(a_list), 'rewards': to(r_list),
            'next_obs': to(no_list), 'boot': to(b_list),
            'valid': torch.ones(len(idx), N, device=device)}
