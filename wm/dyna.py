import numpy as np
import torch

from sac_chunked.chunk_utils import pool_chunk
from helpers.interop import jax_to_torch

def decode_obs(bridge, carry, obs_key, device):
    """ Latent -> predicted observation, as a torch tensor.

        The actor and critic live in observation space, so every point where
        an imagined latent has to talk to them goes through here. The result
        is a PREDICTION, not a real state: it carries decoder error on top of
        whatever dynamics error accumulated to reach this latent. wm_report
        measures both against real replay states. """
    decoded = bridge.decode_state(carry)[obs_key]
    return torch.as_tensor(np.asarray(decoded, dtype=np.float32), device=device)

def success_seed_indices(batch_np, n, chunk_len, rng, reward_thresh,
                         obs_key='state'):
    """ Flat indices into a Dreamer batch's (B*T) axis that sit 1..chunk_len
        steps BEFORE an above-threshold reward, so an imagined chunk launched
        there overlaps the reward moment. This is the whole point of Dyna on a
        sparse task: real signal moments are the scarcest resource, and
        seeding imagination right before them multiplies each one into many
        counterfactual training chunks.

        Index 0 of each sequence is skipped as a hit: when a window starts at
        a true episode start it carries the fabricated reward=0.0 that
        ogbench_to_dreamer_episode prepends, which clears any threshold and
        would seed on resets instead of signal.

        Returns an int64 array of up to n indices -- empty when no sequence in
        the batch contains a hit (caller falls back to uniform seeds). """
    rew = np.asarray(batch_np['reward'], dtype=np.float32)
    n_seq, t_len = rew.shape
    hits = [np.flatnonzero(rew[b, 1:] > reward_thresh) + 1 for b in range(n_seq)]
    seqs = [b for b in range(n_seq) if len(hits[b])]
    if not seqs or n <= 0:
        return np.zeros((0,), dtype=np.int64)
    out = np.empty((n,), dtype=np.int64)
    for i in range(n):
        b = seqs[rng.integers(0, len(seqs))]
        r = hits[b][rng.integers(0, len(hits[b]))]
        offset = rng.integers(1, chunk_len + 1)
        out[i] = b * t_len + max(int(r) - int(offset), 0)
    return out

def imagine_transitions(bridge, policy, pool, seed_idx, obs_flat, chunk_len,
                        gamma, device, obs_key='state', reward_shift=0.0):
    """ Synthetic chunk transitions for the critic, one imagined chunk per
        seed.

        For each seed index: start from the REAL observation (critic input
        stays in-distribution), sample a chunk from the CURRENT policy at that
        observation, roll it through the model from the posterior latent, and
        read off the pooled reward, the survival product, and the DECODED end
        observation. The caller builds the TD target from these; nothing here
        touches the actor.

        pool:     flattened posterior latents from bridge.seed_pool.
        seed_idx: (n,) int64 flat indices into pool / obs_flat.
        obs_flat: (B*T, obs_dim) real observations, same flat layout as pool.

        Returns a dict of torch tensors:
          obs      (n, obs_dim)   REAL seed observation
          chunk    (n, chunk_len * action_dim)  policy chunk (imagined action)
          reward   (n, 1)         pooled model reward for the chunk
          cont     (n, 1)         product of the chunk_len cont probabilities
          next_obs (n, obs_dim)   DECODED observation after the chunk """
    obs_t = torch.as_tensor(obs_flat[seed_idx], device=device).float()
    with torch.no_grad():
        chunk_t = policy.sample_chunk(obs_t)

    seed_carry = bridge.place_seed({k: v[seed_idx] for k, v in pool.items()})
    carry, _, reward_j, cont_j = bridge.img_chunk(
        seed_carry, chunk_t.detach().cpu().numpy(), chunk_len)

    # (n, chunk_len) -> (chunk_len, n, 1), step-major, matching pool_chunk.
    r = jax_to_torch(reward_j, device).transpose(0, 1).unsqueeze(-1) + reward_shift
    c = jax_to_torch(cont_j, device).transpose(0, 1).unsqueeze(-1)
    pooled_r, pooled_c = pool_chunk(r, c, gamma)
    next_obs = decode_obs(bridge, carry, obs_key, device)

    return {'obs': obs_t, 'chunk': chunk_t, 'reward': pooled_r,
            'cont': pooled_c, 'next_obs': next_obs}