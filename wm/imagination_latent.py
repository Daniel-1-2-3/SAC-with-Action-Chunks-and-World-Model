import torch
from sac_chunked.chunk_utils import pool_chunk
from helpers.interop import jax_to_torch

def imagine_chunk_rollout_latent(bridge, agent, seed_carry, num_chunks,
                                 chunk_len, device, gamma, reward_shift=0.0):
    """ Imagines num_chunks consecutive chunks forward from seed_carry under
        the CURRENT actor, entirely in latent space -- no decoder anywhere.
        Used by the latent-MVE trainer only.

        This is the original codebase's imagination path, kept deliberately:
        the actor and critic in that trainer eat RSSM features
        (deter + flattened stoch via bridge.get_feat), so the imagined
        continuation can hand its endpoint latents straight to the target
        critic. The decoded-state variant of this rollout is what flatlined;
        the latent variant is what learned (-1700 by 1.25M in the original
        run).

        The seed is the latent at the state the real replayed chunk ended at.
        Each chunk is sampled fresh from the actor at the FEATURES of the
        latent the previous chunk reached, which keeps the continuation
        on-policy -- replay's own following actions came from an old policy
        and would estimate that policy's return instead.

        Returns, all chunk-major (num_chunks, batch, ...):
            chunk_rewards  discounted, termination-masked reward sum per chunk
            chunk_conts    product of the chunk_len cont probabilities
            next_feats     RSSM features AFTER each chunk (bootstrap points) """
    carry = seed_carry
    feat_t = jax_to_torch(bridge.get_feat(carry), device)

    chunk_rewards, chunk_conts, next_feats = [], [], []
    for _ in range(num_chunks):
        with torch.no_grad():
            chunk_t = agent.sample_chunk(feat_t)
        chunk_np = chunk_t.detach().cpu().numpy()

        carry, _, reward_j, cont_j = bridge.img_chunk(carry, chunk_np, chunk_len)

        # (B, chunk_len) -> (chunk_len, B, 1), step-major, matching pool_chunk.
        r = jax_to_torch(reward_j, device).transpose(0, 1).unsqueeze(-1) + reward_shift
        c = jax_to_torch(cont_j, device).transpose(0, 1).unsqueeze(-1)
        chunk_r, chunk_c = pool_chunk(r, c, gamma)

        feat_t = jax_to_torch(bridge.get_feat(carry), device)

        chunk_rewards.append(chunk_r)
        chunk_conts.append(chunk_c)
        next_feats.append(feat_t)

    return (torch.stack(chunk_rewards), torch.stack(chunk_conts),
            torch.stack(next_feats))