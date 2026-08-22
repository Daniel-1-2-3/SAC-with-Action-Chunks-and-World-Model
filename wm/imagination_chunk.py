import torch
from sac_chunked.chunk_utils import pool_chunk
from helpers.interop import jax_to_torch

def imagine_chunk_rollout(bridge, agent, seed_carry, num_chunks, chunk_len,
                          device, gamma, reward_shift=0.0):
    """ Imagines num_chunks consecutive chunks forward from seed_carry, under
        the CURRENT actor, and returns them as chunk-level quantities.

        For MVE the seed is the latent at the END of a real replayed chunk, so
        chunk i here is the (i+1)-th chunk after the real one. Each chunk is
        sampled fresh from the actor at the latent the previous chunk reached,
        which is what keeps the whole continuation on-policy -- that is the
        entire reason these can extend the target where replay's own following
        actions cannot.

        The chunk_len steps WITHIN a chunk are rolled in one dispatch via
        bridge.img_chunk (RSSM.imagine scans internally). The outer loop over
        chunks stays in Python because the actor lives in PyTorch and has to
        be re-queried between chunks.

        Returns, all (num_chunks * batch, ...) flattened chunk-major:
            chunk_rewards  discounted, termination-masked reward sum per chunk
            chunk_conts    product of the chunk_len cont probabilities
            next_feats     latent AFTER each chunk (the bootstrap points)
            step_rewards   (chunk_len, num_chunks * batch, 1), unpooled, so the
                           caller can check no reward leaks past a termination
                           inside a chunk """
    carry = seed_carry
    feat_t = jax_to_torch(bridge.get_feat(carry), device)

    chunk_rewards, chunk_conts, next_feats, step_rewards = [], [], [], []

    for _ in range(num_chunks):
        with torch.no_grad():
            chunk_t = agent.sample_chunk(feat_t)
        chunk_np = chunk_t.detach().cpu().numpy()

        carry, feats_j, reward_j, cont_j = bridge.img_chunk(carry, chunk_np, chunk_len)

        # (B, chunk_len) -> (chunk_len, B, 1), step-major, matching pool_chunk.
        r = jax_to_torch(reward_j, device).transpose(0, 1).unsqueeze(-1) + reward_shift
        c = jax_to_torch(cont_j, device).transpose(0, 1).unsqueeze(-1)
        chunk_r, chunk_c = pool_chunk(r, c, gamma)

        # feats_j is (B, chunk_len, D); the latent after the chunk is the last.
        feats_all = jax_to_torch(feats_j, device)
        next_feat_t = feats_all[:, -1]

        chunk_rewards.append(chunk_r)
        chunk_conts.append(chunk_c)
        next_feats.append(next_feat_t)
        step_rewards.append(r)

        feat_t = next_feat_t

    return (torch.cat(chunk_rewards), torch.cat(chunk_conts),
            torch.cat(next_feats), torch.cat(step_rewards, dim=1))