import torch
from sac_chunked.chunk_utils import pool_chunk
from helpers.interop import jax_to_torch
from wm.model_ops import decode_obs

def imagine_chunk_rollout(bridge, agent, seed_carry, num_chunks, chunk_len,
                          device, gamma, obs_key='state', reward_shift=0.0):
    """ Imagines num_chunks consecutive chunks forward from seed_carry under
        the CURRENT actor, and returns them as chunk-level quantities. Used by
        the MVE trainer only.

        The seed is the latent at the state the real replayed chunk ended at,
        so chunk i here is the (i+1)-th chunk after the real one. Each chunk is
        sampled fresh from the actor at the decoded state the previous chunk
        reached, which is what keeps the continuation on-policy -- the entire
        reason these can extend the target where replay's own following
        actions cannot.

        The chunk_len steps WITHIN a chunk are rolled in one dispatch via
        bridge.img_chunk (RSSM.imagine scans internally). The outer loop stays
        in Python because the actor lives in PyTorch and has to be re-queried
        between chunks.

        Returns, all chunk-major (num_chunks, batch, ...):
            chunk_rewards  discounted, termination-masked reward sum per chunk
            chunk_conts    product of the chunk_len cont probabilities
            next_obs       decoded observation AFTER each chunk (bootstrap pts) """
    carry = seed_carry
    obs_t = decode_obs(bridge, carry, obs_key, device)

    chunk_rewards, chunk_conts, next_obs = [], [], []
    for _ in range(num_chunks):
        with torch.no_grad():
            chunk_t = agent.sample_chunk(obs_t)
        chunk_np = chunk_t.detach().cpu().numpy()

        carry, _, reward_j, cont_j = bridge.img_chunk(carry, chunk_np, chunk_len)

        # (B, chunk_len) -> (chunk_len, B, 1), step-major, matching pool_chunk.
        r = jax_to_torch(reward_j, device).transpose(0, 1).unsqueeze(-1) + reward_shift
        c = jax_to_torch(cont_j, device).transpose(0, 1).unsqueeze(-1)
        chunk_r, chunk_c = pool_chunk(r, c, gamma)

        obs_t = decode_obs(bridge, carry, obs_key, device)

        chunk_rewards.append(chunk_r)
        chunk_conts.append(chunk_c)
        next_obs.append(obs_t)

    return (torch.stack(chunk_rewards), torch.stack(chunk_conts),
            torch.stack(next_obs))