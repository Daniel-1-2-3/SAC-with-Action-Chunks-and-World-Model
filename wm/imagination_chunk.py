import torch
from sac_chunked.chunk_utils import pool_chunk
from helpers.interop import jax_to_torch

def imagine_chunk_rollout(bridge, agent, seed_carry, num_chunks, chunk_len,
                          device, gamma, reward_shift=0.0):
    """ Imagines num_chunks * chunk_len environment steps, but exposes them to
        the critic as num_chunks chunk-level transitions.

        The RSSM is NOT asked to jump. It is stepped one step at a time, fed
        the chunk's actions in order, exactly as imagination.imagine_rollout
        already does -- the only change is where the actions come from. The
        actor is queried once per chunk instead of once per step, and the
        chunk_len actions are held in a buffer.

        This preserves causality end to end: every predicted latent is
        computed from the realized previous latent, never from the chunk start
        in parallel. It also keeps per-step reward resolution, which a jumpy
        model would destroy on a task whose reward is a single transition.

        Returns, all flattened to (num_chunks * batch, ...), chunk-major:
            feats         latent at each chunk start
            chunks        the chunk_len * action_dim vector executed
            chunk_rewards discounted, termination-masked reward sum
            chunk_conts   product of the chunk_len cont probabilities
            next_feats    latent after the whole chunk
            weights       survival weight, decayed by gamma**chunk_len * cont
            step_rewards  (chunk_len, num_chunks * batch, 1), kept unpooled so
                          the caller can check that no reward leaks past a
                          termination inside a chunk """
    carry = seed_carry
    feat_t = jax_to_torch(bridge.get_feat(carry), device)
    action_dim = agent.action_dim
    gamma_h = gamma ** chunk_len

    feats, chunks, chunk_rewards, chunk_conts, next_feats, weights, step_rewards = \
        [], [], [], [], [], [], []
    weight = torch.ones(feat_t.shape[0], 1, device=device)

    for _ in range(num_chunks):
        with torch.no_grad():
            chunk_t = agent.sample_chunk(feat_t)
        chunk_np = chunk_t.detach().cpu().numpy()

        rewards_k, conts_k = [], []
        next_feat_flat = None
        for k in range(chunk_len):
            action_np = chunk_np[:, k * action_dim:(k + 1) * action_dim]
            carry, next_feat_flat, reward_j, cont_j = bridge.img_step(carry, action_np)
            rewards_k.append(jax_to_torch(reward_j, device).reshape(-1, 1) + reward_shift)
            conts_k.append(jax_to_torch(cont_j, device).reshape(-1, 1))

        r = torch.stack(rewards_k, dim=0)
        c = torch.stack(conts_k, dim=0)
        chunk_r, chunk_c = pool_chunk(r, c, gamma)
        next_feat_t = jax_to_torch(next_feat_flat, device)

        feats.append(feat_t)
        chunks.append(chunk_t)
        chunk_rewards.append(chunk_r)
        chunk_conts.append(chunk_c)
        next_feats.append(next_feat_t)
        weights.append(weight)
        step_rewards.append(r)

        weight = weight * (gamma_h * chunk_c)
        feat_t = next_feat_t

    return (
        torch.cat(feats), torch.cat(chunks), torch.cat(chunk_rewards),
        torch.cat(chunk_conts), torch.cat(next_feats), torch.cat(weights),
        torch.cat(step_rewards, dim=1),
    )