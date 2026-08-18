import numpy as np
import torch

def pool_chunk(rewards, conts, gamma):
    """ Folds chunk_len single-step (reward, cont) pairs into one chunk-level
        pair.

        rewards, conts: (chunk_len, batch, 1), step-major.
        Returns chunk_reward, chunk_cont: both (batch, 1).

        Each reward is multiplied by the running product of the cont
        probabilities BEFORE it, so a success at step 2 of 5 does not get
        credited with the three steps that would not have happened. This is
        the whole reason the RSSM is still stepped one step at a time -- a
        model that jumps chunk_len steps cannot express partial termination. """
    horizon = rewards.shape[0]
    chunk_reward = torch.zeros_like(rewards[0])
    alive = torch.ones_like(conts[0])
    discount = 1.0
    for k in range(horizon):
        chunk_reward = chunk_reward + discount * alive * rewards[k]
        alive = alive * conts[k]
        discount = discount * gamma
    return chunk_reward, alive

def pool_chunk_np(rewards, conts, gamma):
    """ numpy twin of pool_chunk for the no-world-model arm, which pools real
        replay rewards instead of imagined ones.
        rewards, conts: (batch, chunk_len). Returns (batch, 1) each. """
    horizon = rewards.shape[1]
    chunk_reward = np.zeros((rewards.shape[0], 1), dtype=np.float32)
    alive = np.ones((rewards.shape[0], 1), dtype=np.float32)
    discount = 1.0
    for k in range(horizon):
        chunk_reward += discount * alive * rewards[:, k:k + 1]
        alive = alive * conts[:, k:k + 1]
        discount *= gamma
    return chunk_reward, alive

def chunk_lambda_targets(chunk_rewards, chunk_conts, next_values, gamma_h, lam, num_chunks):
    """ Identical recursion to the single-step lambda_targets in
        sac_wm_agent.py, run over chunks instead of steps. The only change is
        that gamma has become gamma ** chunk_len, because chunk_len real
        environment steps elapse between consecutive entries.

        All inputs are the flattened (num_chunks * batch, 1) tensors that
        imagine_chunk_rollout returns, chunk-major. """
    batch = chunk_rewards.shape[0] // num_chunks
    r = chunk_rewards.reshape(num_chunks, batch, 1)
    c = chunk_conts.reshape(num_chunks, batch, 1)
    v = next_values.reshape(num_chunks, batch, 1)

    ret = v[-1]
    outs = []
    for t in reversed(range(num_chunks)):
        ret = r[t] + gamma_h * c[t] * ((1 - lam) * v[t] + lam * ret)
        outs.append(ret)
    return torch.stack(outs[::-1], dim=0).reshape(num_chunks * batch, 1)

def chunk_pair_indices(batch_np, chunk_len, action_key='action'):
    """ Builds aligned (flat latent index, action chunk) pairs from a Dreamer
        batch, for training the flow behavior model.

        seed_pool flattens the (batch, seq_len) leading dims, so flat index i
        corresponds to sequence i // seq_len at timestep i % seq_len.

        Dreamer's convention (see OGBenchMethods.ogbench_to_dreamer_episode)
        is that action[t] is the action taken FROM state[t], so the chunk
        launched at latent t is action[t : t + chunk_len]. Windows that run
        past the end of the sampled sequence, or that cross an episode
        boundary, are dropped -- the last action of an episode is zero
        padding, not a real action. """
    actions = np.asarray(batch_np[action_key])
    is_last = np.asarray(batch_np['is_last']).astype(bool)
    n_seq, seq_len, action_dim = actions.shape

    if seq_len < chunk_len:
        empty_idx = np.zeros((0,), dtype=np.int64)
        empty_chunks = np.zeros((0, chunk_len * action_dim), dtype=np.float32)
        return empty_idx, empty_chunks

    starts = np.arange(seq_len - chunk_len + 1)
    window = starts[:, None] + np.arange(chunk_len)[None, :]
    ok = ~is_last[:, window].any(-1)

    seq_ok, start_ok = np.nonzero(ok)
    flat_idx = seq_ok * seq_len + starts[start_ok]
    chunks = actions[seq_ok[:, None], window[start_ok]]
    return flat_idx.astype(np.int64), chunks.reshape(len(seq_ok), chunk_len * action_dim).astype(np.float32)

def temporal_coherence(positions, stride=5):
    """ QC's action-coherency proxy (their Figure 4, right): mean L2 distance
        between end-effector positions `stride` steps apart. Jitter and pauses
        drive it down, committed motion drives it up. Higher is better, and it
        is the single number that says whether the chunk distribution is
        actually coherent or just open-loop noise.

        positions: (T, 3) array of end-effector xyz over one episode. """
    pos = np.asarray(positions, dtype=np.float32)
    if len(pos) <= stride:
        return 0.0
    deltas = pos[stride:] - pos[:-stride]
    return float(np.linalg.norm(deltas, axis=-1).mean())
