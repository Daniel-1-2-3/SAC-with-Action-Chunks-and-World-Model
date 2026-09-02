import numpy as np

def pool_chunk_np(rewards, masks, terminals, gamma):
    """ utils/datasets.py sample_sequence, vectorized over a batch of windows.

        rewards, masks, terminals: (batch, chunk_len).
        Returns (chunk_reward, chunk_mask, valid), each (batch, 1), taken at
        the last position of the window.

          reward[i] = sum_{k<=i} gamma^k * r[k]      plain discounted sum
          mask[i]   = min(mask[i-1], mask[i])        running min
          valid[i]  = 1 - terminals[i-1]             1 if the window had not
                                                     already ended before i

        `valid` is why QC does not need to reject windows that cross an
        episode boundary: it keeps them and zeroes their contribution to the
        loss instead, which preserves the uniform sampling distribution over
        start states. Rejection biases sampling away from episode ends, which
        on a cube task is exactly where the completions are. """
    horizon = rewards.shape[1]
    chunk_reward = rewards[:, 0].astype(np.float32).copy()
    chunk_mask = masks[:, 0].astype(np.float32).copy()
    run_term = terminals[:, 0].astype(np.float32).copy()
    valid = np.ones_like(chunk_reward)
    discount = 1.0
    for k in range(1, horizon):
        discount *= gamma
        chunk_reward = chunk_reward + discount * rewards[:, k]
        chunk_mask = np.minimum(chunk_mask, masks[:, k])
        valid = 1.0 - run_term
        run_term = np.maximum(run_term, terminals[:, k])
    return (chunk_reward[:, None], chunk_mask[:, None], valid[:, None])

def step_valid_np(terminals):
    """ Per-position validity inside a window, (batch, chunk_len). Position 0
        is always valid; position k is valid if the window had not terminated
        by k-1. Used to mask the BC flow loss per action within the chunk. """
    batch, horizon = terminals.shape
    valid = np.ones((batch, horizon), dtype=np.float32)
    run_term = terminals[:, 0].astype(np.float32).copy()
    for k in range(1, horizon):
        valid[:, k] = 1.0 - run_term
        run_term = np.maximum(run_term, terminals[:, k])
    return valid

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