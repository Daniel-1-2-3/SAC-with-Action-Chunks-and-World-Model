import numpy as np
import torch

def pool_chunk(rewards, conts, gamma):
    """ Folds chunk_len single-step (reward, cont) pairs into one chunk-level
        pair, for IMAGINED rollouts.

        rewards, conts: (chunk_len, batch, 1), step-major.
        Returns chunk_reward, chunk_cont: both (batch, 1).

        Each reward is multiplied by the running product of the cont
        probabilities BEFORE it, so a success at step 2 of 5 does not get
        credited with the three steps that would not have happened. This is
        the whole reason the RSSM is still stepped one step at a time -- a
        model that jumps chunk_len steps cannot express partial termination.

        QC has no analogue: it works on real data, where termination is a hard
        flag rather than a probability, so it uses a plain discounted sum plus
        a running-min mask (see pool_chunk_np). The soft version here is the
        world-model arm's own, not a deviation from the reference. """
    horizon = rewards.shape[0]
    chunk_reward = torch.zeros_like(rewards[0])
    alive = torch.ones_like(conts[0])
    discount = 1.0
    for k in range(horizon):
        chunk_reward = chunk_reward + discount * alive * rewards[k]
        alive = alive * conts[k]
        discount = discount * gamma
    return chunk_reward, alive

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
    """ Builds aligned (flat latent index, action chunk, per-step valid) triples
        from a Dreamer batch, for the BC flow term in the world-model arm.

        seed_pool flattens the (batch, seq_len) leading dims, so flat index i
        corresponds to sequence i // seq_len at timestep i % seq_len.

        Dreamer's convention (see OGBenchMethods.ogbench_to_dreamer_episode)
        is that action[t] is the action taken FROM state[t], so the chunk
        launched at latent t is action[t : t + chunk_len].

        Windows are NOT rejected for crossing an episode boundary, matching
        QC's sample_sequence -- they are kept and masked per position, so the
        seed distribution stays uniform. Only windows running past the end of
        the sampled sequence are dropped, since those actions do not exist. """
    actions = np.asarray(batch_np[action_key])
    is_last = np.asarray(batch_np['is_last']).astype(np.float32)
    n_seq, seq_len, action_dim = actions.shape

    if seq_len < chunk_len:
        return (np.zeros((0,), dtype=np.int64),
                np.zeros((0, chunk_len * action_dim), dtype=np.float32),
                np.zeros((0, chunk_len), dtype=np.float32))

    starts = np.arange(seq_len - chunk_len + 1)
    window = starts[:, None] + np.arange(chunk_len)[None, :]

    seq_ok, start_ok = np.nonzero(np.ones((n_seq, len(starts)), dtype=bool))
    flat_idx = seq_ok * seq_len + starts[start_ok]
    chunks = actions[seq_ok[:, None], window[start_ok]]
    terms = is_last[seq_ok[:, None], window[start_ok]]
    valid = step_valid_np(terms)
    return (flat_idx.astype(np.int64),
            chunks.reshape(len(seq_ok), chunk_len * action_dim).astype(np.float32),
            valid)

def real_chunk_transitions(batch_np, chunk_len, gamma, action_key='action'):
    """ Real (not imagined) chunk transitions from a Dreamer batch, for
        MIXING into the critic/actor update alongside imagined ones.

        Same window convention as chunk_pair_indices: action[t] is taken FROM
        state[t], so the chunk launched at latent index t is
        action[t : t+chunk_len]. Reward and discount do NOT share that
        indexing -- OGBenchMethods.ogbench_to_dreamer_episode assigns
        Dreamer reward[t] to ARRIVING at state[t] ("reward[t+1] = OGBench
        reward[t]"), and discount/is_terminal are built the same way. So the
        reward and discount attributable to the chunk launched at t are at
        [t+1 : t+chunk_len+1], one index ahead of the actions. Getting this
        wrong silently misattributes every reward by one step.

        Also requires room for a NEXT latent (index t+chunk_len) inside the
        same sampled sequence, since the critic target needs
        Q(next_feat, ...) -- chunk_pair_indices doesn't need this because BC
        pairs have no bootstrap target. This makes the valid start range one
        shorter than chunk_pair_indices': starts <= seq_len - chunk_len - 1.

        `valid` marks windows that had not already ended before their FIRST
        real reward position (t+1), matching pool_chunk_np's semantics.

        Returns (flat_idx, next_flat_idx, chunks, chunk_reward, chunk_mask,
        valid), each length-matched: flat_idx/next_flat_idx are int64 pool
        indices, chunks is (N, chunk_len*action_dim), the rest are (N, 1). """
    actions = np.asarray(batch_np[action_key])
    reward = np.asarray(batch_np['reward'], dtype=np.float32)
    discount = np.asarray(batch_np['discount'], dtype=np.float32)
    is_last = np.asarray(batch_np['is_last']).astype(np.float32)
    n_seq, seq_len, action_dim = actions.shape

    if seq_len < chunk_len + 1:
        z = lambda n: np.zeros((0, n), dtype=np.float32)
        return (np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64),
                np.zeros((0, chunk_len * action_dim), dtype=np.float32),
                z(1), z(1), z(1))

    starts = np.arange(seq_len - chunk_len) # one shorter than chunk_pair_indices
    action_window = starts[:, None] + np.arange(chunk_len)[None, :]
    reward_window = action_window + 1 # shifted: reward/discount lag actions by 1

    seq_ok, start_ok = np.nonzero(np.ones((n_seq, len(starts)), dtype=bool))
    s = starts[start_ok]
    flat_idx = (seq_ok * seq_len + s).astype(np.int64)
    next_flat_idx = (seq_ok * seq_len + s + chunk_len).astype(np.int64)
    chunks = actions[seq_ok[:, None], action_window[start_ok]]
    chunks = chunks.reshape(len(seq_ok), chunk_len * action_dim).astype(np.float32)

    r = reward[seq_ok[:, None], reward_window[start_ok]]
    m = discount[seq_ok[:, None], reward_window[start_ok]]
    t = is_last[seq_ok[:, None], reward_window[start_ok]]
    chunk_reward, chunk_mask, valid = pool_chunk_np(r, m, t, gamma)
    return flat_idx, next_flat_idx, chunks, chunk_reward, chunk_mask, valid

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