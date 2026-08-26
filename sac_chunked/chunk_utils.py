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

def mve_continuation(img_rewards, img_conts, final_value, gamma_h, lam=1.0,
                     inter_values=None):
    """ The imagined continuation that replaces QC-FQL's Q(s_next) inside the
        target. The caller multiplies this by gamma^h * mask and adds the real
        chunk reward, so this function returns ONLY the bracketed term:

            [ R_1 + g*c_1 * [ R_2 + g*c_2 * [ ... + g*c_N * Q(s_N) ] ] ]

        with g = gamma^chunk_len. lam=1.0 is exactly that nesting: the only
        bootstrap is Q at the deepest imagined state, and inter_values is not
        needed (the caller should not spend the forward passes computing it).
        lam<1 blends the shallower imagined bootstraps Q(s_1..s_N) in, which
        caps how far a model that degrades with depth can drag the target; it
        requires inter_values.

        img_rewards, img_conts: (num_chunks, batch, 1), chunk-major.
        final_value:  (batch, 1), target-critic value at the deepest state.
        inter_values: (num_chunks, batch, 1) or None; inter_values[-1] must
                      equal final_value when supplied.

        Returns (batch, 1). """
    num_chunks = img_rewards.shape[0]
    if lam < 1.0 and inter_values is None:
        raise ValueError('mve_continuation needs inter_values when lam < 1.0')
    ret = final_value
    for t in reversed(range(num_chunks)):
        blended = ret if lam >= 1.0 else (1.0 - lam) * inter_values[t] + lam * ret
        ret = img_rewards[t] + gamma_h * img_conts[t] * blended
    return ret

def real_chunk_transitions(batch_np, chunk_len, gamma, obs_key='state',
                           action_key='action'):
    """ Real chunk transitions from a Dreamer batch: everything the critic and
        actor need, in the same form the no-world-model arm builds them.

        action[t] is taken FROM state[t], so the chunk launched at index t is
        action[t : t+chunk_len]. Reward and discount do NOT share that
        indexing -- OGBenchMethods.ogbench_to_dreamer_episode assigns Dreamer
        reward[t] to ARRIVING at state[t] ("reward[t+1] = OGBench reward[t]"),
        and discount/is_terminal are built the same way. So the reward and
        discount attributable to the chunk launched at t are at
        [t+1 : t+chunk_len+1], one index ahead of the actions. Getting this
        wrong silently misattributes every reward by one step.

        Requires room for the NEXT state (index t+chunk_len) inside the same
        sampled sequence, since the target needs Q(s_{t+h}, .).

        Windows are NOT rejected for crossing an episode boundary, matching
        QC's sample_sequence -- they are kept and masked, so the start-state
        distribution stays uniform.

        Returns a dict of numpy arrays, all length-matched on the first axis:
          idx, next_idx   int64 flat indices into the (batch*seq_len) axis
          obs, next_obs   (N, obs_dim)
          chunk           (N, chunk_len * action_dim)
          reward, mask    (N, 1) pooled over the chunk
          valid           (N, 1) window had not already ended
          step_valid      (N, chunk_len) per-position, for the BC flow loss """
    obs = np.asarray(batch_np[obs_key], dtype=np.float32)
    actions = np.asarray(batch_np[action_key], dtype=np.float32)
    reward = np.asarray(batch_np['reward'], dtype=np.float32)
    discount = np.asarray(batch_np['discount'], dtype=np.float32)
    is_last = np.asarray(batch_np['is_last']).astype(np.float32)
    n_seq, seq_len, action_dim = actions.shape
    obs_dim = obs.shape[-1]

    if seq_len < chunk_len + 1:
        return None

    obs_flat = obs.reshape(-1, obs_dim)
    starts = np.arange(seq_len - chunk_len)
    action_window = starts[:, None] + np.arange(chunk_len)[None, :]
    reward_window = action_window + 1 # shifted: reward/discount lag actions by 1

    seq_ok, start_ok = np.nonzero(np.ones((n_seq, len(starts)), dtype=bool))
    s = starts[start_ok]
    idx = (seq_ok * seq_len + s).astype(np.int64)
    next_idx = (seq_ok * seq_len + s + chunk_len).astype(np.int64)

    chunk = actions[seq_ok[:, None], action_window[start_ok]]
    chunk = chunk.reshape(len(seq_ok), chunk_len * action_dim).astype(np.float32)

    r = reward[seq_ok[:, None], reward_window[start_ok]]
    m = discount[seq_ok[:, None], reward_window[start_ok]]
    t = is_last[seq_ok[:, None], reward_window[start_ok]]
    chunk_reward, chunk_mask, valid = pool_chunk_np(r, m, t, gamma)

    return {
        'idx': idx,
        'next_idx': next_idx,
        'obs': obs_flat[idx],
        'next_obs': obs_flat[next_idx],
        'chunk': chunk,
        'reward': chunk_reward,
        'mask': chunk_mask,
        'valid': valid,
        'step_valid': step_valid_np(t),
    }

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