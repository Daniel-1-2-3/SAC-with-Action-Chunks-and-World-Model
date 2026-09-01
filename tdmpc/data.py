""" Short real windows for the latent model, cut out of the same replay
    batches QC-FQL trains on. Both consumers see identical data; only what
    they do with it differs. """

import numpy as np

from sac_chunked.chunk_utils import step_valid_np


def latent_rollout_windows(batch_np, horizon, obs_key='state',
                           action_key='action'):
    """ Every length-`horizon` window in a Dreamer-format sequence batch.

        The reward indexing is the same trap as in real_chunk_transitions:
        ogbench_to_dreamer_episode stores reward[t] as the reward for ARRIVING
        at state[t], so the reward earned by action[t] sits at reward[t+1].
        Discount and is_last follow the same convention. Getting this wrong
        trains the reward head one step out of phase, which is invisible in
        the loss and fatal to the score.

        Windows that run past the end of an episode are kept and masked
        (`valid`), matching how QC handles the same situation -- rejecting
        them would bias sampling away from episode ends, which on a cube task
        is exactly where the rewards are.

        Returns a dict of numpy arrays:
          obs     (N, horizon+1, obs_dim)
          action  (N, horizon, action_dim)
          reward  (N, horizon, 1)
          mask    (N, horizon, 1)  0 where the real transition terminated
          valid   (N, horizon, 1)  0 once the window ran past episode end """
    obs = np.asarray(batch_np[obs_key], dtype=np.float32)
    actions = np.asarray(batch_np[action_key], dtype=np.float32)
    reward = np.asarray(batch_np['reward'], dtype=np.float32)
    discount = np.asarray(batch_np['discount'], dtype=np.float32)
    is_last = np.asarray(batch_np['is_last']).astype(np.float32)
    n_seq, seq_len, action_dim = actions.shape

    if seq_len < horizon + 1:
        return None

    starts = np.arange(seq_len - horizon)
    act_win = starts[:, None] + np.arange(horizon)[None, :]
    obs_win = starts[:, None] + np.arange(horizon + 1)[None, :]
    rew_win = act_win + 1  # reward/discount/is_last lag the action by one

    seq_idx = np.repeat(np.arange(n_seq), len(starts))
    win_idx = np.tile(np.arange(len(starts)), n_seq)
    rows = seq_idx[:, None]

    t = is_last[rows, rew_win[win_idx]]
    return {
        'obs': obs[rows, obs_win[win_idx]],
        'action': actions[rows, act_win[win_idx]],
        'reward': reward[rows, rew_win[win_idx]][..., None],
        'mask': discount[rows, rew_win[win_idx]][..., None],
        'valid': step_valid_np(t)[..., None],
    }
