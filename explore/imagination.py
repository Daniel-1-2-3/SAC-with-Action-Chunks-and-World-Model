""" Plan2Explore's core loop, adapted: the explorer learns to seek expected
    future novelty from trajectories imagined under the world model, labeled
    with ensemble disagreement -- zero environment steps spent.

    Adaptation vs. the paper (documented design decision): P2E backprops
    pathwise through the Dreamer graph into the explorer; here the explorer
    is a SEAR learner, so imagination instead MANUFACTURES OFF-POLICY DATA:
    imagined episodes in the same dreamer format as real replay, consumed by
    the explorer's ordinary multi-horizon TD update. Same objective
    (maximize expected future disagreement, computed entirely in
    imagination), different optimizer route.

    Rollout: start states are real observations sampled from recent replay,
    posterior-encoded; the explorer proposes chunks against DECODED
    observations; the prior rolls each step.

    Performance: the step loop only advances the recurrence and decodes at
    CHUNK BOUNDARIES (the explorer needs an observation once per chunk).
    Per-step observations for episode storage and all disagreement rewards
    are computed in two batched passes over the stacked features afterward.
"""
import numpy as np
import torch


@torch.no_grad()
def imagine_episodes(wm, explorer, ensemble, start_obs, horizon_chunks,
                     obs_key='state', action_key='action'):
    """ start_obs: (B, obs_dim) numpy of real states. Returns a list of B
        dreamer-format episodes of length 1 + horizon_chunks * chunk_len
        (index 0 is the fabricated pre-step, matching real replay). """
    device = wm.device
    B = len(start_obs)
    N, A = explorer.chunk_len, explorer.act_dim
    carry = wm.init(B)
    carry, feat = wm.encode_step(
        carry, start_obs, np.zeros((B, A), np.float32), np.ones(B, bool))

    feat_seq, act_seq = [feat], []
    boundary_obs = wm.decode(feat)
    for _ in range(horizon_chunks):
        chunks, _ = explorer.policy.sample(boundary_obs)     # (B, N, A)
        for i in range(N):                # recurrence: sequential by nature
            carry = wm.img_step(carry, chunks[:, i])
            feat_seq.append(wm.feat(carry))
            act_seq.append(chunks[:, i])
        boundary_obs = wm.decode(feat_seq[-1])

    T = horizon_chunks * N
    feats = torch.stack(feat_seq, 1)                          # (B, T+1, F)
    acts = torch.stack(act_seq, 1)                            # (B, T, A)
    # batched: decode every step's observation in one pass
    obs_all = wm.decode(feats.reshape(B * (T + 1), -1)
                        ).reshape(B, T + 1, -1)
    # batched: disagreement reward for every (obs_t, a_t) in one pass
    rew_all = ensemble.disagreement(
        obs_all[:, :-1].reshape(B * T, -1),
        acts.reshape(B * T, -1)).reshape(B, T)

    obs_np = obs_all.cpu().numpy()
    act_np = acts.cpu().numpy()
    rew_np = rew_all.cpu().numpy()
    episodes = []
    for b in range(B):                    # cheap python: list packaging only
        ep = {
            obs_key: obs_np[b].astype(np.float32),
            action_key: np.concatenate(
                [np.zeros((1, A), np.float32), act_np[b]], 0),
            'reward': np.concatenate(
                [np.zeros(1, np.float32), rew_np[b]], 0),
            'is_first': np.zeros(T + 1, bool),
            'is_last': np.zeros(T + 1, bool),
            'is_terminal': np.zeros(T + 1, bool),
            'cont': np.ones(T + 1, np.float32),
        }
        ep['is_first'][0] = True
        ep['is_last'][-1] = True
        episodes.append(ep)
    return episodes