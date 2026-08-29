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
    observations (so it acts in the same space it collects in); the prior
    rolls each step; each imagined (obs, action) is labeled with ensemble
    disagreement as its reward.
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
    zero_act = torch.zeros(B, A, device=device)
    carry, feat = wm.encode_step(
        carry, start_obs, zero_act.cpu().numpy(), np.ones(B, bool))
    obs_t = wm.decode(feat)                              # (B, obs_dim)

    obs_seq, act_seq, rew_seq = [obs_t], [], []
    for _ in range(horizon_chunks):
        chunks, _ = explorer.policy.sample(obs_t)        # (B, N, A)
        for i in range(N):
            a = chunks[:, i]
            rew_seq.append(ensemble.disagreement(obs_t, a))
            act_seq.append(a)
            carry = wm.img_step(carry, a)
            obs_t = wm.decode(wm.feat(carry))
            obs_seq.append(obs_t)

    obs_np = torch.stack(obs_seq, 1).cpu().numpy()       # (B, T+1... , D)
    act_np = torch.stack(act_seq, 1).cpu().numpy()       # (B, T, A)
    rew_np = torch.stack(rew_seq, 1).cpu().numpy()       # (B, T)
    episodes = []
    T = obs_np.shape[1]
    for b in range(B):
        ep = {
            obs_key: obs_np[b].astype(np.float32),
            # shift: action[t] leads INTO obs[t]; index 0 fabricated
            action_key: np.concatenate(
                [np.zeros((1, A), np.float32), act_np[b]], 0),
            'reward': np.concatenate(
                [np.zeros(1, np.float32), rew_np[b]], 0),
            'is_first': np.zeros(T, bool),
            'is_last': np.zeros(T, bool),
            'is_terminal': np.zeros(T, bool),
            'cont': np.ones(T, np.float32),
        }
        ep['is_first'][0] = True
        ep['is_last'][-1] = True
        episodes.append(ep)
    return episodes
