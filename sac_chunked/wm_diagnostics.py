"""World-model accuracy checks for the two quantities MVE actually consumes.

The MVE target uses exactly two model outputs:

    reward   the pooled discounted reward of each imagined chunk, at depths
             1..N
    state    the decoded observation at the deepest imagined latent, which is
             what the observation-space critic bootstraps on

Everything here measures one of those against real replay data, so a bad
number points at a specific term in the target rather than at "the model".

Why this runs against REPLAY and not the live env:

  - The previous version drove the real environment to collect ground truth.
    Those env steps were not counted in the training budget and they perturbed
    the same env the collection loop uses.
  - It sampled states by running the current policy, so when the policy sits
    at the reward floor every ground-truth reward is identical, every
    correlation divides by ~0 and comes back NaN, and the reward MAE looks
    excellent for the trivial reason that predicting a constant is easy.
    Sampling with bias_start_to_reward=True guarantees the window contains an
    above-baseline reward, so the correlations are defined and the MAE is
    measured where it matters.

The tradeoff, stated plainly: rollouts here are teacher-forced with the
actions in replay, so this measures model accuracy on the data distribution,
not on the current policy's distribution. When the policy is far from the data
these numbers are optimistic. They are still the right gate, because a model
that cannot predict rewards on data it was trained on will not predict them
on-policy either.

Metrics are logged only if they carry information. Thresholds, standard errors
and sample counts are constants -- they are printed once, not logged as
timeseries.
"""
import numpy as np
import torch

from sac_chunked.chunk_utils import pool_chunk, pool_chunk_np
from helpers.interop import jax_to_torch
from helpers.ogbench_methods import OGBenchMethods
from wm.imagination_chunk import decode_obs

def _corr(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if len(a) < 6 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])

def wm_report(bridge, replay, chunk_config, chunk_len, depth, gamma, device,
              rng, batch_size, seq_len, obs_key='state', action_key='action',
              num_states=256, model_samples=4):
    span = chunk_len * depth
    if seq_len < span + 1 or not replay.ready(seq_len):
        return {}

    batch_np = replay.sample_batch(batch_size, seq_len, rng=rng,
                                   bias_start_to_reward=True)
    obs = np.asarray(batch_np[obs_key], dtype=np.float32)
    act = np.asarray(batch_np[action_key], dtype=np.float32)
    rew = np.asarray(batch_np['reward'], dtype=np.float32)
    disc = np.asarray(batch_np['discount'], dtype=np.float32)
    last = np.asarray(batch_np['is_last']).astype(np.float32)
    n_seq, t_len, obs_dim = obs.shape
    action_dim = act.shape[-1]

    starts = np.arange(t_len - span)
    seq_grid, start_grid = np.meshgrid(np.arange(n_seq), starts, indexing='ij')
    seq_i, start_i = seq_grid.ravel(), start_grid.ravel()
    if len(seq_i) < 6:
        return {}
    sel = rng.choice(len(seq_i), size=min(num_states, len(seq_i)), replace=False)
    seq_i, start_i = seq_i[sel], start_i[sel]
    n = len(seq_i)

    pool = bridge.seed_pool(OGBenchMethods.to_jax(batch_np), n_seq)
    seed_np = {k: v[(seq_i * t_len + start_i).astype(np.int64)] for k, v in pool.items()}

    # --- ground truth, per depth ------------------------------------------
    real_chunks, r_true, obs_true, cont_true = [], [], [], []
    for k in range(depth):
        a0 = start_i + k * chunk_len
        awin = a0[:, None] + np.arange(chunk_len)[None, :]
        rwin = awin + 1
        real_chunks.append(act[seq_i[:, None], awin].reshape(n, chunk_len * action_dim))
        pooled, _, _ = pool_chunk_np(rew[seq_i[:, None], rwin],
                                     disc[seq_i[:, None], rwin],
                                     last[seq_i[:, None], rwin], gamma)
        r_true.append(pooled[:, 0])
        obs_true.append(obs[seq_i, start_i + (k + 1) * chunk_len])
        cont_true.append(1.0 - last[seq_i[:, None], rwin].max(axis=1))

    # --- model rollout, averaged over prior draws -------------------------
    r_draws = [[] for _ in range(depth)]
    c_draws = [[] for _ in range(depth)]
    o_draws = [[] for _ in range(depth)]
    seed_decoded = None
    with torch.no_grad():
        for _ in range(max(1, model_samples)):
            carry = bridge.place_seed(seed_np)
            if seed_decoded is None:
                seed_decoded = decode_obs(bridge, carry, obs_key, device).cpu().numpy()
            for k in range(depth):
                carry, _, reward_j, cont_j = bridge.img_chunk(
                    carry, real_chunks[k], chunk_len)
                r = jax_to_torch(reward_j, device).transpose(0, 1).unsqueeze(-1)
                r = r + chunk_config.reward_shift
                c = jax_to_torch(cont_j, device).transpose(0, 1).unsqueeze(-1)
                pooled_r, pooled_c = pool_chunk(r, c, gamma)
                r_draws[k].append(pooled_r.squeeze(-1).cpu().numpy())
                c_draws[k].append(pooled_c.squeeze(-1).cpu().numpy())
                o_draws[k].append(decode_obs(bridge, carry, obs_key, device).cpu().numpy())

    out = {}
    out['wm/obs_abs_mean'] = float(np.abs(obs).mean())
    out['wm/decode_mae_seed'] = float(np.abs(seed_decoded - obs[seq_i, start_i]).mean())

    for k in range(depth):
        stack = np.stack(r_draws[k])
        pred = stack.mean(0)
        true = r_true[k]
        out[f'wm/reward_mae_chunk{k+1}'] = float(np.abs(pred - true).mean())
        out[f'wm/reward_bias_chunk{k+1}'] = float((pred - true).mean())
        out[f'wm/reward_corr_chunk{k+1}'] = _corr(pred, true)
        out[f'wm/reward_true_std_chunk{k+1}'] = float(np.std(true))
        out[f'wm/reward_pred_std_chunk{k+1}'] = float(np.std(pred))
        out[f'wm/decode_mae_chunk{k+1}'] = float(
            np.abs(np.stack(o_draws[k]).mean(0) - obs_true[k]).mean())
        if k == 0:
            # What TRAINING consumes is ONE draw, not the averaged prediction.
            # The gap between these two is pure RSSM sampling noise entering
            # the target.
            single = stack[0]
            out['wm/reward_mae_chunk1_single_draw'] = float(np.abs(single - true).mean())
            spread = stack.std(0) if stack.shape[0] > 1 else np.zeros_like(pred)
            out['wm/uncert_spread_mean'] = float(spread.mean())
            out['wm/uncert_corr'] = _corr(spread, np.abs(pred - true))
            cpred = np.stack(c_draws[k]).mean(0)
            out['wm/cont_mae'] = float(np.abs(cpred - cont_true[k]).mean())
            out['wm/cont_pred_mean'] = float(cpred.mean())
            out['wm/cont_true_mean'] = float(cont_true[k].mean())
    return out

def print_wm_report(m, depth):
    if not m:
        print('[wm] report skipped: replay too short for the rollout span')
        return
    g = lambda k: m.get(k, float('nan'))
    print(f"[wm] decode mae  seed {g('wm/decode_mae_seed'):.4f}  " +
          '  '.join(f"c{k+1} {g(f'wm/decode_mae_chunk{k+1}'):.4f}" for k in range(depth)) +
          f"   (|obs| {g('wm/obs_abs_mean'):.3f})")
    print('[wm] reward mae  ' +
          '  '.join(f"c{k+1} {g(f'wm/reward_mae_chunk{k+1}'):.4f}" for k in range(depth)) +
          f"   1-draw {g('wm/reward_mae_chunk1_single_draw'):.4f}")
    print(f"[wm] reward corr c1 {g('wm/reward_corr_chunk1'):+.3f}"
          f"   true_std {g('wm/reward_true_std_chunk1'):.4f}"
          f"   pred_std {g('wm/reward_pred_std_chunk1'):.4f}")
    print(f"[wm] cont        mae {g('wm/cont_mae'):.4f}"
          f"   pred {g('wm/cont_pred_mean'):.3f}  true {g('wm/cont_true_mean'):.3f}")