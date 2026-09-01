""" Accuracy checks for the two quantities the chunk score actually consumes.

The score is

    sum_t gamma^t * r_model(z_t, a_t)  +  gamma^H * Q_model(z_H, pi(z_H))

so exactly three things can be wrong with it, and each has its own number
here:

  reward     the pooled discounted reward of an imagined chunk, against the
             real pooled reward of the same chunk in replay. MAE and
             correlation. Correlation is the one that matters -- selection
             ranks candidates, so a constant offset is harmless and a lost
             ordering is fatal.
  latent     how far the rolled latent has drifted from the encoding of the
             real observation it should have landed on, relative to the
             typical distance between two encoded states. This replaces the
             decoded-state error of the previous model: there is no decode
             any more, so drift is measured where it actually lives.
  value      correlation between the model's terminal Q(z_H, pi(z_H)) and the
             QC critic's own value at the same real state. Low correlation
             does not by itself mean the model is wrong -- the two are trained
             on different objectives -- but together with pick_agreement it
             says whether the model's value ordering is its own or the
             critic's.

Why this runs against REPLAY and not the live environment: driving the real
env for ground truth spends env steps that are not in the training budget and
perturbs the same env the collection loop uses. Windows are sampled with
bias_start_to_reward so they are guaranteed to contain reward variation --
without that every ground-truth reward is identical, every correlation divides
by ~0, and the reward MAE looks excellent for the trivial reason that
predicting a constant is easy.

The tradeoff, stated plainly: rollouts here are teacher-forced with the
actions in replay, so this measures model accuracy on the data distribution,
not on the current policy's distribution. When the policy is far from the data
these numbers are optimistic. They are still the right gate, because a model
that cannot predict rewards on data it was trained on will not predict them
on-policy either.

Everything here is offline against replay and spends ZERO environment steps.
"""

import numpy as np
import torch

from tdmpc.data import latent_rollout_windows


def _corr(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if len(a) < 6 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


@torch.no_grad()
def model_report(model, policy, replay, chunk_len, depth, gamma, device, rng,
                 batch_size, seq_len, obs_key='state', action_key='action',
                 num_windows=256):
    """ depth is measured in CHUNKS. The selector only ever imagines
        rollout_chunks ahead, so depth=1 measures exactly what it consumes. """
    span = chunk_len * depth
    if num_windows <= 0 or seq_len < span + 1 or not replay.ready(seq_len):
        return {}

    batch_np = replay.sample_batch(batch_size, seq_len, rng=rng,
                                   bias_start_to_reward=True)
    w = latent_rollout_windows(batch_np, span, obs_key=obs_key,
                               action_key=action_key)
    if w is None or len(w['obs']) == 0:
        return {}

    # Only windows that stay inside one episode: a masked-out step has no
    # ground truth to compare against, and averaging over it would quietly
    # improve every number here.
    keep = np.flatnonzero(w['valid'][:, :, 0].min(axis=1) > 0.5)
    if len(keep) < 6:
        return {}
    if len(keep) > num_windows:
        keep = rng.choice(keep, size=num_windows, replace=False)

    to = lambda x: torch.as_tensor(x[keep], device=device).float()
    obs, act = to(w['obs']), to(w['action'])
    real_r = to(w['reward'])[:, :, 0].cpu().numpy()

    discounts = gamma ** np.arange(span, dtype=np.float32)
    real_pooled = (real_r * discounts[None, :]).sum(axis=1)

    z = model.encode(obs[:, 0])
    pred_pooled = np.zeros(len(keep), dtype=np.float32)
    disc = 1.0
    for t in range(span):
        pred_pooled += disc * model.net.reward_pred(z, act[:, t]).squeeze(-1).cpu().numpy()
        z = model.net.next(z, act[:, t])
        disc *= gamma

    z_real_end = model.encode(obs[:, span])
    drift = (z - z_real_end).pow(2).sum(-1).sqrt()
    # Scale reference: the mean distance between two DIFFERENT encoded real
    # states in this batch. Raw latent distance is meaningless on its own
    # (SimNorm fixes the latent's scale, not its spread); drift relative to
    # this says whether the rollout still points at the right state.
    perm = torch.randperm(z_real_end.shape[0], device=device)
    spread = (z_real_end - z_real_end[perm]).pow(2).sum(-1).sqrt().mean()

    model_v = model.terminal_value(z_real_end).squeeze(-1).cpu().numpy()
    critic_v = policy.chunk_target_values(obs[:, span]).squeeze(-1).cpu().numpy()

    return {
        'wm/reward_mae': float(np.abs(pred_pooled - real_pooled).mean()),
        'wm/reward_corr': _corr(pred_pooled, real_pooled),
        'wm/reward_pred_std': float(np.std(pred_pooled)),
        'wm/reward_real_std': float(np.std(real_pooled)),
        'wm/latent_drift': float(drift.mean().item()),
        'wm/latent_drift_rel': float((drift.mean() / (spread + 1e-8)).item()),
        'wm/value_critic_corr': _corr(model_v, critic_v),
        'wm/value_mean': float(np.mean(model_v)),
        'wm/windows': float(len(keep)),
    }


def print_wm_report(m, depth):
    if not m:
        print('  wm report: skipped (not enough replay yet)')
        return
    print(f'  wm report @ depth {depth} chunk(s), {int(m["wm/windows"])} replay windows')
    print(f'    reward  mae {m["wm/reward_mae"]:.4f}  corr {m["wm/reward_corr"]:.3f}'
          f'  (pred std {m["wm/reward_pred_std"]:.3f} vs real {m["wm/reward_real_std"]:.3f})')
    print(f'    latent  drift {m["wm/latent_drift"]:.4f}'
          f'  relative to state spread {m["wm/latent_drift_rel"]:.3f}')
    print(f'    value   mean {m["wm/value_mean"]:.2f}'
          f'  corr with QC critic {m["wm/value_critic_corr"]:.3f}')
