""" Accuracy checks for the quantities the model-based arms actually consume.

Every arm reads the model through the same three quantities --

    reward   pooled discounted reward of an imagined chunk
    latent   where the rollout lands, relative to where the real data went
    value    Q_model(z, pi(z)) at the end of the horizon

-- so each has its own number here, measured against real replay:

  wm/reward_corr        correlation of imagined pooled chunk reward with the
                        real pooled reward of the SAME chunk. Selection ranks
                        candidates, so a constant offset is harmless and a
                        lost ordering is fatal: this is the number to watch.
  wm/latent_drift_rel   distance from the rolled latent to the encoding of
                        the real observation it should have landed on,
                        relative to the typical distance between two encoded
                        states. Replaces the decoded-state error of the old
                        RSSM scorer; there is no decode now, so drift is
                        measured where it lives.
  wm/value_critic_corr  correlation between the model's terminal value and
                        the QC critic's own value at the same real state. Low
                        does not by itself mean the model is wrong (different
                        objectives), but with select/pick_agreement it says
                        whether the model's ordering is its own.

Half the windows come from sample_reward_windows (guaranteed to contain an
above-baseline reward step) and half are uniform. Uniform alone makes every
ground-truth reward identical on a sparse task, so every correlation divides
by ~0 and the MAE looks excellent for the trivial reason that predicting a
constant is easy; reward-biased alone can do the same when success ends the
episode (every valid window then ends on the success step). The mix always
has both kinds, so the correlation is defined.

Stated plainly: rollouts here are teacher-forced with the actions in replay,
so this measures accuracy on the data distribution, not on the current
policy's. When the policy is far from the data these numbers flatter the model.
They are still the right gate: a model that cannot predict rewards on data it
was trained on will not predict them on-policy either.

Offline against replay. ZERO environment steps. """

import numpy as np
import torch


def _corr(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if len(a) < 6 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


@torch.no_grad()
def model_report(model, policy, replay, chunk_len, depth, gamma, device, rng,
                 num_windows=256):
    """ depth is in CHUNKS: 1 measures exactly what a single-chunk scorer
        consumes. """
    span = chunk_len * depth
    half = max(num_windows // 2, 3)
    w_hit = replay.sample_reward_windows(half, span, device, rng)
    w_uni = replay.sample_model_windows(half, span, device, rng, online_frac=0.0)
    if w_uni is None:
        return {}
    w = w_uni if w_hit is None else {
        k: torch.cat([w_hit[k], w_uni[k]], dim=0) for k in w_uni}
    # Only windows that stay inside one episode: a masked step has no ground
    # truth, and averaging over it would quietly improve every number here.
    keep = (w['valid'][:, :, 0].min(dim=1).values > 0.5).nonzero().squeeze(-1)
    if len(keep) < 6:
        return {}
    obs, next_obs, act = w['obs'][keep], w['next_obs'][keep], w['action'][keep]
    real_r = w['reward'][keep][:, :, 0].cpu().numpy()

    discounts = gamma ** np.arange(span, dtype=np.float32)
    real_pooled = (real_r * discounts[None, :]).sum(axis=1)

    z = model.encode(obs[:, 0])
    pred_pooled = torch.zeros(len(keep), device=device)
    disc = 1.0
    for t in range(span):
        pred_pooled += disc * model.net.reward_pred(z, act[:, t]).squeeze(-1)
        z = model.net.next(z, act[:, t])
        disc *= gamma
    pred_pooled = pred_pooled.cpu().numpy()

    z_real_end = model.encode(next_obs[:, -1])
    drift = (z - z_real_end).pow(2).sum(-1).sqrt()
    perm = torch.randperm(z_real_end.shape[0], device=device)
    spread = (z_real_end - z_real_end[perm]).pow(2).sum(-1).sqrt().mean()

    model_v = model.terminal_value(z_real_end).squeeze(-1).cpu().numpy()
    critic_v = policy.chunk_target_values(next_obs[:, -1]).squeeze(-1).cpu().numpy()

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
        print('  wm report: skipped (fewer than 6 windows fully inside one episode; raise diag_windows)')
        return
    print(f'  wm report @ depth {depth} chunk(s), {int(m["wm/windows"])} replay windows')
    print(f'    reward  mae {m["wm/reward_mae"]:.4f}  corr {m["wm/reward_corr"]:.3f}'
          f'  (pred std {m["wm/reward_pred_std"]:.3f} vs real {m["wm/reward_real_std"]:.3f})')
    print(f'    latent  drift {m["wm/latent_drift"]:.4f}'
          f'  relative to state spread {m["wm/latent_drift_rel"]:.3f}')
    print(f'    value   mean {m["wm/value_mean"]:.2f}'
          f'  corr with QC critic {m["wm/value_critic_corr"]:.3f}')


@torch.no_grad()
def chunk_q_report(model, policy, replay, chunk_len, device, rng,
                   num_windows=256, hit_thresh=-3.0):
    """ Accuracy of the chunk Q the MVE arm feeds into the critic's TD
        target. ZERO env steps; empty dict unless tdmpc.q_mode=chunk.

    The arm only ever asks the model one question -- how good is this REAL
    state with this chunk -- so every number here is that question, asked on
    real replayed chunks and answered by both value functions:

      diagnosis/chunkq_critic_corr  correlation of the two values on the
                                    same (state, chunk) pairs. The readiness
                                    gauge: the target is only as good as
                                    the model's agreement with a value
                                    function that works.
      diagnosis/chunkq_hit_gap      mean Q on chunks containing a step above
      diagnosis/criticq_hit_gap     hit_thresh, minus mean Q on uniform
                                    chunks, per value function. Does each
                                    one separate progress from nothing, and
                                    by a similar margin? A model Q flat here
                                    while the critic's gap opens cannot
                                    carry the target.
      diagnosis/chunkq_mean/_std,   scale and spread of each. A collapsing
      diagnosis/criticq_mean/_std   chunkq_std is the model Q going
                                    constant.

    Half the windows are drawn to contain an above-hit_thresh step and half
    uniformly: uniform alone makes every ground-truth reward identical on a
    sparse task, so the contrast vanishes. """
    if getattr(model, 'q_mode', 'step') != 'chunk':
        return {}
    half = max(num_windows // 2, 3)

    def grab(w):
        if w is None:
            return None
        keep = (w['valid'][:, :, 0].min(dim=1).values > 0.5).nonzero().squeeze(-1)
        if len(keep) < 3:
            return None
        obs0 = w['obs'][keep][:, 0]
        return obs0, w['action'][keep].reshape(len(keep), -1)

    hit = grab(replay.sample_reward_windows(half, chunk_len, device, rng,
                                            thresh=hit_thresh))
    uni = grab(replay.sample_model_windows(half, chunk_len, device, rng,
                                           online_frac=0.0))
    if uni is None:
        return {}

    def scores(obs0, chunk):
        mq = model.chunk_q(model.encode(obs0), chunk).squeeze(-1)
        cq = policy._agg(policy.critic(obs0, chunk)).squeeze(-1)
        return mq, cq

    mq_u, cq_u = scores(*uni)
    out = {
        'diagnosis/chunkq_mean': float(mq_u.mean()),
        'diagnosis/chunkq_std': float(mq_u.std()),
        'diagnosis/criticq_mean': float(cq_u.mean()),
        'diagnosis/criticq_std': float(cq_u.std()),
        'diagnosis/chunkq_windows': float(mq_u.shape[0]),
    }
    if hit is not None:
        mq_h, cq_h = scores(*hit)
        out['diagnosis/chunkq_hit_gap'] = float(mq_h.mean() - mq_u.mean())
        out['diagnosis/criticq_hit_gap'] = float(cq_h.mean() - cq_u.mean())
        mq, cq = torch.cat([mq_h, mq_u]), torch.cat([cq_h, cq_u])
    else:
        mq, cq = mq_u, cq_u
    out['diagnosis/chunkq_critic_corr'] = _corr(mq.cpu().numpy(), cq.cpu().numpy())

    keys = ('chunkq_critic_corr', 'chunkq_hit_gap', 'criticq_hit_gap',
            'chunkq_std', 'criticq_std')
    print('  chunk-q: ' + '  '.join(f'{k} {out["diagnosis/" + k]:.3f}'
                                    for k in keys if 'diagnosis/' + k in out))
    return out
