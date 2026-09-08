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
                   num_windows=256, select_n=16):
    """ Chunk-mode Q accuracy checks, measured against the QC critic on real
        replay data. ZERO env steps; empty dict unless tdmpc.q_mode=chunk.

    The chunk Q's job is to reproduce the critic's chunk ordering from the
    latent, so every number here is a head-to-head on identical inputs:

      diagnosis/chunkq_critic_corr   correlation of the two values on REAL
                                     replayed chunks (mixed reward-hit +
                                     uniform windows, as model_report mixes)
      diagnosis/chunkq_hit_gap       mean chunk-Q on reward-containing
      diagnosis/criticq_hit_gap      windows minus on uniform windows, per
                                     scorer. Both should be positive once
                                     the value separates success states;
                                     the model matching the critic's gap is
                                     the pre-check that its value carries
                                     the same signal.
      diagnosis/cand_pick_agree      fresh actor candidates (select_n per
      diagnosis/cand_rank_corr       state, the act-time distribution): how
      diagnosis/cand_regret_q        often both scorers pick the same chunk,
                                     their mean per-state Spearman, and what
                                     the critic thinks the model's pick
                                     costs (Q units). The eval-time twin of
                                     the select/ shadow stats -- same
                                     protocol at every eval, in BOTH runs of
                                     a pair.
      diagnosis/prior_minus_actor_q  the critic's value of the model prior's
                                     unrolled chunk minus of the actor's
                                     chunk, same states. Strongly negative =
                                     lazy prior = pessimistic bootstrap in
                                     the chunk TD target (the known weak
                                     link).
      diagnosis/chunkq_mean/_std,    scale and spread of each scorer, for
      diagnosis/criticq_mean/_std    calibration drift. """
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

    hit = grab(replay.sample_reward_windows(half, chunk_len, device, rng))
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

    obs_c = uni[0][:min(64, uni[0].shape[0])]
    M, n = obs_c.shape[0], int(select_n)
    if n >= 2 and M >= 3:
        feat = obs_c.repeat_interleave(n, dim=0)
        cands = policy.sample_chunk(feat)
        mq_c = model.chunk_q(model.encode(feat), cands).squeeze(-1).reshape(M, n)
        cq_c = policy._agg(policy.critic(feat, cands)).squeeze(-1).reshape(M, n)
        m_pick, c_pick = mq_c.argmax(-1), cq_c.argmax(-1)
        r_m = torch.argsort(torch.argsort(mq_c, dim=-1), dim=-1).float()
        r_c = torch.argsort(torch.argsort(cq_c, dim=-1), dim=-1).float()
        rm = r_m - r_m.mean(-1, keepdim=True)
        rc = r_c - r_c.mean(-1, keepdim=True)
        spear = ((rm * rc).mean(-1)
                 / (rm.std(-1, correction=0) * rc.std(-1, correction=0) + 1e-8))
        regret = cq_c.max(-1).values - cq_c.gather(-1, m_pick[:, None]).squeeze(-1)
        out['diagnosis/cand_pick_agree'] = float((m_pick == c_pick).float().mean())
        out['diagnosis/cand_rank_corr'] = float(spear.mean())
        out['diagnosis/cand_regret_q'] = float(regret.mean())

        prior = model.prior_chunk(model.encode(obs_c), sample=False)
        pq = policy._agg(policy.critic(obs_c, prior)).squeeze(-1)
        aq = policy._agg(policy.critic(obs_c, policy.sample_chunk(obs_c))).squeeze(-1)
        out['diagnosis/prior_minus_actor_q'] = float((pq - aq).mean())

    keys = ('cand_pick_agree', 'cand_rank_corr', 'chunkq_critic_corr',
            'chunkq_hit_gap', 'criticq_hit_gap', 'prior_minus_actor_q')
    line = '  '.join(f'{k} {out["diagnosis/" + k]:.3f}'
                     for k in keys if 'diagnosis/' + k in out)
    print(f'  chunk-q: {line}')
    return out
