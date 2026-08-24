"""World-model accuracy checks, run at every eval.

Reports whether the model is good enough for MVE to help, and enough about it
to judge other integration styles later without another run.

Groups, and what each decides:

  reward/*      Teacher-forced reward accuracy across states (the "tf_corr"
                check). Force the model with the chunk the policy actually
                executed, compare predicted pooled reward against what really
                happened. Gates everything -- MVE consumes predicted rewards
                directly.

                reward/std_ratio matters as much as reward/corr: a head that
                collapses to predicting the mean can still show a nonzero
                correlation while contributing nothing. std_ratio well below 1
                is that failure.

  depth/*       Same check at rollout depth 1..N, using the chunks the policy
                actually took. This is exactly what MVE consumes, since it
                appends num_chunks imagined chunks to the target. If accuracy
                collapses by chunk 2, the deep part of the target is noise.

  mve/*         The break-even. MVE trades reliance on the critic for reliance
                on the model. At gamma=0.99 the bootstrap coefficient only
                moves 0.951 -> 0.818 even at depth 3, so the purchase is small
                while model error is paid at near-full weight. Working through
                the variances, MVE helps when

                    reward_std * sqrt(1 - corr^2)  <  0.309 * critic_err

                The 0.309 is invariant to depth: deeper rollouts buy more
                bootstrap reduction but accumulate proportionally more error,
                and the two cancel. mve/required_corr inverts this against the
                critic's measured TD error; mve/margin is depth-1 corr minus
                that. Positive margin => MVE should help.

  cont/*        Continue-head accuracy. A wrong cont silently rescales every
                pooled reward and every bootstrap.

  sensitivity/* Does the prediction move the same direction as reality when
                the ACTION changes? Finite differences, so no autodiff across
                the JAX/PyTorch boundary. Gates analytic-gradient approaches.

  uncert/*      Does spread across prior draws predict the model's own error?
                Gates uncertainty-weighted variants (STEVE) and exploration
                bonuses.

Every metric ships with its sample count and standard error. A correlation
from a handful of states has SE near 0.1, where anything below 0.2 is
indistinguishable from zero -- counts are reported so that ambiguity is
visible rather than assumed away.

Env steps spent here are diagnostic and are NOT added to the training budget.
"""
import jax
import numpy as np
import torch

from sac_chunked.chunk_utils import pool_chunk
from helpers.interop import jax_to_torch

ENV_LOW, ENV_HIGH = -1.0, 1.0

# ---------------------------------------------------------------- sim state

def sim_snapshot(env):
    """MuJoCo qpos/qvel snapshot, or None if the env does not expose them.
    Needed to replay a perturbed action from an identical start state."""
    u = env
    for _ in range(8):
        if hasattr(u, 'data') and hasattr(getattr(u, 'data', None), 'qpos'):
            break
        u = getattr(u, 'env', getattr(u, 'unwrapped', None))
        if u is None:
            return None
    if not (hasattr(u, 'data') and hasattr(u.data, 'qpos')):
        return None
    return (u, u.data.qpos.copy(), u.data.qvel.copy())

def sim_restore(snap):
    u, qpos, qvel = snap
    u.data.qpos[:] = qpos
    u.data.qvel[:] = qvel
    try:
        import mujoco
        mujoco.mj_forward(u.model, u.data)
    except Exception:
        if hasattr(u, 'set_state'):
            u.set_state(qpos, qvel)

# ---------------------------------------------------------------- stats

def _corr(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    # <6 points is not a correlation; n=3 returns +-1 for any monotone data
    if len(a) < 6 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])

def _se(n):
    return float('nan') if n < 4 else 1.0 / np.sqrt(n - 3)

# ---------------------------------------------------------------- report

def wm_report(env, bridge, policy, chunk_config, chunk_len, gamma, device, rng,
              action_dim, num_states=100, depth=3, model_samples=4,
              sens_eps=0.3, critic_loss=None):
    gamma_h = gamma ** chunk_len
    out = {}

    obs, _ = env.reset()
    if sim_snapshot(env) is None:
        print('[wm_diag] SKIPPED: env exposes no MuJoCo qpos/qvel')
        return {'wm_diag/unavailable': 1.0}

    enc, dyn = bridge.init_encode(1)
    prevact = np.zeros((1, action_dim), dtype=np.float32)
    is_first = np.array([True])
    step_env = lambda a: env.step(ENV_LOW + (a + 1.0) * 0.5 * (ENV_HIGH - ENV_LOW))

    def roll(carry_host, chunks_np, n_samples):
        """Average n_samples prior draws. RSSM.imagine samples stoch at every
        step, so ONE rollout is a single Monte Carlo draw -- correlating one
        draw understates accuracy badly (a model at true 0.7 reads ~0.40).
        Returns (mean pooled reward, mean cont product, spread, final latent)."""
        ps, cs, last = [], [], None
        for _ in range(max(1, n_samples)):
            nxt, feats_j, rew_j, cont_j = bridge.img_chunk(
                bridge.place_seed(carry_host), chunks_np, chunk_len)
            r = jax_to_torch(rew_j, device).transpose(0, 1).unsqueeze(-1) + chunk_config.reward_shift
            c = jax_to_torch(cont_j, device).transpose(0, 1).unsqueeze(-1)
            p, cp = pool_chunk(r, c, gamma)
            ps.append(p); cs.append(cp)
            last = (jax_to_torch(feats_j, device)[:, -1], nxt)
        P = torch.stack(ps)
        return (P.mean(0).squeeze(-1).cpu().numpy(),
                torch.stack(cs).mean(0).squeeze(-1).cpu().numpy(),
                (P.std(0).squeeze(-1).cpu().numpy() if len(ps) > 1
                 else np.zeros(P.shape[1])),
                last)

    r_pred, r_true = [], []
    d_pred = [[] for _ in range(depth)]
    d_true = [[] for _ in range(depth)]
    c_pred, c_true = [], []
    s_dp, s_dt = [], []
    u_sp, u_err = [], []

    tries = 0
    while len(r_pred) < num_states and tries < num_states * 5:
        tries += 1
        state = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        enc, dyn, feat_j = bridge.encode_step(enc, dyn, state, prevact, is_first)
        # dyn is a live DEVICE carry; device_get before any host-side use or
        # the JAX transfer guard rejects the op.
        carry_h = {k: np.asarray(jax.device_get(v)) for k, v in dyn.items()}
        snap = sim_snapshot(env)

        feat_np = np.asarray(jax.device_get(feat_j))[0].copy()
        chunk = policy.act(feat_np, eval_mode=False)

        with torch.no_grad():
            p0, cp0, sp0, _ = roll(carry_h, chunk.reshape(1, -1), model_samples)

        # ---- action sensitivity: perturb, does prediction move like reality?
        pert = np.clip(chunk + rng.normal(0, sens_eps, chunk.shape), -1, 1)
        with torch.no_grad():
            pp, _, _, _ = roll(carry_h, pert.reshape(1, -1), model_samples)
        sim_restore(snap)
        tot_p, disc = 0.0, 1.0
        for k in range(chunk_len):
            _, rw, te, tr, _ = step_env(pert[k])
            tot_p += disc * rw; disc *= gamma
            if te or tr:
                break

        # ---- depth: roll `depth` chunks the policy actually takes, comparing
        # the model's prediction at each depth against the real outcome.
        sim_restore(snap)
        e2, d2 = bridge.init_encode(1)
        pa = np.zeros((1, action_dim), dtype=np.float32)
        fi = np.array([True])
        ch_carry, ok = carry_h, True
        f2 = feat_j
        for k in range(depth):
            ck = policy.act(np.asarray(jax.device_get(f2))[0].copy(), eval_mode=False)
            with torch.no_grad():
                pk, _, _, lastk = roll(ch_carry, ck.reshape(1, -1), 1)
            tot, disc, dead, o2 = 0.0, 1.0, False, None
            for j in range(chunk_len):
                o2, rw, te, tr, _ = step_env(ck[j])
                tot += disc * rw; disc *= gamma
                pa = ck[j].reshape(1, -1).astype(np.float32)
                if te or tr:
                    dead = True; break
            d_pred[k].append(float(pk[0])); d_true[k].append(tot)
            if dead:
                break
            e2, d2, f2 = bridge.encode_step(
                e2, d2, np.asarray(o2, dtype=np.float32).reshape(1, -1), pa, fi)
            fi = np.array([False])
            ch_carry = {kk: np.asarray(jax.device_get(vv)) for kk, vv in lastk[1].items()}

        # ---- execute the policy chunk for real (reward-head ground truth)
        sim_restore(snap)
        tot, disc, done = 0.0, 1.0, False
        for k in range(chunk_len):
            obs, rw, te, tr, _ = step_env(chunk[k])
            tot += disc * rw; disc *= gamma
            prevact = chunk[k].reshape(1, -1).astype(np.float32)
            is_first = np.array([False])
            if te or tr:
                done = True; break

        r_pred.append(float(p0[0])); r_true.append(tot)
        c_pred.append(float(cp0[0])); c_true.append(0.0 if done else 1.0)
        u_sp.append(float(sp0[0])); u_err.append(abs(float(p0[0]) - tot))
        s_dp.append(float(pp[0] - p0[0])); s_dt.append(tot_p - tot)

        if done:
            obs, _ = env.reset()
            enc, dyn = bridge.init_encode(1)
            prevact = np.zeros((1, action_dim), dtype=np.float32)
            is_first = np.array([True])

    n = len(r_pred)
    if n < 6:
        print(f'[wm_diag] SKIPPED: only {n} usable states after {tries} tries')
        return {'wm_diag/unavailable': 2.0, 'wm_diag/states': float(n)}

    out['wm_diag/states'] = float(n)
    out['reward/corr'] = _corr(r_pred, r_true)
    out['reward/se'] = _se(n)
    out['reward/pred_std'] = float(np.std(r_pred))
    out['reward/true_std'] = float(np.std(r_true))
    out['reward/mae'] = float(np.mean(np.abs(np.array(r_pred) - np.array(r_true))))
    # <1 => the head hedges toward the mean; >1 => it overshoots
    out['reward/std_ratio'] = float(np.std(r_pred) / max(np.std(r_true), 1e-8))

    for k in range(depth):
        if len(d_pred[k]) >= 6:
            out[f'depth/corr_chunk{k+1}'] = _corr(d_pred[k], d_true[k])
            out[f'depth/mae_chunk{k+1}'] = float(np.mean(
                np.abs(np.array(d_pred[k]) - np.array(d_true[k]))))
            out[f'depth/n_chunk{k+1}'] = float(len(d_pred[k]))
    if 'depth/corr_chunk1' in out and f'depth/corr_chunk{depth}' in out:
        out['depth/decay'] = out['depth/corr_chunk1'] - out[f'depth/corr_chunk{depth}']

    out['cont/corr'] = _corr(c_pred, c_true)
    out['cont/pred_mean'] = float(np.mean(c_pred))
    out['cont/true_mean'] = float(np.mean(c_true))

    out['sensitivity/pred_delta_abs'] = float(np.mean(np.abs(s_dp)))
    out['sensitivity/true_delta_abs'] = float(np.mean(np.abs(s_dt)))
    out['sensitivity/direction_corr'] = _corr(s_dp, s_dt)
    out['sensitivity/se'] = _se(n)

    out['uncert/spread_mean'] = float(np.mean(u_sp))
    out['uncert/spread_vs_error_corr'] = _corr(u_sp, u_err)
    out['uncert/se'] = _se(n)

    # ---- MVE break-even
    out['mve/K'] = 0.309
    c1 = out.get('depth/corr_chunk1', float('nan'))
    if critic_loss is not None and out['reward/true_std'] > 1e-8:
        # critic_loss sums `ensemble` weighted MSE terms -> RMSE per member
        ce = float(np.sqrt(max(critic_loss, 0.0) / max(chunk_config.ensemble, 1)))
        ratio = ce / out['reward/true_std']
        out['mve/critic_err'] = ce
        out['mve/critic_err_over_reward_std'] = ratio
        x = min(0.309 * ratio, 1.0)
        out['mve/required_corr'] = float(np.sqrt(max(1 - x * x, 0.0)))
        if not np.isnan(c1):
            out['mve/margin'] = c1 - out['mve/required_corr']
    return out

def print_wm_report(m):
    if 'wm_diag/unavailable' in m:
        return
    g = lambda k: m.get(k, float('nan'))
    print(f"[wm_diag] {g('wm_diag/states'):.0f} states")
    print(f"  reward       corr {g('reward/corr'):+.3f} +/-{g('reward/se'):.3f}"
          f"   std_ratio {g('reward/std_ratio'):.2f}   mae {g('reward/mae'):.2f}")
    ds = [f"c{k}:{g(f'depth/corr_chunk{k}'):+.2f}" for k in (1, 2, 3)
          if f'depth/corr_chunk{k}' in m]
    if ds:
        print(f"  depth        {'   '.join(ds)}    decay {g('depth/decay'):+.3f}")
    if 'mve/required_corr' in m:
        mg = g('mve/margin')
        verdict = 'MVE should help' if mg > 0 else 'MVE likely hurts'
        print(f"  mve          corr {g('depth/corr_chunk1'):+.3f} vs required "
              f"{g('mve/required_corr'):.3f}   margin {mg:+.3f}   -> {verdict}")
    print(f"  cont         corr {g('cont/corr'):+.3f}"
          f"   pred {g('cont/pred_mean'):.3f}  true {g('cont/true_mean'):.3f}")
    print(f"  sensitivity  dir_corr {g('sensitivity/direction_corr'):+.3f}"
          f" +/-{g('sensitivity/se'):.3f}")
    print(f"  uncert       spread_vs_error {g('uncert/spread_vs_error_corr'):+.3f}"
          f" +/-{g('uncert/se'):.3f}")