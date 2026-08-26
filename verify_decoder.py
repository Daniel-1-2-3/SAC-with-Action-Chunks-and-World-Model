"""Post-mortem checks for the decoded-state pathway that MVE and Dyna consumed.

Runs against a saved world-model checkpoint (wm_latest.pkl) and, optionally, a
saved policy checkpoint (chunk_latest.pt), using the offline dataset for real
states. No env stepping, no training, no imports from any trainer -- works on
any of the three pod builds.

    python verify_decoder.py --wm_ckpt out_dir/wm_latest.pkl \
        [--chunk_ckpt out_dir/chunk_latest.pt] [--n 512]

Checks, and what each verdict means:
  [1] batched-vs-single decode equality  -> differ = the batched decode_feat
      is corrupting rows (a code bug, not a weak decoder)
  [2] per-dim regression slope/intercept -> slopes far from 1 = wrong output
      space/scale (bug); slopes ~1 = decoder is in the right space
  [3] per-dim MAE vs per-dim std         -> error concentrated in a few dims
      = the headline MAE overstates the damage
  [4] Q(real) vs Q(decoded) gap          -> small next to critic RMSE = decode
      error did NOT corrupt values and the MVE/Dyna failure lies elsewhere;
      large = mechanism confirmed; positive sign = optimism bias
  [5] model reward: replay chunk vs policy chunk at the same states
      -> policy chunks scored systematically higher = the model flatters
      actions it never saw executed (off-distribution optimism)
"""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('MUJOCO_GL', 'egl')

import argparse
import pathlib
import elements
import numpy as np
import ruamel.yaml as yaml
import torch

from dreamer.wm_agent import WorldModelAgent
from dreamer.wm_bridge import WorldModelBridge
from helpers.ogbench_methods import OGBenchMethods
from sac_chunked.sac_chunk_agent import ChunkAgent

OBS_KEY = 'state'
ACTION_KEY = 'action'

def pooled(reward, cont, gamma):
    # numpy mirror of chunk_utils.pool_chunk: (n, chunk_len) -> (n,), (n,)
    n, h = reward.shape
    out = np.zeros(n, np.float64)
    alive = np.ones(n, np.float64)
    disc = 1.0
    for k in range(h):
        out += disc * alive * reward[:, k]
        alive *= cont[:, k]
        disc *= gamma
    return out, alive

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wm_ckpt', required=True)
    ap.add_argument('--chunk_ckpt', default=None)
    ap.add_argument('--n', type=int, default=512)
    args = ap.parse_args()

    folder = pathlib.Path(__file__).parent
    configs = yaml.YAML(typ='safe').load(elements.Path(folder / 'configs.yaml').read())
    config = elements.Config(configs['defaults'])
    g = config.train_sac_chunked_wm.general
    c = config.train_sac_chunked_wm.chunk
    seq_len, wm_batch = config.batch_length, config.batch_size
    chunk_len, gamma = c.chunk_len, c.gamma
    rng = np.random.default_rng(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env, train_dataset, _ = OGBenchMethods.load_ogbench(g.env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    obs_space, act_space = OGBenchMethods.make_spaces(obs_dim, action_dim, OBS_KEY, ACTION_KEY)
    agent_config = elements.Config(
        **config.agent, logdir='/tmp/verify_wm', seed=config.seed, jax=config.jax,
        batch_size=wm_batch, batch_length=seq_len, replay_context=0,
        report_length=seq_len, replica=0, replicas=1)
    wm_agent = WorldModelAgent(obs_space, act_space, agent_config)
    cp = elements.Checkpoint(pathlib.Path(args.wm_ckpt))
    cp.agent = wm_agent
    cp.load()
    bridge = WorldModelBridge(wm_agent, ACTION_KEY, obs_key=OBS_KEY)

    eps = OGBenchMethods.make_dreamer_episodes(
        train_dataset, min_length=seq_len, obs_key=OBS_KEY, action_key=ACTION_KEY)
    try:
        batch = OGBenchMethods.sample_dreamer_batch(
            eps, wm_batch, seq_len, obs_key=OBS_KEY, action_key=ACTION_KEY,
            rng=rng, bias_start_to_reward=True, reward_thresh=-2.5)
    except TypeError:
        batch = OGBenchMethods.sample_dreamer_batch(
            eps, wm_batch, seq_len, obs_key=OBS_KEY, action_key=ACTION_KEY, rng=rng)

    obs = np.asarray(batch[OBS_KEY], np.float32)
    act = np.asarray(batch[ACTION_KEY], np.float32)
    n_seq, t_len, _ = obs.shape
    obs_flat = obs.reshape(-1, obs_dim)
    pool = bridge.seed_pool(OGBenchMethods.to_jax(batch), wm_batch)

    starts = np.arange(t_len - chunk_len)
    sgrid, tgrid = np.meshgrid(np.arange(n_seq), starts, indexing='ij')
    cand = (sgrid.ravel() * t_len + tgrid.ravel()).astype(np.int64)
    idx = rng.choice(cand, size=min(args.n, len(cand)), replace=False)
    real = obs_flat[idx]

    carry = bridge.place_seed({k: v[idx] for k, v in pool.items()})
    dec = np.asarray(bridge.decode_state(carry)[OBS_KEY], np.float32)

    print('=' * 72)
    print(f'[1] batched vs single decode ({min(8, len(idx))} spot rows)')
    worst = 0.0
    for i in rng.choice(len(idx), size=min(8, len(idx)), replace=False):
        c1 = bridge.place_seed({k: v[idx[i:i+1]] for k, v in pool.items()})
        d1 = np.asarray(bridge.decode_state(c1)[OBS_KEY], np.float32)[0]
        worst = max(worst, float(np.abs(d1 - dec[i]).max()))
    print(f'    max |single - batched| = {worst:.6f}')
    print('    VERDICT: ' + ('batching bug -- rows differ' if worst > 1e-3
                             else 'batched path consistent with single'))

    print('=' * 72)
    print('[2] per-dim regression decoded = a*real + b  (space/scale bug check)')
    slopes, inters = np.zeros(obs_dim), np.zeros(obs_dim)
    for d in range(obs_dim):
        if np.std(real[:, d]) < 1e-6:
            slopes[d], inters[d] = 1.0, 0.0
            continue
        slopes[d], inters[d] = np.polyfit(real[:, d], dec[:, d], 1)
    bad = np.flatnonzero((slopes < 0.7) | (slopes > 1.3))
    print(f'    slope mean {slopes.mean():.3f} | dims with slope outside [0.7,1.3]: '
          f'{len(bad)}/{obs_dim} -> {bad[:12].tolist()}')
    print('    VERDICT: ' + ('wrong output space/scale on flagged dims'
                             if len(bad) > obs_dim // 3 else
                             'output space looks right'))

    print('=' * 72)
    print('[3] per-dim MAE vs per-dim std (where the 0.27 headline lives)')
    mae_d = np.abs(dec - real).mean(0)
    std_d = real.std(0) + 1e-8
    order = np.argsort(-mae_d)
    print(f'    overall MAE {mae_d.mean():.4f} | obs |mean| {np.abs(real).mean():.4f}')
    print(f'    top-8 dims by MAE (dim: mae / std):')
    for d in order[:8]:
        print(f'      dim {d:3d}: {mae_d[d]:.4f} / {std_d[d]:.4f}')
    share = mae_d[order[:8]].sum() / max(mae_d.sum(), 1e-8)
    print(f'    top-8 dims carry {share:.0%} of total error')

    if args.chunk_ckpt:
        print('=' * 72)
        print('[4] Q(real) vs Q(decoded), same chunk -- the mechanism test')
        policy = ChunkAgent(
            repr_dim=obs_dim, action_dim=action_dim, chunk_len=chunk_len,
            device=device, lr=c.lr, hidden_dim=c.hidden_dim,
            num_layers=c.num_layers, critic_target_tau=c.critic_target_tau,
            ensemble=c.ensemble, alpha=c.alpha, flow_steps=c.flow_steps,
            q_agg=c.q_agg, compile_nets=False)
        _raw = torch.load(args.chunk_ckpt, map_location=device)
        # Checkpoints saved with compile_nets=True carry torch.compile's
        # "_orig_mod." prefix; this script builds uncompiled modules.
        _raw = {k: {kk.replace('_orig_mod.', ''): vv for kk, vv in v.items()}
                for k, v in _raw.items()}
        policy.load_state_dict_all(_raw)
        rt = torch.as_tensor(real, device=device)
        dt = torch.as_tensor(dec, device=device)
        with torch.no_grad():
            a = policy.sample_chunk(rt)
            qr = policy._agg(policy.critic(rt, a)).squeeze(-1)
            qd = policy._agg(policy.critic(dt, a)).squeeze(-1)
        gap = (qd - qr).cpu().numpy()
        print(f'    mean |Q(dec)-Q(real)| = {np.abs(gap).mean():.3f}   '
              f'signed mean = {gap.mean():+.3f}   max |gap| = {np.abs(gap).max():.3f}')
        print(f'    reference Q scale: mean Q(real) = {qr.mean().item():.1f}')
        print('    VERDICT: gap >> critic RMSE (~sqrt(critic_loss/2)) = decode '
              'corrupts values; gap << it = failure lies elsewhere; '
              'positive signed mean = optimism the bootstrap amplifies')

        print('=' * 72)
        print('[5] model reward: replay chunk vs policy chunk at same states')
        t0 = (idx % t_len)
        awin = t0[:, None] + np.arange(chunk_len)[None, :]
        replay_chunks = act[idx // t_len][np.arange(len(idx))[:, None], awin]
        replay_chunks = replay_chunks.reshape(len(idx), chunk_len * action_dim)
        with torch.no_grad():
            policy_chunks = policy.sample_chunk(rt).cpu().numpy()
        out = {}
        for name, chunks in (('replay', replay_chunks), ('policy', policy_chunks)):
            cseed = bridge.place_seed({k: v[idx] for k, v in pool.items()})
            _, _, rj, cj = bridge.img_chunk(cseed, chunks.astype(np.float32), chunk_len)
            pr, pc = pooled(np.asarray(jax.device_get(rj), np.float64),
                            np.asarray(jax.device_get(cj), np.float64), gamma)
            out[name] = pr
            print(f'    {name} chunks: model pooled reward mean {pr.mean():+.3f} '
                  f'(cont mean {pc.mean():.3f})')
        diff = out['policy'] - out['replay']
        print(f'    mean(policy - replay) = {diff.mean():+.3f}   '
              f'frac policy-scored-higher = {(diff > 0).mean():.2f}')
        print('    VERDICT: large positive mean = model flatters unexecuted '
              'actions (off-distribution optimism)')
    else:
        print('=' * 72)
        print('[4/5] skipped: pass --chunk_ckpt to run the Q-gap and '
              'policy-chunk optimism checks')

    env.close()

if __name__ == '__main__':
    main()