import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('MUJOCO_GL', 'egl')

import argparse
import pathlib
import elements
import jax
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dreamer.wm_agent import WorldModelAgent
from dreamer.wm_bridge import WorldModelBridge
from sac_wm_agent import SACWorldModelAgent
from sac_wm_utils import set_seed_everywhere
from interop import extract_state
from ogbench_methods import OGBenchMethods
from train_joint import load_config, build_agent_config

OBS_KEY = 'state'
ACTION_KEY = 'action'
ENV_ACTION_LOW = -1.0
ENV_ACTION_HIGH = 1.0


def run_eval(env, policy, n_episodes, tag, bridge=None, action_dim=None, seed=0):
    """ bridge=None  -> pure SAC, actor reads the raw observation.
        bridge given -> WM+SAC, observation is encoded through the RSSM first. """
    returns, successes, steps_to_success = [], [], []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep) # same episode set for both arms
        if bridge is not None:
            enc_carry, dyn_carry = bridge.init_encode(1)
            prevact = np.zeros((1, action_dim), dtype=np.float32)
            is_first = np.array([True])

        done = False
        ep_return, t, ep_success, hit_at = 0.0, 0, False, None

        while not done:
            state = extract_state(obs, OBS_KEY)

            if bridge is None:
                feat_np = state.reshape(-1)
            else:
                enc_carry, dyn_carry, feat_j = bridge.encode_step(
                    enc_carry, dyn_carry, state, prevact, is_first)
                # JAX blocks implicit device-to-host transfers, so np.asarray
                # on a device array raises. Same explicit device_get that
                # evaluation.py and train_joint.py use.
                feat_np = np.asarray(jax.device_get(feat_j))[0].copy()

            action = policy.act(feat_np, step=0, eval_mode=True)
            env_action = ENV_ACTION_LOW + (action + 1.0) * 0.5 * (ENV_ACTION_HIGH - ENV_ACTION_LOW)
            next_obs, reward, terminated, truncated, info = env.step(env_action)

            if bridge is not None:
                prevact = action.reshape(1, -1).astype(np.float32)
                is_first = np.array([False])

            done = bool(terminated or truncated)
            ep_return += float(reward)
            t += 1
            if bool(info.get('success', False)) and not ep_success:
                ep_success = True
                hit_at = t

            obs = next_obs

        returns.append(ep_return)
        successes.append(float(ep_success))

        # nan for failed episodes so the array stays aligned with `eps` and
        # matplotlib simply leaves a gap where there was no success
        steps_to_success.append(hit_at if hit_at is not None else np.nan)

        if (ep + 1) % 50 == 0:
            print(f'    {ep+1}/{n_episodes} | running success {np.mean(successes):.3f}')

    return np.array(returns), np.array(successes), np.array(steps_to_success)


def load_sac_only(ckpt, env, config, device):
    sac = config.joint.sac
    policy = SACWorldModelAgent(
        repr_dim=env.observation_space.shape[0],
        action_shape=(env.action_space.shape[0],), device=device,
        lr=sac.lr, feature_dim=sac.feature_dim, hidden_dim=sac.hidden_dim,
        critic_target_tau=sac.critic_target_tau, init_ent_coef=sac.init_ent_coef)
    policy.load_state_dict_all(torch.load(ckpt, map_location=device))
    return policy


def load_wm_sac(sac_ckpt, wm_ckpt, env, config, device):
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    obs_space, act_space = OGBenchMethods.make_spaces(
        obs_dim, action_dim, OBS_KEY, ACTION_KEY)

    agent_config = build_agent_config(
        config, 1, config.batch_length, pathlib.Path(wm_ckpt).parent / 'wm_ckpts')
    wm_agent = WorldModelAgent(obs_space, act_space, agent_config)

    wm_cp = elements.Checkpoint(pathlib.Path(wm_ckpt))
    wm_cp.agent = wm_agent
    wm_cp.load()
    bridge = WorldModelBridge(wm_agent, ACTION_KEY, obs_key=OBS_KEY)

    rssm = agent_config.dyn.rssm
    feat_dim = int(rssm.deter + rssm.stoch * rssm.classes)
    sac = config.joint.sac
    policy = SACWorldModelAgent(
        repr_dim=feat_dim, action_shape=(action_dim,), device=device,
        lr=sac.lr, feature_dim=sac.feature_dim, hidden_dim=sac.hidden_dim,
        critic_target_tau=sac.critic_target_tau, init_ent_coef=sac.init_ent_coef)
    policy.load_state_dict_all(torch.load(sac_ckpt, map_location=device))
    return policy, bridge


def plot_all(out_dir, sac, wm, n):
    """ sac / wm are (returns, successes, steps_to_success) tuples. """
    out_dir.mkdir(parents=True, exist_ok=True)
    sac_ret, sac_succ, sac_sts = sac
    wm_ret, wm_succ, wm_sts = wm
    eps = np.arange(n)
    C_SAC, C_WM = 'tab:orange', 'tab:blue'

    # 1. per-episode return
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(eps, sac_ret, color=C_SAC, alpha=0.45, lw=0.8, label='Pure SAC')
    ax.plot(eps, wm_ret, color=C_WM, alpha=0.45, lw=0.8, label='WM + SAC')
    ax.axhline(sac_ret.mean(), color=C_SAC, ls='--', lw=1.5)
    ax.axhline(wm_ret.mean(), color=C_WM, ls='--', lw=1.5)
    ax.set_xlabel('episode'); ax.set_ylabel('return')
    ax.set_title(f'Episode return over {n} episodes (dashed = mean)')
    ax.legend(); fig.tight_layout()
    fig.savefig(out_dir / 'episode_return.png', dpi=120); plt.close(fig)

    # 2. steps to success -- only episodes that succeeded, so gaps are real
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(eps, sac_sts, '.', color=C_SAC, ms=4, label='Pure SAC')
    ax.plot(eps, wm_sts, '.', color=C_WM, ms=4, label='WM + SAC')
    ax.set_xlabel('episode'); ax.set_ylabel('steps to success')
    ax.set_title('Steps to success (only successful episodes plotted)')
    ax.legend(); fig.tight_layout()
    fig.savefig(out_dir / 'steps_to_success.png', dpi=120); plt.close(fig)

    # 3. success rate + mean return bars
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4))
    a1.bar(['Pure SAC', 'WM + SAC'], [sac_succ.mean(), wm_succ.mean()],
           color=[C_SAC, C_WM])
    a1.set_ylim(0, 1); a1.set_ylabel('success rate')
    for i, v in enumerate([sac_succ.mean(), wm_succ.mean()]):
        a1.text(i, v + 0.02, f'{v:.3f}', ha='center')
    a2.bar(['Pure SAC', 'WM + SAC'], [sac_ret.mean(), wm_ret.mean()],
           color=[C_SAC, C_WM])
    a2.set_ylabel('mean return')
    for i, v in enumerate([sac_ret.mean(), wm_ret.mean()]):
        a2.text(i, v, f'{v:.1f}', ha='center', va='bottom')
    fig.suptitle(f'Summary over {n} episodes')
    fig.tight_layout()
    fig.savefig(out_dir / 'summary.png', dpi=120); plt.close(fig)

    # 4. running success rate -- shows whether the gap is stable
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(eps, np.cumsum(sac_succ) / (eps + 1), color=C_SAC, label='Pure SAC')
    ax.plot(eps, np.cumsum(wm_succ) / (eps + 1), color=C_WM, label='WM + SAC')
    ax.set_xlabel('episode'); ax.set_ylabel('running success rate')
    ax.set_ylim(0, 1)
    ax.set_title('Cumulative success rate')
    ax.legend(); fig.tight_layout()
    fig.savefig(out_dir / 'running_success_rate.png', dpi=120); plt.close(fig)

    np.savez(out_dir / 'raw_results.npz',
             sac_returns=sac_ret, sac_successes=sac_succ, sac_steps_to_success=sac_sts,
             wm_returns=wm_ret, wm_successes=wm_succ, wm_steps_to_success=wm_sts)
    print(f'\nSaved 4 plots + raw_results.npz to {out_dir}/')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sac_ckpt', type=str,
                    default='sac_train_out/sac_final.pt')
    ap.add_argument('--joint_sac_ckpt', type=str,
                    default='joint_train_out/sac_final.pt')
    ap.add_argument('--joint_wm_ckpt', type=str,
                    default='joint_train_out/wm_latest.pkl')
    ap.add_argument('--episodes', type=int, default=500)
    ap.add_argument('--seed', type=int, default=12345)
    args = ap.parse_args()

    folder = pathlib.Path(__file__).parent
    config = load_config(folder, argv=[])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed_everywhere(args.seed)

    env, _, _ = OGBenchMethods.load_ogbench(config.joint.general.env_name)
    action_dim = env.action_space.shape[0]
    n = args.episodes

    print(f'\nLoading pure SAC: {args.sac_ckpt}')
    sac_policy = load_sac_only(args.sac_ckpt, env, config, device)
    print(f'=== Pure SAC, {n} episodes ===')
    sac = run_eval(env, sac_policy, n, 'sac_only',
                   bridge=None, action_dim=action_dim, seed=args.seed)

    print(f'\nLoading WM + SAC: {args.joint_sac_ckpt}')
    wm_policy, bridge = load_wm_sac(
        args.joint_sac_ckpt, args.joint_wm_ckpt, env, config, device)
    print(f'=== WM + SAC, {n} episodes ===')
    wm = run_eval(env, wm_policy, n, 'wm_sac',
                  bridge=bridge, action_dim=action_dim, seed=args.seed)

    sac_ret, sac_succ, _ = sac
    wm_ret, wm_succ, _ = wm

    print('\n' + '=' * 60)
    print(f'{"metric":24s} {"Pure SAC":>16s} {"WM + SAC":>16s}')
    print('=' * 60)
    print(f'{"success rate":24s} {sac_succ.mean():>16.3f} {wm_succ.mean():>16.3f}')
    print(f'{"mean return":24s} {sac_ret.mean():>16.2f} {wm_ret.mean():>16.2f}')
    print('=' * 60)

    plot_all(pathlib.Path('comparison'), sac, wm, n)
    env.close()


if __name__ == '__main__':
    main()