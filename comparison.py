import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('MUJOCO_GL', 'egl')

import argparse
import pathlib
import elements
import jax
import numpy as np
import torch
import wandb

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
    returns, successes = [], []

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
                enc_carry, dyn_carry, feat = bridge.encode_step(
                    enc_carry, dyn_carry, state, prevact, is_first)
                feat_np = np.asarray(feat).reshape(-1)

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

        log = {f'{tag}/episode_return': ep_return}
        if hit_at is not None:
            log[f'{tag}/steps_to_success'] = hit_at
        wandb.log(log, step=ep)

        if (ep + 1) % 50 == 0:
            print(f'    {ep+1}/{n_episodes} | running success {np.mean(successes):.3f}')

    return np.array(returns), np.array(successes)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sac_ckpt', type=str,
                    default='joint_train_out_sac_only/sac_final.pt')
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

    wandb.init(project=config.joint.general.wandb_project,
               mode=config.joint.general.wandb_mode,
               name='comparison')

    env, _, _ = OGBenchMethods.load_ogbench(config.joint.general.env_name)
    action_dim = env.action_space.shape[0]
    n = args.episodes

    print(f'\nLoading pure SAC: {args.sac_ckpt}')
    sac_policy = load_sac_only(args.sac_ckpt, env, config, device)
    print(f'=== Pure SAC, {n} episodes ===')
    sac_ret, sac_succ = run_eval(env, sac_policy, n, 'sac_only',
                                 bridge=None, action_dim=action_dim, seed=args.seed)

    print(f'\nLoading WM + SAC: {args.joint_sac_ckpt}')
    wm_policy, bridge = load_wm_sac(
        args.joint_sac_ckpt, args.joint_wm_ckpt, env, config, device)
    print(f'=== WM + SAC, {n} episodes ===')
    wm_ret, wm_succ = run_eval(env, wm_policy, n, 'wm_sac',
                               bridge=bridge, action_dim=action_dim, seed=args.seed)

    print('\n' + '=' * 60)
    print(f'{"metric":24s} {"Pure SAC":>16s} {"WM + SAC":>16s}')
    print('=' * 60)
    print(f'{"success rate":24s} {sac_succ.mean():>16.3f} {wm_succ.mean():>16.3f}')
    print(f'{"mean return":24s} {sac_ret.mean():>16.2f} {wm_ret.mean():>16.2f}')
    print('=' * 60)

    wandb.log({
        'summary/sac_only_success_rate': float(sac_succ.mean()),
        'summary/sac_only_mean_return': float(sac_ret.mean()),
        'summary/wm_sac_success_rate': float(wm_succ.mean()),
        'summary/wm_sac_mean_return': float(wm_ret.mean()),
        'summary/episodes': n,
    })

    env.close()
    wandb.finish()


if __name__ == '__main__':
    main()