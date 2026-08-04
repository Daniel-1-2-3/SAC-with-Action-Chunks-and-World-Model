"""
check_rollout_animation.py
--------------------------
Proprioceptive equivalent of the decode_and_save / evaluate pattern from models.py.

Runs two rollouts from a seed state:
  1. Teacher-forced: world model steps forward with REAL actions
  2. Random actions: world model steps forward with random actions

For each rollout, saves:
  - A PNG plot of each state dimension over the horizon
  - An MP4 video of the robot arm moving (decoded states replayed into MuJoCo)

Usage:
    python check_rollout_animation.py \
        --wm_ckpt joint_train_out/wm_latest.pkl \
        --horizon 30 \
        --n_seeds 3 \
        --out_dir rollout_check
"""

import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('MUJOCO_GL', 'egl')

import argparse
import pathlib

import elements
import jax
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ruamel.yaml as yaml
import cv2

from dreamer.wm_agent import WorldModelAgent
from dreamer.wm_bridge import WorldModelBridge
from interop import unwrap
from ogbench_methods import OGBenchMethods

OBS_KEY = 'state'
ACTION_KEY = 'action'

# cube-single-play-v0 state layout: qpos=15 dims, qvel=13 dims = 28 total
QPOS_DIM = 15
QVEL_DIM = 13


def load_config(folder):
    configs_txt = elements.Path(folder / 'configs.yaml').read()
    configs = yaml.YAML(typ='safe').load(configs_txt)
    parsed, _ = elements.Flags(configs=['defaults']).parse_known([])
    config = elements.Config(configs['defaults'])
    for name in parsed.configs:
        config = config.update(configs[name])
    return config


def rollout_teacher_forced(bridge, dyn_carry, real_actions):
    decoded_states = []
    carry = dyn_carry
    for action_np in real_actions:
        action_np = action_np.reshape(1, -1).astype(np.float32)
        next_carry, _, _, _ = bridge.img_step(carry, action_np)
        decoded = bridge.decode_state(next_carry)
        decoded_states.append(decoded[OBS_KEY].flatten().copy())
        carry = next_carry
    return decoded_states


def rollout_random_actions(bridge, dyn_carry, action_dim, horizon, rng):
    decoded_states = []
    carry = dyn_carry
    for _ in range(horizon):
        action_np = rng.uniform(-1, 1, (1, action_dim)).astype(np.float32)
        next_carry, _, _, _ = bridge.img_step(carry, action_np)
        decoded = bridge.decode_state(next_carry)
        decoded_states.append(decoded[OBS_KEY].flatten().copy())
        carry = next_carry
    return decoded_states


def render_video(env, states, out_path, fps=10, width=256, height=256):
    """
    Replay decoded states into MuJoCo and render each frame to an MP4.
    states: list of 28-dim numpy arrays (qpos[15] + qvel[13])
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    u = env.unwrapped

    for state in states:
        qpos = state[:QPOS_DIM]
        qvel = state[QPOS_DIM:QPOS_DIM + QVEL_DIM]
        try:
            # set_state needs exact nq/nv sizes -- pad or trim if needed
            nq = u.model.nq
            nv = u.model.nv
            qpos_in = np.zeros(nq, dtype=np.float64)
            qvel_in = np.zeros(nv, dtype=np.float64)
            qpos_in[:min(len(qpos), nq)] = qpos[:min(len(qpos), nq)]
            qvel_in[:min(len(qvel), nv)] = qvel[:min(len(qvel), nv)]
            u.set_state(qpos_in, qvel_in)

            # Try multiple render paths -- OGBench envs vary
            frame = None
            if hasattr(u, 'mujoco_renderer'):
                frame = u.mujoco_renderer.render('rgb_array')
            elif hasattr(u, 'render'):
                frame = u.render()
            if frame is None:
                frame = np.zeros((height, width, 3), dtype=np.uint8)

            frame = cv2.resize(np.asarray(frame, dtype=np.uint8), (width, height))
            writer.write(frame[..., ::-1])  # RGB -> BGR
        except Exception as e:
            print(f'    render warning: {e}')
            writer.write(np.zeros((height, width, 3), dtype=np.uint8))

    writer.release()
    print(f'  Video saved: {out_path}')


def plot_rollout(decoded_states, real_states, title, out_path, obs_dim,
                 n_dims_shown=8):
    decoded_arr = np.stack(decoded_states)
    real_arr = np.stack(real_states) if real_states is not None else None

    n_show = min(n_dims_shown, obs_dim)
    ncols = 4
    nrows = (n_show + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = axes.flatten()
    fig.suptitle(title, fontsize=12, y=1.01)

    for dim in range(n_show):
        ax = axes[dim]
        ax.plot(decoded_arr[:, dim], color='steelblue', label='decoded')
        if real_arr is not None:
            ax.plot(real_arr[:, dim], color='tomato', linestyle='--', label='real')
        ax.set_title(f'dim {dim}', fontsize=9)
        ax.set_xlabel('step')
        ax.set_ylabel('value', fontsize=7)
        if dim == 0:
            ax.legend(fontsize=7)

    for i in range(n_show, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot saved: {out_path}')


def compute_l2_stats(decoded_states, label):
    arr = np.stack(decoded_states)
    diffs = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    avg = diffs.mean()
    mx = diffs.max()
    print(f'  [{label}] avg L2/step={avg:.4f}  max={mx:.4f}', end='  ')
    if avg < 0.01:
        print('*** STATIC ***')
    elif avg < 0.05:
        print('*** WEAK ***')
    else:
        print('OK')
    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wm_ckpt', type=str, required=True)
    parser.add_argument('--horizon', type=int, default=30)
    parser.add_argument('--n_seeds', type=int, default=3)
    parser.add_argument('--n_dims_shown', type=int, default=8)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--video_w', type=int, default=256)
    parser.add_argument('--video_h', type=int, default=256)
    parser.add_argument('--out_dir', type=str, default='rollout_check')
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    folder = pathlib.Path(__file__).parent
    config = load_config(folder)

    print('Loading environment + offline dataset...')
    env, train_dataset, _ = OGBenchMethods.load_ogbench(
        config.joint.general.env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    obs_space, act_space = OGBenchMethods.make_spaces(
        obs_dim, action_dim, OBS_KEY, ACTION_KEY)

    seq_len = config.batch_length
    agent_config = elements.Config(
        **config.agent,
        logdir=str(folder / 'wm_ckpts'),
        seed=config.seed,
        jax=config.jax,
        batch_size=1,
        batch_length=seq_len,
        replay_context=0,
        report_length=seq_len,
        replica=0,
        replicas=1,
    )

    print('Building world model agent...')
    wm_agent = WorldModelAgent(obs_space, act_space, agent_config)

    print(f'Loading checkpoint: {args.wm_ckpt}')
    wm_cp = elements.Checkpoint(pathlib.Path(args.wm_ckpt))
    wm_cp.agent = wm_agent
    wm_cp.load()
    bridge = WorldModelBridge(wm_agent, ACTION_KEY, obs_key=OBS_KEY)

    rng = np.random.default_rng(0)
    offline_eps = OGBenchMethods.make_dreamer_episodes(
        train_dataset, min_length=seq_len + args.horizon,
        obs_key=OBS_KEY, action_key=ACTION_KEY)
    print(f'Loaded {len(offline_eps)} offline episodes')

    print(f'\n{"="*70}')
    print(f'Rollout animation check | horizon={args.horizon} | n_seeds={args.n_seeds}')
    print(f'{"="*70}')

    for seed_idx in range(args.n_seeds):
        ep = offline_eps[rng.integers(0, len(offline_eps))]
        ep_len = len(ep[OBS_KEY])
        start = rng.integers(0, ep_len - args.horizon - 1)
        real_state = ep[OBS_KEY][start].astype(np.float32).reshape(1, -1)
        real_obs_seq = ep[OBS_KEY][start : start + args.horizon + 1]
        real_act_seq = ep[ACTION_KEY][start : start + args.horizon]

        print(f'\nSeed {seed_idx+1} | episode step {start}')

        # Encode seed state
        enc_carry, dyn_carry = bridge.init_encode(1)
        enc_carry, dyn_carry, _ = bridge.encode_step(
            enc_carry, dyn_carry, real_state,
            np.zeros((1, action_dim), np.float32),
            np.array([True]))

        seed_decoded = bridge.decode_state(dyn_carry)[OBS_KEY].flatten()
        seed_l2 = float(np.linalg.norm(seed_decoded - real_state[0]))
        print(f'  Seed reconstruction L2={seed_l2:.4f}')

        # ---- Teacher-forced rollout ----
        print('  Teacher-forced rollout (real actions):')
        tf_decoded = rollout_teacher_forced(bridge, dyn_carry, real_act_seq)
        tf_real_next = [real_obs_seq[t + 1] for t in range(len(tf_decoded))]
        tf_l2 = compute_l2_stats(tf_decoded, 'teacher-forced')
        tf_pred_err = np.mean([np.linalg.norm(tf_decoded[t] - tf_real_next[t])
                               for t in range(len(tf_decoded))])
        print(f'  Prediction error vs real: {tf_pred_err:.4f}')

        plot_rollout(
            tf_decoded, tf_real_next,
            title=f'Seed {seed_idx+1} | Teacher-forced | avg L2/step={tf_l2:.4f} | pred_err={tf_pred_err:.4f}',
            out_path=out_dir / f'seed{seed_idx+1}_teacher_forced.png',
            obs_dim=obs_dim, n_dims_shown=args.n_dims_shown)

        # render decoded states as video
        render_video(env, tf_decoded,
                     out_path=out_dir / f'seed{seed_idx+1}_teacher_forced.mp4',
                     fps=args.fps, width=args.video_w, height=args.video_h)
        # render real states as video for comparison
        render_video(env, list(real_obs_seq[1:]),
                     out_path=out_dir / f'seed{seed_idx+1}_real.mp4',
                     fps=args.fps, width=args.video_w, height=args.video_h)

        # ---- Random-action rollout ----
        print('  Random-action rollout:')
        rand_decoded = rollout_random_actions(
            bridge, dyn_carry, action_dim, args.horizon, rng)
        rand_l2 = compute_l2_stats(rand_decoded, 'random')

        plot_rollout(
            rand_decoded, real_states=None,
            title=f'Seed {seed_idx+1} | Random actions | avg L2/step={rand_l2:.4f}',
            out_path=out_dir / f'seed{seed_idx+1}_random_actions.png',
            obs_dim=obs_dim, n_dims_shown=args.n_dims_shown)

        render_video(env, rand_decoded,
                     out_path=out_dir / f'seed{seed_idx+1}_random_actions.mp4',
                     fps=args.fps, width=args.video_w, height=args.video_h)

        print(f'  Summary: teacher_forced_L2={tf_l2:.4f}  random_L2={rand_l2:.4f}')
        if tf_l2 < 0.01 and rand_l2 < 0.01:
            print('  *** COMPLETELY STATIC ***')
        elif rand_l2 < tf_l2 * 0.5:
            print('  *** SUSPICIOUS: random actions move less than real actions ***')
        elif tf_pred_err < 0.5 and tf_l2 > 0.02:
            print('  OK: dynamic and tracking real transitions')

    print(f'\nAll outputs saved to {out_dir}/')
    env.close()


if __name__ == '__main__':
    main()