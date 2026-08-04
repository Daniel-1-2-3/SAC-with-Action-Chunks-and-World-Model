"""
check_rollout_animation.py
--------------------------
Proprioceptive equivalent of the decode_and_save / evaluate pattern from models.py.

In models.py:
    z = encoder(s)
    z_eval = latent_imagination(..., forced=True)   # teacher-forced with real actions
    decode_and_save(z_eval, "single_frame")         # -> frames that should animate
    z_eval = latent_imagination(..., random=True)   # free imagination, random actions
    decode_and_save(z_eval, 'random')               # -> frames that show dynamics

We do the same two rollouts, but instead of decoding to pixels and saving PNGs,
we decode to the 28-dim proprioceptive state and plot each joint angle over the
horizon as a line. If the world model has learned dynamics:
  - teacher-forced:  decoded state should closely track the REAL next observations
  - random actions:  decoded state should visibly diverge from the seed over time

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
import ogbench

from dreamer.wm_agent import WorldModelAgent
from dreamer.wm_bridge import WorldModelBridge
from interop import unwrap
from ogbench_methods import OGBenchMethods

OBS_KEY = 'state'
ACTION_KEY = 'action'


def load_config(folder):
    configs_txt = elements.Path(folder / 'configs.yaml').read()
    configs = yaml.YAML(typ='safe').load(configs_txt)
    # parse_known returns (parsed, leftover). Pass an empty list so elements
    # never sees --wm_ckpt / --horizon / etc. and tries to match them against
    # config keys (which raises the KeyError seen at startup).
    parsed, _ = elements.Flags(configs=['defaults']).parse_known([])
    config = elements.Config(configs['defaults'])
    for name in parsed.configs:
        config = config.update(configs[name])
    # Don't call elements.Flags(config).parse() at all -- this script has its
    # own argparse and doesn't accept config overrides on the command line.
    return config


def rollout_teacher_forced(bridge, enc_carry, dyn_carry, real_actions, action_dim):
    """
    Mirrors models.py latent_imagination(..., forced=True, random=False).
    Steps the world model forward using REAL actions from the episode.
    Returns list of decoded state arrays, one per step.
    """
    decoded_states = []
    carry = dyn_carry
    prevact = np.zeros((1, action_dim), dtype=np.float32)
    for t, action_np in enumerate(real_actions):
        action_np = action_np.reshape(1, -1).astype(np.float32)
        next_carry, _, _, _ = bridge.img_step(carry, action_np)
        decoded = bridge.decode_state(next_carry)
        decoded_states.append(decoded[OBS_KEY].flatten().copy())
        carry = next_carry
    return decoded_states


def rollout_random_actions(bridge, dyn_carry, action_dim, horizon, rng):
    """
    Mirrors models.py latent_imagination(..., random=True).
    Steps the world model forward with random actions.
    Returns list of decoded state arrays, one per step.
    """
    decoded_states = []
    carry = dyn_carry
    for _ in range(horizon):
        action_np = rng.uniform(-1, 1, (1, action_dim)).astype(np.float32)
        next_carry, _, _, _ = bridge.img_step(carry, action_np)
        decoded = bridge.decode_state(next_carry)
        decoded_states.append(decoded[OBS_KEY].flatten().copy())
        carry = next_carry
    return decoded_states


def plot_rollout(decoded_states, real_states, title, out_path, obs_dim,
                 n_dims_shown=8):
    """
    Like decode_and_save in models.py but for proprioceptive state.
    Plots each joint dimension over the rollout horizon as a line.
    Real states are shown as dashed lines for comparison.
    """
    steps = len(decoded_states)
    decoded_arr = np.stack(decoded_states)          # (steps, obs_dim)
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
            ax.plot(real_arr[:, dim], color='tomato', linestyle='--',
                    label='real')
        ax.set_title(f'dim {dim}', fontsize=9)
        ax.set_xlabel('step')
        if dim == 0:
            ax.legend(fontsize=7)

        # L2 change per step -- if this is near zero the world model is static
        diffs = np.linalg.norm(np.diff(decoded_arr, axis=0), axis=1)
        ax.set_ylabel(f'L2/step={diffs.mean():.4f}', fontsize=7)

    for i in range(n_show, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


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
    parser.add_argument('--n_dims_shown', type=int, default=8,
                        help='How many state dimensions to plot')
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

    batch_size = config.batch_size
    seq_len = config.batch_length
    agent_config = elements.Config(
        **config.agent,
        logdir=str(folder / 'wm_ckpts'),
        seed=config.seed,
        jax=config.jax,
        batch_size=batch_size,
        batch_length=seq_len,
        replay_context=0,
        report_length=seq_len,
        replica=0,
        replicas=1,
    )

    print('Building world model agent...')
    wm_agent = WorldModelAgent(obs_space, act_space, agent_config)

    print(f'Loading checkpoint: {args.wm_ckpt}')
    raw = np.load(args.wm_ckpt, allow_pickle=True)
    state = {k: unwrap(raw[k]) for k in raw.files}
    wm_agent.load(state)
    bridge = WorldModelBridge(wm_agent, ACTION_KEY, obs_key=OBS_KEY)

    rng = np.random.default_rng(0)
    offline_eps = OGBenchMethods.make_dreamer_episodes(
        train_dataset, min_length=seq_len + args.horizon,
        obs_key=OBS_KEY, action_key=ACTION_KEY)
    print(f'Loaded {len(offline_eps)} offline episodes (min_length={seq_len+args.horizon})')

    print(f'\n{"="*70}')
    print(f'Rollout animation check | horizon={args.horizon} | n_seeds={args.n_seeds}')
    print(f'{"="*70}')

    for seed_idx in range(args.n_seeds):
        ep = offline_eps[rng.integers(0, len(offline_eps))]
        ep_len = len(ep[OBS_KEY])
        # Pick a start that leaves enough room for the full horizon
        start = rng.integers(0, ep_len - args.horizon - 1)
        real_state = ep[OBS_KEY][start].astype(np.float32).reshape(1, -1)

        # Real observations and actions starting from `start`
        real_obs_seq = ep[OBS_KEY][start : start + args.horizon + 1]  # (H+1, obs_dim)
        real_act_seq = ep[ACTION_KEY][start : start + args.horizon]    # (H, action_dim)

        print(f'\nSeed {seed_idx+1} | episode step {start}')
        print(f'  Real state[:5]: {real_state[0, :5].round(3)}')

        # Encode the seed state into a posterior latent -- mirrors:
        #   z = self.encoder(s)  from models.py evaluate()
        enc_carry, dyn_carry = bridge.init_encode(1)
        enc_carry, dyn_carry, _ = bridge.encode_step(
            enc_carry, dyn_carry, real_state,
            np.zeros((1, action_dim), np.float32),
            np.array([True]))

        # Decode the seed latent to verify encoding works at all
        seed_decoded = bridge.decode_state(dyn_carry)[OBS_KEY].flatten()
        seed_l2 = float(np.linalg.norm(seed_decoded - real_state[0]))
        print(f'  Seed reconstruction L2={seed_l2:.4f}  '
              f'(should be small if encoder is trained)')

        # ---- Rollout 1: teacher-forced with real actions ----
        # Mirrors: z_eval = latent_imagination(..., forced=True)[0]
        #          decode_and_save(z_eval, "single_frame")
        print('  Teacher-forced rollout (real actions):')
        tf_decoded = rollout_teacher_forced(
            bridge, enc_carry, dyn_carry,
            real_act_seq, action_dim)
        tf_real_next = [real_obs_seq[t + 1] for t in range(len(tf_decoded))]
        tf_l2 = compute_l2_stats(tf_decoded, 'teacher-forced')

        # How well does teacher-forced match real next observations?
        tf_pred_err = np.mean([
            np.linalg.norm(tf_decoded[t] - tf_real_next[t])
            for t in range(len(tf_decoded))])
        print(f'  Teacher-forced prediction error vs real: {tf_pred_err:.4f}  '
              f'(lower = world model learned real transitions)')

        plot_rollout(
            tf_decoded, tf_real_next,
            title=(f'Seed {seed_idx+1} | Teacher-forced | '
                   f'avg L2/step={tf_l2:.4f} | pred_err={tf_pred_err:.4f}'),
            out_path=out_dir / f'seed{seed_idx+1}_teacher_forced.png',
            obs_dim=obs_dim,
            n_dims_shown=args.n_dims_shown)

        # ---- Rollout 2: random actions ----
        # Mirrors: z_eval = latent_imagination(..., random=True)[0]
        #          decode_and_save(z_eval, 'random')
        print('  Random-action rollout:')
        rand_decoded = rollout_random_actions(
            bridge, dyn_carry, action_dim, args.horizon, rng)
        rand_l2 = compute_l2_stats(rand_decoded, 'random')

        plot_rollout(
            rand_decoded, real_states=None,
            title=(f'Seed {seed_idx+1} | Random actions | '
                   f'avg L2/step={rand_l2:.4f}'),
            out_path=out_dir / f'seed{seed_idx+1}_random_actions.png',
            obs_dim=obs_dim,
            n_dims_shown=args.n_dims_shown)

        # ---- Summary for this seed ----
        print(f'  Summary: teacher_forced_L2={tf_l2:.4f}  '
              f'random_L2={rand_l2:.4f}')
        if tf_l2 < 0.01 and rand_l2 < 0.01:
            print('  *** COMPLETELY STATIC: world model not learning dynamics ***')
        elif rand_l2 < tf_l2 * 0.5:
            print('  *** SUSPICIOUS: random actions move the latent LESS than '
                  'real actions -- world model may be ignoring action input ***')
        elif tf_pred_err < 0.5 and tf_l2 > 0.02:
            print('  OK: world model is both dynamic and tracks real transitions')

    print(f'\nAll plots saved to {out_dir}/')
    env.close()


if __name__ == '__main__':
    main()