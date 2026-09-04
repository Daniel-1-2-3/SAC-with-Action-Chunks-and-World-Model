""" Run the OFFICIAL QC code (github.com/ColinQiyangLi/qc, vendored as the
    baselines/qc submodule, never edited) with this repo's flags, so a run
    of it is one command away from the matching run of our port.

      python train_official.py --baseline qc    --general.env_name=... --seed=0
      python train_official.py --baseline qcfql --general.env_name=... --seed=0

    --baseline qc      main.py --agent.actor_type=best-of-n
                       --agent.actor_num_samples=chunk.qc_num_samples
                       (the paper's QC; validates train_qc.py)
    --baseline qcfql   main.py --agent.alpha=chunk.alpha (actor_type
                       distill-ddpg, the paper's QC-FQL; the policy of
                       train_control.py --chunk.select_n=1)

    Every other flag is read from configs.yaml exactly as our train_*.py
    read it (--configs presets and dotted overrides included) and mapped to
    main.py's flags; see baselines/README.md for the table. The official
    code runs in its own interpreter (.venv-qc, JAX stack; see
    baselines/setup_qc_venv.sh) because its pins clash with ours.

    Output: <general.out_dir>/qc/<run_group>/<env>/sd<seed>_<time>/eval.csv,
    main.py's own layout. --dry_run prints the command and exits. """

import argparse
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
QC_DIR = ROOT / 'baselines' / 'qc'
VENV_PY = ROOT / '.venv-qc' / 'bin' / 'python'


def official_command(config, baseline, python):
    g, c = config.general, config.chunk
    hidden = '(' + ', '.join([str(int(c.hidden_dim))] * int(c.num_layers)) + ')'
    cmd = [
        str(python), 'main.py',
        f'--run_group={g.run_name or "reproduce"}',
        f'--seed={int(config.seed)}',
        f'--env_name={g.env_name}',
        f'--save_dir={pathlib.Path(g.out_dir).resolve()}',
        f'--offline_steps={int(g.num_offline_steps)}',
        f'--online_steps={int(g.num_online_steps)}',
        f'--log_interval={int(g.log_every)}',
        f'--eval_interval={int(g.eval_every)}',
        f'--start_training={int(g.start_training)}',
        f'--utd_ratio={int(c.utd_ratio)}',
        f'--discount={float(c.gamma)}',
        f'--eval_episodes={int(g.eval_episodes)}',
        f'--horizon_length={int(c.chunk_len)}',
        '--sparse=False',
        f'--agent.lr={float(c.lr)}',
        f'--agent.batch_size={int(c.batch_size)}',
        f'--agent.actor_hidden_dims={hidden}',
        f'--agent.value_hidden_dims={hidden}',
        f'--agent.tau={float(c.critic_target_tau)}',
        f'--agent.q_agg={c.q_agg}',
        f'--agent.num_qs={int(c.ensemble)}',
        f'--agent.flow_steps={int(c.flow_steps)}',
    ]
    if baseline == 'qc':
        cmd += ['--agent.actor_type=best-of-n',
                f'--agent.actor_num_samples={int(c.qc_num_samples)}']
    else:
        cmd += ['--agent.actor_type=distill-ddpg',
                f'--agent.alpha={float(c.alpha)}']
    return cmd


def main(argv=None):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--baseline', choices=['qc', 'qcfql'], required=True)
    ap.add_argument('--dry_run', action='store_true')
    ap.add_argument('--python', default=str(VENV_PY), help='interpreter with the official requirements')
    args, rest = ap.parse_known_args(argv)

    from sac_chunked.experiment import load_config
    config = load_config(ROOT, rest)
    cmd = official_command(config, args.baseline, args.python)
    env = dict(os.environ)
    env.setdefault('MUJOCO_GL', 'egl')
    if config.general.wandb_mode != 'online':
        env['WANDB_MODE'] = config.general.wandb_mode
    env['WANDB_PROJECT'] = config.general.wandb_project
    print('cwd:', QC_DIR)
    print('cmd:', ' '.join(cmd))
    if args.dry_run:
        return 0
    if not (QC_DIR / 'main.py').exists():
        raise SystemExit('baselines/qc is empty: git submodule update --init baselines/qc')
    if not pathlib.Path(args.python).exists():
        raise SystemExit(f'{args.python} not found: bash baselines/setup_qc_venv.sh')
    return subprocess.call(cmd, cwd=QC_DIR, env=env)


if __name__ == '__main__':
    sys.exit(main())
