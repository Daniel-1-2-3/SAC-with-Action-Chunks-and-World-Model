""" Overlay our eval_log.csv curves on the QC paper's released curves.

    The paper's plot_data (github.com/ColinQiyangLi/qc, plot_data/README.md)
    is a pickle keyed by (task, method) with `steps` (millions, 0..2M every
    100k), `means` (success rate) and 95% bootstrap CI `ci_lows` /
    `ci_highs`. After Commit 5 it is at baselines/qc/plot_data/.

      python scripts/plot_vs_paper.py --run runs/t4_qc_s0 [--run ...] \\
          [--method QC --method QC-FQL] [--paper_pkl PATH] [--out PATH.png]

    The task is read from each run's eval_log.csv (its `env` column):
    'cube-triple-play-singletask-task4-v0' -> 'cube-triple-play-task4'.
    Prints, for every paper step that has one of our evals within 50k
    steps, our success rate next to the paper mean and CI, and whether the
    final point lands inside, above or below the paper's final CI. """

import argparse
import csv
import pathlib
import pickle
import sys

import numpy as np

DEFAULT_PKL = pathlib.Path(__file__).resolve().parent.parent / 'baselines' / 'qc' / 'plot_data' / 'ogbench-individual.pkl'


def paper_task(env_name):
    parts = [p for p in env_name.split('-') if p not in ('singletask', 'v0')]
    return '-'.join(parts)


def load_run(run_dir):
    p = pathlib.Path(run_dir) / 'eval_log.csv'
    rows = list(csv.DictReader(open(p)))
    if not rows:
        raise SystemExit(f'{p}: no eval rows')
    steps = np.array([float(r['env_step']) for r in rows]) / 1e6
    succ = np.array([float(r['success_rate']) for r in rows])
    return rows[0]['env'], rows[0]['arm'], steps, succ


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='append', required=True, help='run dir with eval_log.csv (repeatable)')
    ap.add_argument('--method', action='append', default=None, help='paper method(s) to draw; default QC and QC-FQL')
    ap.add_argument('--paper_pkl', default=str(DEFAULT_PKL))
    ap.add_argument('--out', default=None, help='png path; default <first run>/vs_paper.png')
    ap.add_argument('--tol', type=float, default=0.05, help='match tolerance in M steps')
    args = ap.parse_args(argv)
    methods = args.method or ['QC', 'QC-FQL']

    pkl = pathlib.Path(args.paper_pkl)
    if not pkl.exists():
        raise SystemExit(f'{pkl} not found. Clone github.com/ColinQiyangLi/qc (or init the '
                         f'baselines/qc submodule) and pass --paper_pkl .../plot_data/ogbench-individual.pkl')
    paper = pickle.load(open(pkl, 'rb'))

    runs = [load_run(r) for r in args.run]
    task = paper_task(runs[0][0])
    curves = {}
    for m in methods:
        if (task, m) not in paper:
            print(f'paper has no ({task}, {m}); available for this task: '
                  f'{sorted(k[1] for k in paper if k[0] == task)}')
            continue
        curves[m] = {k: np.asarray(v, dtype=float) for k, v in paper[(task, m)].items()}

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for m, c in curves.items():
        ax.plot(c['steps'], c['means'], label=f'paper {m}', lw=2)
        ax.fill_between(c['steps'], c['ci_lows'], c['ci_highs'], alpha=0.2)
    for (env, arm, steps, succ), run_dir in zip(runs, args.run):
        ax.plot(steps, succ, marker='o', ms=3, lw=1.2, label=f'ours {arm} ({pathlib.Path(run_dir).name})')
    ax.set_xlabel('env steps (M), offline then online')
    ax.set_ylabel('success rate')
    ax.set_title(task)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    out = pathlib.Path(args.out) if args.out else pathlib.Path(args.run[0]) / 'vs_paper.png'
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f'wrote {out}')

    ref = curves.get(methods[0])
    if ref is None:
        return
    for (env, arm, steps, succ), run_dir in zip(runs, args.run):
        print(f'\n{run_dir}  ({arm}, {env}) vs paper {methods[0]} on {task}')
        print(f'{"M steps":>8} {"ours":>7} {"paper":>7} {"ci_low":>7} {"ci_high":>7}')
        for ps, pm, lo, hi in zip(ref['steps'], ref['means'], ref['ci_lows'], ref['ci_highs']):
            j = int(np.argmin(np.abs(steps - ps)))
            if abs(steps[j] - ps) <= args.tol:
                print(f'{ps:>8.1f} {succ[j]:>7.3f} {pm:>7.3f} {lo:>7.3f} {hi:>7.3f}')
        j = int(np.argmin(np.abs(ref['steps'] - steps[-1])))
        lo, hi, pm = ref['ci_lows'][j], ref['ci_highs'][j], ref['means'][j]
        where = 'inside' if lo <= succ[-1] <= hi else ('above' if succ[-1] > hi else 'below')
        print(f'final: ours {succ[-1]:.3f} at {steps[-1]:.2f}M vs paper {pm:.3f} '
              f'[{lo:.3f}, {hi:.3f}] at {ref["steps"][j]:.1f}M -> {where} the CI')


if __name__ == '__main__':
    sys.exit(main())
