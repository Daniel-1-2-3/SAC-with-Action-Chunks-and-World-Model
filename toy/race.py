""" Race the arms against each other on the toy pusher.

      python toy/race.py --arms critic ranking mve explore optimistic --seeds 0 1

    Runs every (arm, seed) with `--configs toy` (plus any extra args after
    `--`), a few at a time, then prints a table of eval return / success per
    arm from the eval_log.csv files. Minutes on a CPU. This is a sanity race,
    not a benchmark: the toy task is easy enough that arms can tie at the
    ceiling. """

import argparse
import csv
import pathlib
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS = {
    'critic': 'train_sac_chunked.py',
    'qcfql': 'train_sac_chunked.py',
    'ranking': 'train_sac_chunked_ranking.py',
    'mve': 'train_sac_chunked_mve.py',
    'explore': 'train_sac_chunked_explore.py',
    'optimistic': 'train_sac_chunked_optimistic.py',
}


def run_one(arm, seed, out_root, extra, threads):
    out = out_root / f'{arm}_s{seed}'
    cmd = [sys.executable, str(ROOT / SCRIPTS[arm]), '--configs', 'toy',
           f'--seed={seed}', f'--general.out_dir={out}']
    if arm == 'qcfql':
        cmd.append('--chunk.select_n=1')
    cmd += extra
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'stdout.txt', 'w') as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT,
                             env={**__import__('os').environ,
                                  'OMP_NUM_THREADS': str(threads),
                                  'MKL_NUM_THREADS': str(threads)})
    return arm, seed, rc


def summarize(out_root, arms, seeds):
    rows = defaultdict(lambda: defaultdict(list))   # arm -> step -> [(ret, succ)]
    for arm in arms:
        for seed in seeds:
            p = out_root / f'{arm}_s{seed}' / 'eval_log.csv'
            if not p.exists():
                continue
            with open(p) as f:
                for r in csv.DictReader(f):
                    rows[arm][int(r['env_step'])].append(
                        (float(r['mean_return']), float(r['success_rate'])))
    steps = sorted({s for a in rows.values() for s in a})
    print('\n=== eval mean_return (success_rate), mean over seeds ===')
    print(f'{"step":>8} | ' + ' | '.join(f'{a:>22}' for a in arms))
    for s in steps:
        cells = []
        for a in arms:
            v = rows[a].get(s)
            if not v:
                cells.append(f'{"-":>22}')
            else:
                ret = sum(x[0] for x in v) / len(v)
                suc = sum(x[1] for x in v) / len(v)
                cells.append(f'{ret:>10.1f} ({suc:>4.2f}) n={len(v)}')
        print(f'{s:>8} | ' + ' | '.join(cells))
    print('\nfinal-third average (later evals, mean over seeds):')
    for a in arms:
        if not rows[a]:
            continue
        ss = sorted(rows[a])
        tail = ss[len(ss) * 2 // 3:]
        vals = [x for s in tail for x in rows[a][s]]
        ret = sum(x[0] for x in vals) / len(vals)
        suc = sum(x[1] for x in vals) / len(vals)
        print(f'  {a:>12}: return {ret:8.1f}   success {suc:.2f}   ({len(tail)} evals x {len(seeds)} seeds)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', nargs='+', default=['critic', 'ranking', 'mve', 'explore', 'optimistic'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1])
    ap.add_argument('--out', default='runs/toy_race')
    ap.add_argument('--parallel', type=int, default=2)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--summary-only', action='store_true')
    args, extra = ap.parse_known_args()
    extra = [e for e in extra if e != '--']
    out_root = ROOT / args.out
    if not args.summary_only:
        jobs = [(a, s) for s in args.seeds for a in args.arms]
        with ThreadPoolExecutor(args.parallel) as ex:
            for arm, seed, rc in ex.map(lambda j: run_one(*j, out_root, extra, args.threads), jobs):
                print(f'[race] {arm} seed {seed} -> exit {rc}', flush=True)
    summarize(out_root, args.arms, args.seeds)
