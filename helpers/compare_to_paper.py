"""Reads eval_log.csv from one or more runs and prints an informal comparison.

    python helpers/compare_to_paper.py train_sac_chunked_wm_out/eval_log.csv train_sac_chunked_out/eval_log.csv

Rows from the same file are grouped by (arm, env, seed). Sample efficiency is
reported as environment steps to first cross a success threshold, which is the
comparison that matters here -- all three of the world model's structural
advantages are sample-efficiency arguments, and none of them predicts a higher
final ceiling.

The paper reference numbers below are read off Figure 2 of "Reinforcement
Learning with Action Chunking" by eye. They are approximate, and the setups do
not line up: the paper pretrains offline for 1M steps and then runs 1M online
steps, while these runs seed the buffer from the offline data and go straight
online. Treat the comparison as a sanity check on the order of magnitude, not
as a benchmark result.
"""

import collections
import csv
import sys

PAPER_REFERENCE = {
    'cube-double': {
        'source': 'QC paper, Figure 2 (approximate, read from the plot)',
        'note': '1M offline pretrain + 1M online steps, 5 seeds x 5 tasks',
        'methods': {
            'QC (chunked, best-of-N)': 0.90,
            'QC-FQL (chunked, distilled)': 0.88,
            'FQL (no chunking)': 0.82,
            'BFN (no chunking)': 0.85,
            'RLPD (no offline pretrain)': 0.75,
        },
    },
    'cube-triple': {
        'source': 'QC paper, Figure 2 (approximate, read from the plot)',
        'note': 'same setup; near-zero success for every method during offline phase',
        'methods': {
            'QC (chunked, best-of-N)': 0.60,
            'FQL (no chunking)': 0.25,
            'RLPD (no offline pretrain)': 0.35,
        },
    },
}

THRESHOLDS = (0.25, 0.50, 0.75)

def load(paths):
    runs = collections.defaultdict(list)
    for path in paths:
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                key = (row['arm'], row['env'], row['seed'])
                runs[key].append({
                    'env_step': int(row['env_step']),
                    'updates': int(row['gradient_updates']),
                    'ret': float(row['mean_return']),
                    'succ': float(row['success_rate']),
                    'coh': float(row['coherence']),
                    'wall': float(row['wall_time_s']),
                })
    for rows in runs.values():
        rows.sort(key=lambda r: r['env_step'])
    return runs

def steps_to(rows, threshold):
    for r in rows:
        if r['succ'] >= threshold:
            return r['env_step']
    return None

def fmt(value):
    return '--' if value is None else f'{value:,}'

def main(paths):
    runs = load(paths)
    if not runs:
        print('No rows found.')
        return

    by_arm = collections.defaultdict(list)
    for (arm, env, seed), rows in sorted(runs.items()):
        by_arm[(arm, env)].append((seed, rows))

    print()
    print('PER-SEED')
    header = f'{"arm":<16}{"seed":>6}{"final succ":>12}{"best succ":>11}{"final ret":>11}{"coherence":>11}'
    header += ''.join(f'{f"->{int(t*100)}%":>10}' for t in THRESHOLDS)
    print(header)
    print('-' * len(header))
    for (arm, env), entries in sorted(by_arm.items()):
        for seed, rows in entries:
            line = f'{arm:<16}{seed:>6}{rows[-1]["succ"]:>12.2f}'
            line += f'{max(r["succ"] for r in rows):>11.2f}'
            line += f'{rows[-1]["ret"]:>11.1f}{rows[-1]["coh"]:>11.4f}'
            line += ''.join(f'{fmt(steps_to(rows, t)):>10}' for t in THRESHOLDS)
            print(line)

    print()
    print('MEAN OVER SEEDS')
    for (arm, env), entries in sorted(by_arm.items()):
        n = len(entries)
        final = sum(rows[-1]['succ'] for _, rows in entries) / n
        best = sum(max(r['succ'] for r in rows) for _, rows in entries) / n
        coh = sum(rows[-1]['coh'] for _, rows in entries) / n
        wall = sum(rows[-1]['wall'] for _, rows in entries) / n
        print(f'  {arm}  ({env}, {n} seed{"s" if n != 1 else ""})')
        print(f'    final success   {final:.3f}')
        print(f'    best success    {best:.3f}')
        print(f'    coherence       {coh:.4f}')
        print(f'    wall clock      {wall / 3600:.1f} h')
        for t in THRESHOLDS:
            hit = [steps_to(rows, t) for _, rows in entries]
            reached = [h for h in hit if h is not None]
            if reached:
                print(f'    steps to {int(t*100):>2}%    {sum(reached)//len(reached):,} '
                      f'({len(reached)}/{n} seeds)')
            else:
                print(f'    steps to {int(t*100):>2}%    not reached')
        if n < 5:
            print(f'    WARNING: {n} seed(s). Seed spread on this task is wide enough '
                  f'that a gap under ~15 points is not distinguishable from noise.')

    envs = {env for _, env in by_arm}
    for env in sorted(envs):
        for key, ref in PAPER_REFERENCE.items():
            if key in env:
                print()
                print(f'PAPER REFERENCE for {key}')
                print(f'  {ref["source"]}')
                print(f'  {ref["note"]}')
                for method, value in ref['methods'].items():
                    print(f'    {method:<32}{value:.2f}')
                print('  Not directly comparable -- different training budget and '
                      'offline/online split. Order of magnitude only.')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
