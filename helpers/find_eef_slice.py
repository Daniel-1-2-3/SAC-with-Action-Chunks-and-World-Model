"""Finds which observation dimensions hold end-effector xyz.

    python helpers/find_eef_slice.py
    python helpers/find_eef_slice.py --env cube-triple-play-singletask-v0

Commands the arm along one axis at a time and reports which dims move with it.
A dim that tracks +x and ignores +y and +z is effector x.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import ogbench

def probe(env, axis, action_dim, steps=15, magnitude=0.6, seed=0):
    obs, _ = env.reset(seed=seed)
    traj = [np.asarray(obs, dtype=np.float32)]
    action = np.zeros(action_dim, dtype=np.float32)
    action[axis] = magnitude
    for _ in range(steps):
        obs, _, term, trunc, _ = env.step(action)
        traj.append(np.asarray(obs, dtype=np.float32))
        if term or trunc:
            break
    return np.stack(traj)[-1] - np.stack(traj)[0]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--env', default='cube-double-play-singletask-v0')
    p.add_argument('--top', type=int, default=6)
    args = p.parse_args()

    env = ogbench.make_env_and_datasets(args.env, env_only=True)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f'{args.env}: obs_dim={obs_dim} action_dim={action_dim}\n')

    moves = {name: probe(env, i, action_dim)
             for i, name in enumerate(['+x', '+y', '+z'])}

    print(f'{"dim":>5}{"+x":>10}{"+y":>10}{"+z":>10}   verdict')
    print('-' * 52)
    picks = {}
    for d in range(obs_dim):
        row = [moves[k][d] for k in ('+x', '+y', '+z')]
        best = int(np.argmax(np.abs(row)))
        strength = abs(row[best])
        others = max(abs(row[i]) for i in range(3) if i != best)
        # selective = moves a lot on one axis, barely on the others
        selective = strength > 0.05 and strength > 3 * max(others, 1e-6)
        if strength < 0.02:
            continue
        tag = ''
        if selective:
            axis = 'xyz'[best]
            tag = f'<-- effector {axis}?'
            picks.setdefault(axis, (d, strength))
            if strength > picks[axis][1]:
                picks[axis] = (d, strength)
        print(f'{d:>5}{row[0]:>10.4f}{row[1]:>10.4f}{row[2]:>10.4f}   {tag}')

    print()
    if len(picks) == 3:
        dims = sorted(picks[a][0] for a in 'xyz')
        print(f'best guess: x={picks["x"][0]}  y={picks["y"][0]}  z={picks["z"][0]}')
        if dims == list(range(dims[0], dims[0] + 3)):
            print(f'\ncontiguous -> set in BOTH chunk config sections:')
            print(f'    eef_slice: [{dims[0]}, {dims[0] + 3}]')
        else:
            print(f'\nNOT contiguous ({dims}). eef_slice assumes a contiguous '
                  f'range, so temporal_coherence needs indexing by these dims '
                  f'instead.')
    else:
        print(f'only found {len(picks)} of 3 axes cleanly. Raise --magnitude or '
              f'inspect the table above by hand -- cube position dims move too '
              f'when the arm pushes a cube, which can mask the selectivity test.')

if __name__ == '__main__':
    main()