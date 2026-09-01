""" The paper's own sanity check (Sec 3.1, Fig 1): bit-flipping.
    DQN alone fails past ~13 bits; DQN+HER solves up to 50.

    This imports Replay and Normalizer FROM train_her.py, so it tests the
    exact relabeling code the cube run uses. Only the learner is swapped
    (DQN, since bit-flipping is discrete).
"""
import numpy as np
import torch
import torch.nn as nn

from train_her import Replay


class BitGoals:
    def __init__(self, n):
        self.n = n
        self.goal_dim = n

    def reward(self, achieved, goal):
        a = np.asarray(achieved)
        g = np.asarray(goal)
        return -(np.abs(a - g).sum(-1) > 0).astype(np.float32)


def run(n_bits, use_her, epochs, seed=0):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    gt = BitGoals(n_bits)
    q = nn.Sequential(nn.Linear(2 * n_bits, 256), nn.ReLU(),
                      nn.Linear(256, n_bits))
    q_t = nn.Sequential(nn.Linear(2 * n_bits, 256), nn.ReLU(),
                        nn.Linear(256, n_bits))
    q_t.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=1e-3)
    replay = Replay(int(1e6))
    future_p = 4 / 5.0 if use_her else 0.0
    gamma = 0.98

    for epoch in range(epochs):
        for cycle in range(50):
            for _ in range(16):
                state = rng.integers(0, 2, n_bits).astype(np.float32)
                goal = rng.integers(0, 2, n_bits).astype(np.float32)
                while np.array_equal(state, goal):
                    goal = rng.integers(0, 2, n_bits).astype(np.float32)
                ep = {'obs': [state.copy()], 'ach': [state.copy()],
                      'act': [], 'goal': goal}
                for _ in range(n_bits):
                    if rng.random() < 0.2:
                        a = int(rng.integers(n_bits))
                    else:
                        with torch.no_grad():
                            x = torch.as_tensor(
                                np.concatenate([state, goal])[None],
                                dtype=torch.float32)
                            a = int(q(x).argmax())
                    state = state.copy()
                    state[a] = 1.0 - state[a]
                    ep['obs'].append(state.copy())
                    ep['ach'].append(state.copy())
                    ep['act'].append(np.float32(a))
                for k in ('obs', 'ach', 'act'):
                    ep[k] = np.array(ep[k], np.float32)
                replay.add(ep)

            for _ in range(40):
                o, g, a, r, o2, g2 = replay.sample(128, future_p, gt, rng)
                x = torch.as_tensor(np.concatenate([o, g], -1),
                                    dtype=torch.float32)
                x2 = torch.as_tensor(np.concatenate([o2, g2], -1),
                                     dtype=torch.float32)
                a_i = torch.as_tensor(a, dtype=torch.long)
                r_t = torch.as_tensor(r, dtype=torch.float32)
                with torch.no_grad():
                    y = torch.clamp(r_t + gamma * q_t(x2).max(-1).values,
                                    -1.0 / (1 - gamma), 0.0)
                pred = q(x).gather(1, a_i[:, None]).squeeze(1)
                loss = ((pred - y) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                for pt, pm in zip(q_t.parameters(), q.parameters()):
                    pt.mul_(0.95).add_(pm, alpha=0.05)

        succ = []
        for _ in range(50):
            state = rng.integers(0, 2, n_bits).astype(np.float32)
            goal = rng.integers(0, 2, n_bits).astype(np.float32)
            hit = 0.0
            for _ in range(n_bits):
                with torch.no_grad():
                    x = torch.as_tensor(
                        np.concatenate([state, goal])[None],
                        dtype=torch.float32)
                    a = int(q(x).argmax())
                state = state.copy()
                state[a] = 1.0 - state[a]
                if np.array_equal(state, goal):
                    hit = 1.0
                    break
            succ.append(hit)
        print(f'  n={n_bits} her={use_her} epoch {epoch} '
              f'success {np.mean(succ):.2f}')
    return np.mean(succ)


if __name__ == '__main__':
    print('paper Fig 1: DQN fails past ~13 bits, DQN+HER solves')
    for n in (10, 20):
        for her in (True, False):
            run(n, her, epochs=4)
