""" Hindsight Experience Replay, faithful to Andrychowicz et al. 2017
    (arXiv:1707.01495), Algorithm 1 + Appendix A. Standalone: no SEAR,
    no chunks, no world model.

    Paper spec implemented exactly:
      - DDPG, actor/critic 3 hidden layers x 64 ReLU, actor tanh output
        scaled to the action range, squared preactivations added to the
        actor loss (Appendix A "Network architecture")
      - goal = object position(s) only, m(s) = s_object; reward
        r(s,a,g) = -[|g - s'_object| > eps] on the state AFTER the action
        (Sec 4.1 "Goals"/"Rewards"); no effector anywhere
      - 'future' relabeling with k=4: 4 of every 5 sampled transitions get
        a goal achieved later in the same episode (Sec 4.5). Implemented
        as sample-time relabeling with p = k/(k+1) = 0.8, the form used in
        the authors' released code; same replay distribution as storing
        k+1 copies
      - initial state-goal pairs where the goal is already satisfied are
        discarded (Appendix A "State-goal distributions")
      - gamma = 0.98 for ALL transitions including episode-ending ones (no
        terminal masking); critic targets clipped to [-1/(1-gamma), 0]
      - input normalization: running mean/std over everything encountered,
        clipped to [-5, 5]
      - exploration: 20% uniform random actions, else policy + Gaussian
        noise with std = 5% of the action range per coordinate
      - training loop: epochs of 50 cycles; each cycle = 16 episodes then
        40 optimization steps on batches of 128 from a 1e6 buffer; target
        nets polyak-updated after every cycle with decay 0.95; the target
        actor is used for evaluation episodes

    Deviations forced by the environment, and nothing else:
      - OGBench episodes are 200 steps, not 50, and the action space is
        the env's own (5-dim). The cycle is kept at the paper's 16
        episodes / 40 updates; --updates_per_cycle 160 instead preserves
        the paper's updates-per-env-step ratio (40 per 800 steps)
      - eps = 0.04 m, the distance at which this env's own reward flips
      - single fixed goal (OGBench singletask), the paper's Sec 4.3 setup
"""
import argparse
import os
os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------- goals
class GoalTools:
    def __init__(self, env, thresh):
        import mujoco
        self.env = env
        self.thresh = thresh
        u = getattr(env, 'unwrapped', env)
        model, self._data = u.model, u.data
        free = int(mujoco.mjtJoint.mjJNT_FREE)
        self.adr = [int(model.jnt_qposadr[j]) for j in range(model.njnt)
                    if int(model.jnt_type[j]) == free]
        self.n = len(self.adr)
        self.goal_dim = 3 * self.n

    def achieved(self):
        return np.concatenate(
            [np.asarray(self._data.qpos[a:a + 3], np.float32)
             for a in self.adr])

    def task_goal(self):
        return np.asarray(self._data.mocap_pos[:self.n],
                          np.float32).reshape(-1)

    def reward(self, achieved, goal):
        a = np.asarray(achieved).reshape(*np.shape(achieved)[:-1], self.n, 3)
        g = np.asarray(goal).reshape(*np.shape(goal)[:-1], self.n, 3)
        dist = np.linalg.norm(a - g, axis=-1)
        return -(dist > self.thresh).sum(-1).astype(np.float32)


# ------------------------------------------------------------- networks
def mlp(inp, out):
    return nn.Sequential(nn.Linear(inp, 64), nn.ReLU(),
                         nn.Linear(64, 64), nn.ReLU(),
                         nn.Linear(64, 64), nn.ReLU(),
                         nn.Linear(64, out))


class Actor(nn.Module):
    def __init__(self, inp, act_dim, act_high):
        super().__init__()
        self.net = mlp(inp, act_dim)
        self.register_buffer('high', torch.as_tensor(act_high,
                                                     dtype=torch.float32))

    def forward(self, x):
        pre = self.net(x)
        return torch.tanh(pre) * self.high, pre


class Normalizer:
    def __init__(self, dim, clip=5.0, eps=1e-4):
        self.mean = np.zeros(dim, np.float64)
        self.m2 = np.ones(dim, np.float64)
        self.count = eps
        self.clip = clip

    def update(self, x):
        x = np.asarray(x, np.float64).reshape(-1, self.mean.shape[0])
        n = x.shape[0]
        d = x.mean(0) - self.mean
        tot = self.count + n
        self.mean += d * n / tot
        self.m2 += x.var(0) * n + d ** 2 * self.count * n / tot
        self.count = tot

    def __call__(self, x):
        std = np.sqrt(self.m2 / self.count)
        return np.clip((x - self.mean) / np.maximum(std, 1e-6),
                       -self.clip, self.clip).astype(np.float32)


# --------------------------------------------------------------- replay
class Replay:
    def __init__(self, capacity):
        self.eps, self.size, self.capacity = [], 0, capacity

    def add(self, ep):
        self.eps.append(ep)
        self.size += len(ep['act'])
        while self.size > self.capacity:
            self.size -= len(self.eps.pop(0)['act'])

    def sample(self, batch, future_p, gt, rng):
        lens = np.array([len(e['act']) for e in self.eps])
        ei = rng.choice(len(self.eps), batch, p=lens / lens.sum())
        o, g, a, o2, g2, r = [], [], [], [], [], []
        for i in ei:
            e = self.eps[i]
            T = len(e['act'])
            t = rng.integers(T)
            goal = e['goal']
            if rng.random() < future_p:
                goal = e['ach'][rng.integers(t + 1, T + 1)]
            o.append(e['obs'][t]); a.append(e['act'][t])
            o2.append(e['obs'][t + 1])
            g.append(goal); g2.append(goal)
            r.append(gt.reward(e['ach'][t + 1], goal))
        return (np.array(o), np.array(g), np.array(a), np.array(r),
                np.array(o2), np.array(g2))


# ---------------------------------------------------------------- train
def train(args):
    import ogbench
    import wandb
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    env = ogbench.make_env_and_datasets(args.env_name, env_only=True)
    gt = GoalTools(env, args.thresh)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_low = np.asarray(env.action_space.low, np.float32)
    act_high = np.asarray(env.action_space.high, np.float32)
    inp = obs_dim + gt.goal_dim
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    actor = Actor(inp, act_dim, act_high).to(dev)
    critic = mlp(inp + act_dim, 1).to(dev)
    actor_t = Actor(inp, act_dim, act_high).to(dev)
    critic_t = mlp(inp + act_dim, 1).to(dev)
    actor_t.load_state_dict(actor.state_dict())
    critic_t.load_state_dict(critic.state_dict())
    opt_a = torch.optim.Adam(actor.parameters(), lr=1e-3)
    opt_c = torch.optim.Adam(critic.parameters(), lr=1e-3)
    norm_o = Normalizer(obs_dim)
    norm_g = Normalizer(gt.goal_dim)
    replay = Replay(int(1e6))
    gamma, ret_min = args.gamma, -1.0 / (1.0 - args.gamma)
    future_p = args.k / (args.k + 1.0)
    noise_std = 0.05 * (act_high - act_low)

    wandb.init(project=args.wandb_project, name=os.path.basename(args.out_dir),
               config=vars(args), mode=args.wandb_mode)
    os.makedirs(args.out_dir, exist_ok=True)

    def inputs(obs, goal, norm=True):
        if norm:
            obs, goal = norm_o(obs), norm_g(goal)
        return torch.as_tensor(np.concatenate([obs, goal], -1),
                               dtype=torch.float32, device=dev)

    def reset_valid():
        for _ in range(10):
            obs, _ = env.reset()
            if gt.reward(gt.achieved(), gt.task_goal()) < 0:
                return obs
        return obs

    env_steps = 0
    for epoch in range(args.epochs):
        for cycle in range(50):
            for _ in range(16):
                obs = reset_valid()
                goal = gt.task_goal()
                ep = {'obs': [obs], 'ach': [gt.achieved()], 'act': [],
                      'goal': goal}
                for _ in range(args.max_episode_steps):
                    if rng.random() < 0.2:
                        act = rng.uniform(act_low, act_high).astype(
                            np.float32)
                    else:
                        with torch.no_grad():
                            act, _ = actor(inputs(obs[None], goal[None]))
                        act = act.cpu().numpy()[0]
                        act = np.clip(act + rng.normal(0, noise_std),
                                      act_low, act_high).astype(np.float32)
                    obs, _, term, trunc, _ = env.step(act)
                    ep['obs'].append(obs)
                    ep['ach'].append(gt.achieved())
                    ep['act'].append(act)
                    env_steps += 1
                    if term or trunc:
                        break
                for k in ('obs', 'ach', 'act'):
                    ep[k] = np.array(ep[k], np.float32)
                replay.add(ep)
                norm_o.update(ep['obs'])
                norm_g.update(np.concatenate([ep['ach'], goal[None]]))

            disp = [float(np.linalg.norm(e['ach'][-1] - e['ach'][0]))
                    for e in replay.eps[-16:]]
            cl_acc, al_acc, q_acc, rz_acc = [], [], [], []
            for _ in range(args.updates_per_cycle):
                o, g, a, r, o2, g2 = replay.sample(args.batch_size,
                                                   future_p, gt, rng)
                x = inputs(o, g)
                x2 = inputs(o2, g2)
                a_t = torch.as_tensor(a, dtype=torch.float32, device=dev)
                r_t = torch.as_tensor(r, dtype=torch.float32, device=dev)
                with torch.no_grad():
                    a2, _ = actor_t(x2)
                    q2 = critic_t(torch.cat([x2, a2], -1)).squeeze(-1)
                    y = torch.clamp(r_t + gamma * q2, ret_min, 0.0)
                q = critic(torch.cat([x, a_t], -1)).squeeze(-1)
                cl = ((q - y) ** 2).mean()
                opt_c.zero_grad(); cl.backward(); opt_c.step()

                pi, pre = actor(x)
                al = -critic(torch.cat([x, pi], -1)).mean() \
                    + (pre ** 2).mean()
                opt_a.zero_grad(); al.backward(); opt_a.step()
                cl_acc.append(cl.item()); al_acc.append(al.item())
                q_acc.append(q.max().item())
                rz_acc.append(float((r == 0).mean()))

            with torch.no_grad():
                for t, m in ((actor_t, actor), (critic_t, critic)):
                    for pt, pm in zip(t.parameters(), m.parameters()):
                        pt.mul_(args.polyak).add_(pm, alpha=1 - args.polyak)

            wandb.log({'her/critic_loss': np.mean(cl_acc),
                       'her/actor_loss': np.mean(al_acc),
                       'her/q_max': np.max(q_acc),
                       'her/relabeled_reward_zero_frac': np.mean(rz_acc),
                       'her/cube_disp_mean': float(np.mean(disp)),
                       'her/cube_moved_frac': float(
                           np.mean(np.array(disp) > 0.01)),
                       'her/buffer_transitions': replay.size},
                      step=env_steps)

        succ, rets = [], []
        for _ in range(args.eval_episodes):
            obs, _ = env.reset()
            goal, ret, done = gt.task_goal(), 0.0, False
            for _ in range(args.max_episode_steps):
                with torch.no_grad():
                    act, _ = actor_t(inputs(obs[None], goal[None]))
                obs, r, term, trunc, info = env.step(act.cpu().numpy()[0])
                ret += r
                done = bool(info.get('success', 0)) or done
                if term or trunc:
                    break
            succ.append(float(done or
                              gt.reward(gt.achieved(), goal) == 0))
            rets.append(ret)
        wandb.log({'eval/success_rate': np.mean(succ),
                   'eval/mean_return': np.mean(rets),
                   'her/epoch': epoch}, step=env_steps)
        print(f'epoch {epoch} steps {env_steps} '
              f'success {np.mean(succ):.2f} return {np.mean(rets):.1f}')
        torch.save({'actor': actor.state_dict(),
                    'critic': critic.state_dict(),
                    'actor_t': actor_t.state_dict(),
                    'critic_t': critic_t.state_dict(),
                    'norm_o': vars(norm_o), 'norm_g': vars(norm_g)},
                   os.path.join(args.out_dir, 'her_latest.pt'))
    wandb.finish()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--env_name', default='cube-single-play-singletask-v0')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--max_episode_steps', type=int, default=200)
    p.add_argument('--updates_per_cycle', type=int, default=40)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--k', type=int, default=4)
    p.add_argument('--gamma', type=float, default=0.98)
    p.add_argument('--polyak', type=float, default=0.95)
    p.add_argument('--thresh', type=float, default=0.04)
    p.add_argument('--eval_episodes', type=int, default=20)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out_dir', default='sear_runs/her_ddpg_single')
    p.add_argument('--wandb_project', default='sear-p2e')
    p.add_argument('--wandb_mode', default='online')
    train(p.parse_args())
