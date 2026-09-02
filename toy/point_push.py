""" A tiny, CPU-cheap stand-in for the OGBench cube tasks, for testing the
    arms against each other in minutes rather than days.

    The task has the two properties that make cube hard and that every arm
    here is designed around: a SPARSE OGBench-style reward (-1 per step until
    the block is in the goal, 0 after) and a real need for temporally coherent
    action sequences (a point agent must reach the block, then push it to the
    goal; random per-step actions almost never do that).

    Observation (10): agent xy, block xy, goal xy, block - agent, goal - block.
    Action (2): agent velocity in [-1, 1]^2, scaled by `speed`.
    eef_slice for the coherence metric is (0, 2), the agent position.

    The offline dataset comes from a noisy scripted pusher, so like the
    OGBench "play" data it contains partial and full successes mixed with
    junk, and a behaviour-constrained policy can extract a decent prior from
    it. """

import numpy as np


class _Box:
    def __init__(self, low, high, shape):
        self.low = np.full(shape, low, np.float32)
        self.high = np.full(shape, high, np.float32)
        self.shape = shape
        self._rng = np.random.default_rng(0)

    def seed(self, s):
        self._rng = np.random.default_rng(s)

    def sample(self):
        return self._rng.uniform(self.low, self.high).astype(np.float32)


class PointPushEnv:
    """ Gymnasium-style API: reset(seed=) -> (obs, info); step(a) ->
        (obs, reward, terminated, truncated, info). """

    def __init__(self, max_steps=150, speed=0.05, goal_radius=0.15,
                 push_radius=0.12, seed=0):
        self.max_steps = max_steps
        self.speed = speed
        self.goal_radius = goal_radius
        self.push_radius = push_radius
        self.observation_space = _Box(-2.0, 2.0, (10,))
        self.action_space = _Box(-1.0, 1.0, (2,))
        self._rng = np.random.default_rng(seed)
        self.t = 0

    def _obs(self):
        return np.concatenate([
            self.agent, self.block, self.goal,
            self.block - self.agent, self.goal - self.block]).astype(np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        r = self._rng
        self.block = r.uniform(-0.4, 0.4, 2).astype(np.float32)
        self.agent = r.uniform(-0.9, 0.9, 2).astype(np.float32)
        while np.linalg.norm(self.agent - self.block) < 0.25:
            self.agent = r.uniform(-0.9, 0.9, 2).astype(np.float32)
        self.goal = r.uniform(-0.6, 0.6, 2).astype(np.float32)
        while np.linalg.norm(self.goal - self.block) < 0.4:
            self.goal = r.uniform(-0.6, 0.6, 2).astype(np.float32)
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, np.float32), -1.0, 1.0) * self.speed
        new_agent = np.clip(self.agent + a, -1.0, 1.0)
        # Pushing contact: while the agent is within push_radius of the block
        # AND moving into it, the block moves WITH the agent's displacement.
        # You can push, not pull. A held action then pushes the block in a
        # straight line, which is what makes the task solvable by open-loop
        # chunks; geometric "slide off the side" contact is not, and defeats
        # even a perfect controller held for 5 steps (measured: 8% success).
        # Moving around the block to line up a push never drags it.
        toward = self.block - self.agent
        if np.linalg.norm(toward) < self.push_radius and float(a @ toward) > 0.0:
            self.block = np.clip(self.block + a, -1.0, 1.0)
        self.agent = new_agent
        self.t += 1
        success = bool(np.linalg.norm(self.goal - self.block) < self.goal_radius)
        reward = 0.0 if success else -1.0
        terminated = success
        truncated = self.t >= self.max_steps and not terminated
        return self._obs(), reward, terminated, truncated, {'success': success}

    def render(self):
        raise NotImplementedError('no rendering for the toy task')

    def close(self):
        pass


def scripted_action(obs, rng, noise=0.5, p_random=0.15, push_radius=0.12):
    """ Noisy pusher: get behind the block relative to the goal WITHOUT
        touching it (detour tangentially when the straight path would push
        it the wrong way), then push toward the goal. With p_random it emits
        a uniformly random action, and it always adds Gaussian noise -- so
        the data has coherent skill segments AND junk, like "play" data. """
    if rng.random() < p_random:
        return rng.uniform(-1, 1, 2).astype(np.float32)
    agent, block, goal = obs[0:2], obs[2:4], obs[4:6]
    to_goal = goal - block
    to_goal /= (np.linalg.norm(to_goal) + 1e-8)
    behind = block - to_goal * 0.10
    if np.linalg.norm(behind - agent) > 0.07:
        target = behind - agent
        rel = block - agent
        dist = np.linalg.norm(rel)
        # Would this step move INTO the block from the wrong side? Then go
        # around it: move along the tangent that heads toward the waypoint.
        if dist < push_radius + 0.06 and float(target @ rel) > 0.0:
            tangent = np.array([-rel[1], rel[0]], np.float32)
            if float(tangent @ target) < 0.0:
                tangent = -tangent
            target = tangent
    else:
        target = to_goal
    a = target / (np.linalg.norm(target) + 1e-8) + rng.normal(0, noise, 2)
    return np.clip(a, -1, 1).astype(np.float32)


def make_offline_dataset(num_episodes=300, seed=0, **env_kwargs):
    """ OGBench-shaped flat dataset dict. """
    env = PointPushEnv(seed=seed, **env_kwargs)
    rng = np.random.default_rng(seed + 1)
    keys = ['observations', 'actions', 'rewards', 'next_observations',
            'terminals', 'masks']
    data = {k: [] for k in keys}
    successes = 0
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + 1000 + ep)
        done = False
        ep_succ = False
        while not done:
            a = scripted_action(obs, rng)
            nobs, r, term, trunc, info = env.step(a)
            data['observations'].append(obs)
            data['actions'].append(a)
            data['rewards'].append(r)
            data['next_observations'].append(nobs)
            data['terminals'].append(bool(term or trunc))
            data['masks'].append(0.0 if term else 1.0)
            ep_succ = ep_succ or info['success']
            obs = nobs
            done = term or trunc
        successes += int(ep_succ)
    out = {k: np.asarray(v) for k, v in data.items()}
    out['observations'] = out['observations'].astype(np.float32)
    out['next_observations'] = out['next_observations'].astype(np.float32)
    out['actions'] = out['actions'].astype(np.float32)
    out['rewards'] = out['rewards'].astype(np.float32)
    out['masks'] = out['masks'].astype(np.float32)
    print(f'[toy] offline dataset: {num_episodes} episodes, '
          f'{len(out["rewards"])} transitions, '
          f'scripted success rate {successes / num_episodes:.2f}')
    return out


def make_toy_env_and_dataset(num_episodes=300, seed=0):
    return PointPushEnv(seed=seed), make_offline_dataset(num_episodes, seed)
