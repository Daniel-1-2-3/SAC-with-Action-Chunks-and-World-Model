""" Hindsight Experience Replay (Andrychowicz et al., NeurIPS 2017,
    arXiv:1707.01495) support for goal-conditioned SEAR.

    Paper mechanics implemented here:
      - policy/value inputs are concat(state, goal)  (their Sec. 3.1)
      - relabeling: transitions are replayed with substitute goals; the
        reward is recomputed, which is valid because changing the goal
        does not change the dynamics
      - goal sampling: the 'future' strategy -- substitute goals are
        achieved states sampled from LATER in the same episode -- which
        their ablation found best (Sec. 4.5)
      - rewards stay sparse and binary: 0 within a distance threshold of
        the goal, -1 otherwise (their Eq. in Sec. 3.2; threshold measured
        against this env: reward flips at ~4 cm from the target)

    Goal representation for OGBench cubes: the cube's xyz (3 dims per
    cube). The environment's own task goal is read from the sim's mocap
    target -- the same handle the renderer uses -- and the achieved goal
    is the cube's current position from the free-joint qpos.
"""
import numpy as np


def _mj(env):
    u = getattr(env, 'unwrapped', env)
    for m, d in (('model', 'data'), ('_model', '_data')):
        if hasattr(u, m) and hasattr(u, d):
            return getattr(u, m), getattr(u, d)
    raise RuntimeError('HER: env exposes no MuJoCo handles')


class GoalTools:
    def __init__(self, env, thresh=0.04):
        import mujoco
        self.env = env
        self.thresh = thresh
        model, data = _mj(env)
        free = int(mujoco.mjtJoint.mjJNT_FREE)
        self.adr = [int(model.jnt_qposadr[j]) for j in range(model.njnt)
                    if int(model.jnt_type[j]) == free]
        self.n_cubes = len(self.adr)
        self.goal_dim = 3 * self.n_cubes
        assert model.nmocap >= self.n_cubes, \
            'HER: fewer mocap targets than cubes'
        # goals are expressed in OBSERVATION space so the critic compares
        # like with like: OGBench obs store cube positions as
        # 10*(pos - center); goals in raw meters would force the critic to
        # learn a cross-scale relation (verified blocker in the toy test).
        # The center is calibrated from one (obs, qpos) pair per reset;
        # the 10x scale is OGBench's documented obs scaling.
        self.scale = 10.0
        self.obs_start = 19             # OGBench: cube i block at 19 + 9i
        self.obs_stride = 9
        self.center = np.zeros((self.n_cubes, 3), np.float32)
        print(f'HER: {self.n_cubes} cube(s), goal_dim {self.goal_dim}, '
              f'success threshold {thresh} m (goals in obs space, x10)')

    def calibrate(self, obs):
        """ Solve the per-cube obs offset from the current sim state:
            obs_block = scale * (qpos - center)  =>
            center = qpos - obs_block / scale. Call after each reset. """
        obs = np.asarray(obs, dtype=np.float32)
        _, data = _mj(self.env)
        for i, a in enumerate(self.adr):
            blk = self.obs_start + self.obs_stride * i
            self.center[i] = (np.asarray(data.qpos[a:a + 3], np.float32)
                              - obs[blk:blk + 3] / self.scale)

    def to_obs_space(self, raw):
        """ raw (..., n_cubes*3) meters -> obs-space goal features. """
        r = np.asarray(raw, np.float32).reshape(
            *np.shape(raw)[:-1], self.n_cubes, 3)
        return (self.scale * (r - self.center)).reshape(
            *np.shape(raw)[:-1], self.goal_dim)

    def task_goal(self):
        """ The env's own goal: mocap target position(s), (goal_dim,). """
        _, data = _mj(self.env)
        raw = np.asarray(data.mocap_pos[:self.n_cubes],
                         dtype=np.float32).reshape(-1)
        return self.to_obs_space(raw)

    def achieved(self):
        """ Current cube position(s), (goal_dim,). """
        _, data = _mj(self.env)
        raw = np.concatenate(
            [np.asarray(data.qpos[a:a + 3], dtype=np.float32)
             for a in self.adr])
        return self.to_obs_space(raw)

    def reward(self, achieved, goal):
        """ Sparse binary reward, vectorized over leading dims.
            achieved/goal: (..., goal_dim). All cubes must be within
            threshold (matches the env: -1 per unplaced cube collapses to
            0/-1 for single; for multi-cube we return the env-style
            -(number of unplaced cubes)). """
        a = np.asarray(achieved).reshape(*np.shape(achieved)[:-1],
                                         self.n_cubes, 3)
        g = np.asarray(goal).reshape(*np.shape(goal)[:-1], self.n_cubes, 3)
        placed = (np.linalg.norm(a - g, axis=-1) < self.thresh * self.scale)
        return placed.sum(-1).astype(np.float32) - float(self.n_cubes)


def relabel_windows(seqs, chunk_len, goal_tools, task_goal, her_frac, rng,
                    obs_key='state'):
    """ Given sampled sequences that include per-step 'achieved' (B, T, G),
        produce a per-window goal array and per-window relabeled rewards,
        following HER's 'future' strategy at the window level: with prob
        her_frac a window's goal becomes an achieved state sampled from
        later in ITS OWN sequence; otherwise the real task goal is kept
        with the recorded rewards.

        Returns (goals (B, W, G), rewards (B, T)) where rewards is a full
        (B, T) relabeled reward array windows can be sliced from -- but
        note relabeled rewards differ per window goal, so this returns a
        per-window reward tensor instead: (B, W, N). W = T - chunk_len.
    """
    ach = seqs['achieved']                                # (B, T, G)
    B, T, G = ach.shape
    N = chunk_len
    W = T - N
    if 'goal' in seqs:      # each episode's own collection-time goal
        goals = np.ascontiguousarray(
            seqs['goal'][:, :W]).astype(np.float32)
    else:
        goals = np.tile(task_goal, (B, W, 1)).astype(np.float32)
    # recorded rewards for the real-goal case, sliced per window
    rew = seqs['reward']
    rewards = np.stack([rew[:, t + 1: t + N + 1] for t in range(W)], 1)
    use_her = rng.random((B, W)) < her_frac
    for b in range(B):
        for t in range(W):
            if not use_her[b, t]:
                continue
            # future strategy, chunked adaptation: goal = achieved at a
            # random index AT OR AFTER the window's end, so the goal is
            # strictly future for every prefix inside the window
            j = int(rng.integers(min(t + N, T - 1), T))
            goals[b, t] = ach[b, j]
            rewards[b, t] = goal_tools.reward(
                ach[b, t + 1: t + N + 1], ach[b, j])
    return goals, rewards
