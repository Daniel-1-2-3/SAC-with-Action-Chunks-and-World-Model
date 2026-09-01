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
        # The achieved goal is [end-effector position] + [cube positions].
        # Cube-only goals are DEGENERATE before first contact: an untouched
        # cube never moves, so a 'future' relabeled goal equals the cube's
        # current position and the recomputed reward is 0 at every step --
        # ~80% of windows then say 'whatever you did was correct', which
        # trains a flat critic and an arbitrary policy. The effector moves
        # every step, so including it makes relabeled rewards vary from the
        # first episode: the agent first learns precise 3D positioning (the
        # prerequisite for grasping), and cube-moving goals become
        # learnable once contact starts happening.
        self.eef_kind, self.eef_id = self._find_effector(model)
        self.goal_dim = 3 * (1 + self.n_cubes)
        print(f'HER: {self.n_cubes} cube(s) + effector, goal_dim '
              f'{self.goal_dim}, threshold {thresh} m cube / '
              f'{2 * thresh} m effector (obs space, x10)')

    @staticmethod
    def _find_effector(model):
        import mujoco
        keys = ('effector', 'pinch', 'grip', 'hand', 'attach', 'wrist',
                'tool', 'ee')
        for kind, n, obj in (('site', model.nsite, mujoco.mjtObj.mjOBJ_SITE),
                             ('body', model.nbody, mujoco.mjtObj.mjOBJ_BODY)):
            names = [(i, mujoco.mj_id2name(model, obj, i)) for i in range(n)]
            for i, nm in names:
                if nm and any(k in nm.lower() for k in keys) \
                        and 'target' not in nm.lower():
                    print(f'HER: effector = {kind} "{nm}"')
                    return kind, i
        raise RuntimeError(
            'HER: could not find an effector site/body; names were '
            + str([mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
                   for i in range(model.nsite)]))

    def _eef_raw(self):
        _, data = _mj(self.env)
        src = data.site_xpos if self.eef_kind == 'site' else data.xpos
        return np.asarray(src[self.eef_id], dtype=np.float32)

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

    def _scale(self, raw):
        """ raw (..., 3*(1+n_cubes)) meters -> obs-space goal features.
            All components share cube 0's calibrated center so the goal
            vector sits in the same numeric range as the obs cube block. """
        r = np.asarray(raw, np.float32).reshape(
            *np.shape(raw)[:-1], 1 + self.n_cubes, 3)
        return (self.scale * (r - self.center[0])).reshape(
            *np.shape(raw)[:-1], self.goal_dim)

    def task_goal(self):
        """ The env's own goal: mocap target position(s), (goal_dim,). """
        _, data = _mj(self.env)
        cubes = np.asarray(data.mocap_pos[:self.n_cubes],
                           dtype=np.float32).reshape(self.n_cubes, 3)
        # effector component of the task goal = the first cube's target
        # (to place a cube there the gripper must be there)
        raw = np.concatenate([cubes[0], cubes.reshape(-1)])
        return self._scale(raw)

    def achieved(self):
        """ Current cube position(s), (goal_dim,). """
        _, data = _mj(self.env)
        raw = np.concatenate(
            [self._eef_raw()] +
            [np.asarray(data.qpos[a:a + 3], dtype=np.float32)
             for a in self.adr])
        return self._scale(raw)

    def reward(self, achieved, goal):
        """ Sparse binary reward, vectorized over leading dims.
            achieved/goal: (..., goal_dim). All cubes must be within
            threshold (matches the env: -1 per unplaced cube collapses to
            0/-1 for single; for multi-cube we return the env-style
            -(number of unplaced cubes)). """
        a = np.asarray(achieved).reshape(*np.shape(achieved)[:-1],
                                         1 + self.n_cubes, 3)
        g = np.asarray(goal).reshape(*np.shape(goal)[:-1],
                                     1 + self.n_cubes, 3)
        dist = np.linalg.norm(a - g, axis=-1)
        grip_ok = dist[..., 0] < 2 * self.thresh * self.scale
        placed = (dist[..., 1:] < self.thresh * self.scale).sum(-1)
        # the effector gates cube credit, so reward varies with the arm's
        # own motion from episode one; range matches the env (-n_cubes..0)
        return (placed * grip_ok).astype(np.float32) - float(self.n_cubes)


def relabel_windows(seqs, chunk_len, goal_tools, task_goal, her_frac, rng,
                    obs_key='state'):
    """ Per-window goals and rewards, HER Algorithm 1 adapted to chunks.

        Two points of faithfulness that matter:
        - ONE reward function everywhere. The paper computes r(s,a,g) for
          the ORIGINAL goal as well as for relabeled ones; it never mixes
          recorded environment rewards with recomputed ones. Doing that
          would give the same goal vector two different meanings depending
          on whether its window happened to be relabeled, which is a
          contradiction the critic cannot resolve. Env rewards remain the
          reported metric; they are not a training target here.
        - 'future' strategy: the goal is an achieved state from later in
          the SAME sequence, at or after the window's end so it is strictly
          future for every prefix inside the chunk.

        Returns goals (B, W, G) and rewards (B, W, N), fully vectorized.
    """
    ach = seqs['achieved']                                  # (B, T, G)
    B, T, G = ach.shape
    N = chunk_len
    W = T - N
    if 'goal' in seqs:          # each episode's own collection-time goal
        goals = np.ascontiguousarray(seqs['goal'][:, :W]).astype(np.float32)
    else:
        goals = np.tile(task_goal, (B, W, 1)).astype(np.float32)

    # future goals for every window, then keep them with prob her_frac
    lows = np.minimum(np.arange(W) + N, T - 1)              # (W,)
    j = rng.integers(np.broadcast_to(lows, (B, W)), T)      # (B, W)
    future = ach[np.arange(B)[:, None], j]                  # (B, W, G)
    use_her = (rng.random((B, W)) < her_frac)[..., None]
    goals = np.where(use_her, future, goals).astype(np.float32)

    # rewards from the single reward function, for every window and prefix
    idx = np.arange(1, N + 1)[None, :] + np.arange(W)[:, None]   # (W, N)
    ach_win = ach[:, idx]                                   # (B, W, N, G)
    rewards = goal_tools.reward(ach_win, goals[:, :, None, :])
    return goals, rewards.astype(np.float32)
