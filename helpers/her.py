""" Hindsight Experience Replay (Andrychowicz et al., NeurIPS 2017,
    arXiv:1707.01495) for chunked SEAR. Roadmap step 4.

    Matches the paper:
      - goal = OBJECT position only, m(s) = s_object (their Sec 4.1
        "Goals"). The single-transition version of this logic is validated
        by bitflip_check.py against their Fig 1.
      - 'future' strategy: substitute goals are achieved states from later
        in the same sequence (their Sec 4.5, best in ablation)
      - sparse binary reward, one reward function for real and relabeled
        windows alike (their Alg. 1 computes r(s,a,g) for the original
        goal too, never mixing recorded env rewards with recomputed ones)

    TWO BUGS FIXED HERE, both from earlier non-paper additions:

    1. THE EFFECTOR WAS IN THE GOAL. An earlier version appended the
       end-effector position to the goal vector, reasoning that a static
       cube gives no reward variation while the arm always moves. That is
       not in the paper and it broke the run: the goal vector then meant
       "reach this arm pose AND place the cube," the reward gated cube
       credit on gripper proximity, and eval measured something the env
       never asked for. Goals are cube-only again.

    2. RELABELED GOALS SATISFIED ON ARRIVAL. A 'future' goal drawn from a
       cube that never moved equals where the cube already is, so the
       recomputed reward is 0 at every step of the window -- the critic
       learns "whatever you did was correct" and q_max pins at the return
       ceiling. The paper avoids this two ways: their pushing setup makes
       contact likely from the start (gripper above the table, object
       spawned directly under it), and Appendix A discards initial
       state-goal pairs where the goal is already satisfied. Here,
       reject_satisfied applies the same rule to relabeled goals: if the
       achieved state at the window start is already within the goal
       threshold, fall back to the episode's real goal instead.

    NOTE: rejection is a band-aid over a starved buffer, not a cure. If
    the cube never moves at all, there are no useful future goals to draw
    and rejection just returns the real goal every time. That is the
    exploration problem (roadmap step 3), and the fix is upstream.
"""
import numpy as np


def _mj(env):
    u = getattr(env, 'unwrapped', env)
    for m, d in (('model', 'data'), ('_model', '_data')):
        if hasattr(u, m) and hasattr(u, d):
            return getattr(u, m), getattr(u, d)
    raise RuntimeError('HER: env exposes no MuJoCo handles')


class GoalTools:
    """ Goals in RAW METRES: cube xyz per cube, read from free-joint qpos;
        the task goal is the mocap target the renderer uses. No obs-space
        rescaling and no calibration -- the earlier version rescaled goals
        into observation space, which added a second coordinate frame to
        keep in sync for no benefit the paper needed. """

    def __init__(self, env, thresh=0.04):
        import mujoco
        self.env = env
        self.thresh = thresh
        model, _ = _mj(env)
        free = int(mujoco.mjtJoint.mjJNT_FREE)
        self.adr = [int(model.jnt_qposadr[j]) for j in range(model.njnt)
                    if int(model.jnt_type[j]) == free]
        self.n_cubes = len(self.adr)
        self.goal_dim = 3 * self.n_cubes
        assert model.nmocap >= self.n_cubes, \
            'HER: fewer mocap targets than cubes'
        print(f'HER: {self.n_cubes} cube(s), goal_dim {self.goal_dim}, '
              f'threshold {thresh} m (raw metres, cube positions only)')

    def calibrate(self, obs):
        """ No-op. Kept so callers that calibrated per reset still work. """

    def task_goal(self):
        _, data = _mj(self.env)
        return np.asarray(data.mocap_pos[:self.n_cubes],
                          np.float32).reshape(-1)

    def achieved(self):
        _, data = _mj(self.env)
        return np.concatenate(
            [np.asarray(data.qpos[a:a + 3], np.float32) for a in self.adr])

    def reward(self, achieved, goal):
        """ -(number of cubes not at their goal), matching the env's own
            reward range. Vectorized over leading dims. """
        a = np.asarray(achieved).reshape(*np.shape(achieved)[:-1],
                                         self.n_cubes, 3)
        g = np.asarray(goal).reshape(*np.shape(goal)[:-1], self.n_cubes, 3)
        dist = np.linalg.norm(a - g, axis=-1)
        return -(dist > self.thresh).sum(-1).astype(np.float32)

    def satisfied(self, achieved, goal):
        """ True where every cube is already within threshold. """
        return self.reward(achieved, goal) == 0.0


def relabel_windows(seqs, chunk_len, goal_tools, task_goal, her_frac, rng,
                    obs_key='state', reject_satisfied=True):
    """ Per-window goals and rewards. Returns goals (B, W, G) and rewards
        (B, W, N), fully vectorized. """
    ach = seqs['achieved']                                  # (B, T, G)
    B, T, G = ach.shape
    N = chunk_len
    W = T - N
    if 'goal' in seqs:          # each episode's own collection-time goal
        real = np.ascontiguousarray(seqs['goal'][:, :W]).astype(np.float32)
    else:
        real = np.tile(task_goal, (B, W, 1)).astype(np.float32)

    # 'future': an achieved state at or after the window's end, so it is
    # strictly future for every prefix inside the chunk
    lows = np.minimum(np.arange(W) + N, T - 1)              # (W,)
    j = rng.integers(np.broadcast_to(lows, (B, W)), T)      # (B, W)
    future = ach[np.arange(B)[:, None], j]                  # (B, W, G)

    use_her = rng.random((B, W)) < her_frac
    if reject_satisfied:
        # drop future goals already met at the window start -- they would
        # pay reward 0 for every prefix and teach nothing
        start = ach[:, :W]                                  # (B, W, G)
        use_her &= ~goal_tools.satisfied(start, future)
    goals = np.where(use_her[..., None], future, real).astype(np.float32)

    idx = np.arange(1, N + 1)[None, :] + np.arange(W)[:, None]   # (W, N)
    ach_win = ach[:, idx]                                   # (B, W, N, G)
    rewards = goal_tools.reward(ach_win, goals[:, :, None, :])
    return goals, rewards.astype(np.float32)
