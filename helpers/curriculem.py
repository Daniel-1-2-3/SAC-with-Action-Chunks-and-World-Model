""" Training-time initial-state curriculum. Two mechanisms, both using only
    the simulator's own state -- no demonstrations, no external data, no
    task knowledge:

    1. SPAWN RANDOMIZATION. At reset, cube free-joint positions are
       resampled uniformly inside the box spanned by positions the env
       itself produces at natural resets (learned online, nothing
       hardcoded). Goal zones occupy part of that box, so a fraction of
       randomized spawns start with a cube already in its zone -- which
       manufactures reward-bearing transitions without anyone having to
       find them first, and creates the 'one cube placed' base camp from
       which the second cube can be learned.

    2. RESET-TO-POOLED-STATE. Whenever a step earns better-than-floor
       reward, the full simulator state (qpos, qvel) is snapshotted into a
       pool. A fraction of later training episodes start from a pooled
       state, so states that were reached once are revisited thousands of
       times (Go-Explore's return-then-explore, restricted to states the
       agent itself discovered).

    EVAL NEVER USES EITHER. Evaluation always uses the env's standard
    reset and full episode length; the curriculum shapes only where
    training episodes begin.

    Everything is discovered at runtime and degrades to a no-op with a
    printed reason if the env does not expose the needed MuJoCo handles.
"""
import numpy as np

_OBS_METHODS = ('compute_observation', '_compute_observation', '_get_obs',
                'get_obs', '_get_ob', 'get_ob')


def _mj(env):
    """ (model, data) from a MuJoCo-backed gym env, or None. """
    u = getattr(env, 'unwrapped', env)
    for m, d in (('model', 'data'), ('_model', '_data'),
                 ('sim_model', 'sim_data')):
        if hasattr(u, m) and hasattr(u, d):
            return getattr(u, m), getattr(u, d)
    return None


def _recompute_obs(env):
    """ Ask the env to rebuild its observation from current sim state. """
    u = getattr(env, 'unwrapped', env)
    for name in _OBS_METHODS:
        fn = getattr(u, name, None)
        if callable(fn):
            try:
                return np.asarray(fn(), dtype=np.float32)
            except TypeError:
                continue
    raise RuntimeError(
        'curriculum: could not find an observation method on the env '
        f'(tried {_OBS_METHODS}); disable with curriculum_enabled=False')


class Curriculum:
    def __init__(self, env, spawn_frac=0.0, pool_frac=0.0, pool_size=2000,
                 reward_thresh=-1.5, rng=None, verbose=True):
        self.env = env
        self.spawn_frac = spawn_frac
        self.pool_frac = pool_frac
        self.pool_size = pool_size
        self.reward_thresh = reward_thresh
        self.rng = rng or np.random.default_rng()
        self.pool = []
        self.enabled = False
        self.free_adr = []          # qpos start index of each free joint
        self._lo = self._hi = None  # observed xy box per free joint
        self._resets_seen = 0
        self.warmup_resets = 20     # natural resets used to bound the box
        self.stats = {'pool_size': 0, 'spawned': 0, 'restored': 0}

        handles = _mj(env)
        if handles is None:
            if verbose:
                print('curriculum: no MuJoCo handles on env -- DISABLED')
            return
        model, _ = handles
        try:
            import mujoco
            free = int(mujoco.mjtJoint.mjJNT_FREE)
        except Exception:
            free = 0                # mjJNT_FREE == 0 in MuJoCo
        try:
            for j in range(model.njnt):
                if int(model.jnt_type[j]) == free:
                    self.free_adr.append(int(model.jnt_qposadr[j]))
        except Exception as e:
            if verbose:
                print(f'curriculum: joint scan failed ({e}) -- DISABLED')
            return
        if not self.free_adr:
            if verbose:
                print('curriculum: no free joints found -- DISABLED')
            return
        self.enabled = True
        if verbose:
            print(f'curriculum: {len(self.free_adr)} free joints '
                  f'(objects) found | spawn_frac={spawn_frac} '
                  f'pool_frac={pool_frac}')

    # ---------- state snapshot / restore ----------
    def snapshot(self):
        model, data = _mj(self.env)
        return (np.array(data.qpos, copy=True),
                np.array(data.qvel, copy=True))

    def _apply(self, qpos, qvel):
        import mujoco
        model, data = _mj(self.env)
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_forward(model, data)
        return _recompute_obs(self.env)

    def maybe_pool(self, reward):
        """ Call after each env step; snapshots reward-bearing states. """
        if not self.enabled or self.pool_frac <= 0:
            return
        if reward > self.reward_thresh:
            self.pool.append(self.snapshot())
            if len(self.pool) > self.pool_size:
                self.pool.pop(self.rng.integers(0, len(self.pool)))
            self.stats['pool_size'] = len(self.pool)

    # ---------- reset hook ----------
    def on_reset(self, obs):
        """ Called right after env.reset(). Returns the (possibly new)
            observation. Order: pooled restore first, else spawn
            randomization, else untouched. """
        if not self.enabled:
            return obs
        self._observe_box()
        r = self.rng.random()
        if self.pool and r < self.pool_frac:
            qpos, qvel = self.pool[self.rng.integers(0, len(self.pool))]
            try:
                obs = self._apply(qpos, qvel)
                self.stats['restored'] += 1
            except Exception as e:
                print(f'curriculum: restore failed ({e}) -- DISABLING')
                self.enabled = False
            return obs
        if (self._resets_seen >= self.warmup_resets
                and r < self.pool_frac + self.spawn_frac):
            try:
                obs = self._randomize_spawn()
                self.stats['spawned'] += 1
            except Exception as e:
                print(f'curriculum: spawn failed ({e}) -- DISABLING')
                self.enabled = False
        return obs

    # ---------- spawn randomization ----------
    def _observe_box(self):
        """ Learn the env's own object xy range from natural resets. """
        _, data = _mj(self.env)
        xy = np.array([data.qpos[a:a + 2] for a in self.free_adr])
        if self._lo is None:
            self._lo, self._hi = xy.copy(), xy.copy()
        else:
            self._lo = np.minimum(self._lo, xy)
            self._hi = np.maximum(self._hi, xy)
        self._resets_seen += 1

    def _randomize_spawn(self):
        _, data = _mj(self.env)
        qpos = np.array(data.qpos, copy=True)
        qvel = np.zeros_like(data.qvel)
        span = self._hi - self._lo
        for i, a in enumerate(self.free_adr):
            lo = self._lo[i] - 0.05 * span[i]
            hi = self._hi[i] + 0.05 * span[i]
            qpos[a:a + 2] = self.rng.uniform(lo, hi)
        return self._apply(qpos, qvel)

    def metrics(self):
        return {'pool_size': len(self.pool),
                'episodes_spawned': self.stats['spawned'],
                'episodes_restored': self.stats['restored']}