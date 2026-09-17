"""Best-of-N chunk selection with the wm_explore novelty bonus -- port of
this repo's wm/chunk_selector.py, restricted to the paths the wm_explore
arm runs. The scoring is jitted; the per-episode controller (UCB bandit /
learning-progress gate) is plain host-side Python, exactly as in the
PyTorch version.

Score of candidate i (bonus_scale 'spread', the v5 arm):

    score_i = Q_i + beta * g * sigma_Q * s~_i * nu_i

    Q_i      critic's value (q_agg over the ensemble, from the agent)
    sigma_Q  population std of Q_i across the n candidates
    s~_i     relative critic doubt clamp(s_i / mean s, 0, 1), or 1 with
             use_rel_unc off
    g        controller output in {0, 1} (bandit) or [0, 1] (gate)
    nu_i     clamp(path disagreement / data reference - 1, 0, nu_cap),
             or 1 with novelty 'none'

'unc' (v3): score_i = Q_i + beta * s_i * nu_i.

Eval and g = 0 score by the critic alone -- identical to the plain
combined arm's sample_actions argmax over the same candidates.

NOT PORTED (diagnostics only): the stat accumulators beyond the basics,
episode-start value capture, report_episode_discounted.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


@partial(jax.jit, static_argnames=('bonus_scale', 'novelty', 'use_rel_unc'))
def _score(agent, wm, ob, rng, gate, bonus_scale, novelty, use_rel_unc):
    """(candidates (n, D), score (n,), critic_score (n,), nu (n,))."""
    rng, cand_key, path_key = jax.random.split(rng, 3)
    cands, qs = agent.sample_candidates(ob, rng=cand_key)          # (n,D), (E,n)
    if agent.config['q_agg'] == 'mean':
        critic_score = qs.mean(axis=0)
    else:
        critic_score = qs.min(axis=0)
    n, chunk_dim = cands.shape

    if novelty == 'none':
        nu = jnp.ones(n)
    else:
        acts = cands.reshape(n, wm.chunk_len, wm.action_dim)
        z = wm.encode(jnp.repeat(ob[None, :], n, axis=0))
        d = wm.path_disagreement(z, acts, path_key)                # (n,)
        ratio = d / jnp.maximum(wm.data_disagreement, 1e-10)
        nu = jnp.clip(ratio - 1.0, 0.0, wm.cfg['nu_cap'])

    s_unc = qs.std(axis=0)                                         # (n,)
    if bonus_scale == 'unc':
        scale = wm.cfg['beta'] * s_unc
    else:
        sigma_q = critic_score.std()
        if use_rel_unc:
            s_rel = jnp.clip(s_unc / (s_unc.mean() + 1e-12), 0.0, 1.0)
        else:
            s_rel = jnp.ones(n)
        scale = wm.cfg['beta'] * gate * sigma_q * s_rel
    score = critic_score + scale * nu
    return cands, score, critic_score, nu


class NoveltySelector:
    """Holds the host-side controller state and the episode hooks the loop
    calls (begin_episode, report_episode_return), mirroring
    wm/chunk_selector.py. select() is one jitted scoring call + argmax."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.bandit_arm = 1          # 0 exploit (g=0), 1 explore (g=1)
        self._pulls = []             # (arm, return) of finished REAL episodes
        self._returns = []
        self._g = 1.0                # gate mode: EMA-smoothed value
        self.last = {}               # small log dict, refreshed per select

    @property
    def gate(self):
        if self.cfg['controller'] == 'bandit':
            return float(self.bandit_arm)
        return self._g if self.cfg['progress_gate'] else 1.0

    # ------------------------------------------------------------- hooks

    def begin_episode(self):
        """Bandit controller: pick this episode's arm by sliding-window UCB
        over real episode returns (wm/chunk_selector.py begin_episode)."""
        if self.cfg['controller'] != 'bandit':
            return
        window = self._pulls[-int(self.cfg['bandit_window']):]
        n_total = len(window)
        counts = {k: sum(1 for a, _ in window if a == k) for k in (0, 1)}
        means = {k: float(np.mean([r for a, r in window if a == k]))
                 for k in (0, 1) if counts[k] > 0}
        untried = [k for k in (0, 1) if counts[k] == 0]
        if untried:
            self.bandit_arm = untried[0]
            return
        rets = np.asarray([r for _, r in window])
        rng_ = max(float(rets.max() - rets.min()), 1e-8)
        c = float(self.cfg['bandit_c'])
        score = {k: means[k] / rng_ + c * np.sqrt(2.0 * np.log(n_total) / counts[k])
                 for k in (0, 1)}
        self.bandit_arm = 1 if score[1] >= score[0] else 0

    def report_episode_return(self, ep_return):
        """Finished REAL online episode (never eval). Feeds the bandit, or
        the learning-progress gate (wm/chunk_selector.py, same formula)."""
        self._returns.append(float(ep_return))
        if self.cfg['controller'] == 'bandit':
            self._pulls.append((self.bandit_arm, float(ep_return)))
            return
        if not self.cfg['progress_gate']:
            return
        W = int(self.cfg['progress_window'])
        if len(self._returns) < 2 * W:
            return
        recent = np.asarray(self._returns[-W:])
        older = np.asarray(self._returns[-2 * W:-W])
        frac = float((recent > np.median(older)).mean())
        g = 1.0 - 2.0 * max(frac - 0.5, 0.0)
        tau = float(self.cfg['progress_tau'])
        self._g = (1.0 - tau) * self._g + tau * g

    # ------------------------------------------------------------ select

    def select(self, agent, wm, ob, rng, eval_mode=False):
        """One decision: returns the chosen flattened chunk (np, (D,)).
        eval_mode or gate 0 scores by the critic alone (= combined arm)."""
        gate = 0.0 if eval_mode else self.gate
        cands, score, critic_score, nu = _score(
            agent, wm, jnp.asarray(ob), rng, jnp.asarray(gate, jnp.float32),
            str(self.cfg['bonus_scale']), str(self.cfg['novelty']),
            bool(self.cfg['use_rel_unc']))
        idx = int(jnp.argmax(score))
        if not eval_mode:
            self.last = {
                'select/bandit_arm': float(self.bandit_arm),
                'select/gate': float(gate),
                'select/nu_mean': float(nu.mean()),
                'select/nu_pick': float(nu[idx]),
                'select/score_std': float(critic_score.std()),
                'select/pick_moved': float(idx != int(jnp.argmax(critic_score))),
            }
        return np.asarray(cands[idx])
