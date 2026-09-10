""" WM potential-based reward shaping -- one mixin, three arms.

    The base arm (control / QC / QC-FQL) is unchanged in training, selection
    and evaluation. A TD-MPC2 latent model (tdmpc/agent.py, forced to
    q_mode='chunk') trains alongside on replay, and its value becomes a
    potential phi over states:

        phi(s) = wm chunk-Q, TARGET heads, at (z(s), policy's chunk at s)

    Every critic update, the reward term of the target gains

        bonus = coef * (gamma^h * mask * (phi(s') - c) - (phi(s) - c))

    where c is a slow running mean of phi (shaping.center) -- see
    shape_reward for why the centering is load-bearing.

    via Arm.shape_reward (sac_chunked/experiment.py). Nothing else sees the
    bonus: the replay buffer stores raw rewards, the actor losses and every
    sac/ and diagnosis/ reward metric read raw rewards, eval scores raw
    returns, and best-of-N is invariant (all candidates at s share phi(s)).
    Along a trajectory the bonus telescopes, so it changes how fast value
    propagates, not which policy is best. A CONSTANT error in phi cancels in
    the difference -- the measured -0.5 wm level bias is inert here by
    construction.

    shaping.potential=none is the matched control: the exact base arm, with
    the model still training (identical rng consumption for a one-flag
    pair).

    On QC the policy's chunk is best-of-N itself (QCAgent.sample_chunk), so
    phi costs two best-of-N calls per critic batch -- the same operation
    QC's own target already performs once. """

import torch

from arms.control import ControlArm
from arms.qc_arm import QCArm
from arms.qc_fql import QCFQLArm
from tdmpc.agent import TDMPC2Model
from tdmpc.diagnostics import chunk_q_report, model_report


class WMShapingMixin:
    """ Adds the model, its training, the phi bonus and its diagnostics to
        any Arm subclass. Placed FIRST in the bases so its overrides win;
        describe/log_extra chain to the arm's own via super(). """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        s = self.config.shaping
        assert s.potential in ('wm', 'none'), s.potential
        assert s.chunk_bootstrap in ('actor', 'prior'), s.chunk_bootstrap
        self._shape_stats = {}
        # Slow running mean of phi, for centering. None until the first
        # batch, which initializes it directly (no warmup from 0 -- that
        # would replay a shrinking copy of the constant this removes).
        self._phi_center = None

    def build_model(self):
        self.model = TDMPC2Model(self.obs_dim, self.action_dim, self.device,
                                 self.config.tdmpc, self.gamma,
                                 chunk_len=self.chunk_len, q_mode='chunk')
        return self.model

    def describe(self):
        s = self.config.shaping
        return (f'{super().describe()} + wm shaping (potential: {s.potential}, '
                f'coef: {s.coef})')

    @torch.no_grad()
    def phi(self, obs):
        """ The potential: the wm's TARGET-head chunk value of the policy's
            own chunk at obs, (B, 1). Target heads move slowly (tdmpc.tau),
            keeping the potential quasi-static between updates. """
        return self.model.chunk_q(self.model.encode(obs),
                                  self.policy.sample_chunk(obs), target=True)

    def shape_reward(self, obs, next_obs, reward, mask, metrics_on=False):
        """ bonus = coef * (discount^h * mask * centered phi(next state)
                           - centered phi(start state))

            Centering (shaping.center) subtracts a slow running mean of phi
            from both terms. Without it, phi's LEVEL (~-290 on this task)
            leaks through the discount as a constant

                coef * (discount^h - 1) * level  ~  +14 per chunk

            which compounds through the critic's bootstrap by
            1 / (1 - discount^h) ~ 20.5x into a ~+290 relocation of every
            critic value, chasing phi's drifting level all run -- the
            t4_shaped_s2 failure. Subtracting a constant from a potential is
            still a potential, so policy invariance is untouched; centered,
            a typical state reads ~0 and the bonus carries only the climb. """
        s = self.config.shaping
        if s.potential != 'wm':
            return reward
        with torch.no_grad():
            phi_obs = self.phi(obs)
            phi_next = self.phi(next_obs)
            if s.center:
                m = 0.5 * (phi_obs.mean() + phi_next.mean())
                if self._phi_center is None:
                    self._phi_center = m
                else:
                    self._phi_center = ((1.0 - s.center_tau) * self._phi_center
                                        + s.center_tau * m)
                phi_obs = phi_obs - self._phi_center
                phi_next = phi_next - self._phi_center
            bonus = s.coef * (self.gamma_h * mask * phi_next - phi_obs)
            if metrics_on:
                self._shape_stats = {
                    'diagnosis/shaping_bonus_mean': float(bonus.mean()),
                    'diagnosis/shaping_bonus_abs': float(bonus.abs().mean()),
                    'diagnosis/shaping_bonus_std': float(bonus.std()),
                }
                if self._phi_center is not None:
                    self._shape_stats['diagnosis/shaping_phi_center'] = \
                        float(self._phi_center)
            return reward + bonus

    def log_extra(self):
        out = super().log_extra()
        out.update(self._shape_stats)
        return out

    def model_update(self, replay, metrics_on):
        t = self.config.tdmpc
        w = replay.sample_model_windows(t.batch_size, t.horizon, self.device,
                                        self.rng, online_frac=t.online_frac)
        if w is None:
            return {}
        chunk_batch = replay.sample_chunks(t.batch_size, self.device, self.rng,
                                           self.gamma)
        if chunk_batch is None:
            return {}
        next_chunk = None
        if self.config.shaping.chunk_bootstrap == 'actor':
            with torch.no_grad():
                next_chunk = self.policy.sample_chunk(chunk_batch[6])
        return self.model.update(w['obs'], w['next_obs'], w['action'],
                                 w['reward'], w['mask'], w['valid'],
                                 chunk_batch=chunk_batch, next_chunk=next_chunk,
                                 metrics_on=metrics_on)

    def report(self, replay):
        t = self.config.tdmpc
        if t.diag_windows <= 0:
            return {}
        rep = model_report(self.model, self.policy, replay, self.chunk_len,
                           t.diag_depth, self.gamma, self.device, self.rng,
                           num_windows=t.diag_windows)
        rep.update(chunk_q_report(self.model, self.policy, replay,
                                  self.chunk_len, self.device, self.rng,
                                  num_windows=t.diag_windows,
                                  hit_thresh=t.diag_hit_thresh))
        return rep


class ShapedControlArm(WMShapingMixin, ControlArm):
    name = 'control_shaped'


class ShapedQCArm(WMShapingMixin, QCArm):
    name = 'qc_shaped'


class ShapedQCFQLArm(WMShapingMixin, QCFQLArm):
    name = 'qc_fql_shaped'