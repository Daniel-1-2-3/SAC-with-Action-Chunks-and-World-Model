""" MVE arm -- model value expansion, value-only.

    Everything about the control (arms/control.py) is unchanged: QC-FQL
    training, critic best-of-N selection at act and eval time, the same
    actor, the same replay. The ONE difference is where the critic's TD
    target gets its continuation value.

      control  R_chunk + gamma^h * mask * Q_critic_target(s', actor chunk)
      mve      R_chunk + gamma^h * mask * Q_wm_target(z', actor chunk)

    R_chunk is the REAL pooled discounted reward of the replayed chunk and
    s' is the REAL observation h steps later; only the value function
    differs. The dynamics network is never called in the target, so no
    model prediction error enters it -- unlike the earlier MVE arms, which
    imagined the rollout and capped the imagined end state with the critic.

    The world model (tdmpc/agent.py, forced to q_mode='chunk') trains
    alongside on replay: consistency and reward on windows, and its own
    chunk Q on the SAME chunk transitions the critic uses, bootstrapping
    from the actor's chunk at s' (mve.chunk_bootstrap; 'prior' restores
    TD-MPC2's own convention, which has nothing anchoring it to an offline
    buffer).

    mve.critic_target_source=critic makes this arm the control exactly --
    same script, same rng consumption, model still training and logging --
    i.e. the matched partner for an attribution pair. """

import torch

from arms.control import ControlArm
from tdmpc.agent import TDMPC2Model
from tdmpc.diagnostics import chunk_q_report, model_report


def _corr_t(a, b):
    """ Pearson correlation of two (B, 1) tensors, as a float. """
    a, b = a.reshape(-1).float(), b.reshape(-1).float()
    ac, bc = a - a.mean(), b - b.mean()
    return float((ac * bc).mean()
                 / (ac.std(correction=0) * bc.std(correction=0) + 1e-8))


class MVEArm(ControlArm):
    name = 'mve'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.source = self.config.mve.critic_target_source
        assert self.source in ('critic', 'wm'), self.source
        assert self.config.mve.chunk_bootstrap in ('actor', 'prior')
        self._target_stats = {}

    def build_model(self):
        # q_mode is forced here rather than read from the shared tdmpc block:
        # this arm cannot work with the reference's per-step Q, and the
        # block stays at its TD-MPC2 default for every other consumer.
        self.model = TDMPC2Model(self.obs_dim, self.action_dim, self.device,
                                 self.config.tdmpc, self.gamma,
                                 chunk_len=self.chunk_len, q_mode='chunk')
        return self.model

    def describe(self):
        return (f'{super().describe()} | critic target: {self.source}'
                f' (wm bootstrap: {self.config.mve.chunk_bootstrap})')

    def critic_target(self, next_obs, reward, mask, metrics_on=False):
        """ See the module docstring. source='critic' defers to QC eq. 15
            unchanged; 'wm' swaps only the continuation value. """
        if self.source == 'critic':
            return super().critic_target(next_obs, reward, mask,
                                         metrics_on=metrics_on)
        with torch.no_grad():
            next_chunk = self.policy.sample_chunk(next_obs)
            wm_v = self.model.chunk_q(self.model.encode(next_obs), next_chunk,
                                      target=True)
            target = reward + self.gamma_h * mask * wm_v
            if metrics_on:
                own = self.policy._agg(
                    self.policy.critic_target(next_obs, next_chunk))
                self._target_stats = {
                    # What swapping the value function did to the target.
                    'diagnosis/wm_boot_minus_critic_boot': float((wm_v - own).mean()),
                    'diagnosis/wm_critic_boot_corr': _corr_t(wm_v, own),
                    'diagnosis/wm_boot_mean': float(wm_v.mean()),
                    'diagnosis/critic_boot_mean': float(own.mean()),
                    'diagnosis/wm_boot_std': float(wm_v.std()),
                    'diagnosis/critic_boot_std': float(own.std()),
                }
        return target

    def log_extra(self):
        out = super().log_extra()
        out.update(self._target_stats)
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
        if self.config.mve.chunk_bootstrap == 'actor':
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
