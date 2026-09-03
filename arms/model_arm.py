""" Base for every arm that trains a TD-MPC2 latent model next to QC-FQL.

    QC-FQL training is untouched; the model trains on the same replay
    windows and is consumed however the subclass decides (today: only the
    explore arm, which reads its dynamics ensemble). The model, its update
    and its diagnostics are defined here once so no arm can drift. """

from sac_chunked.experiment import Arm
from tdmpc.agent import TDMPC2Model
from tdmpc.diagnostics import model_report, print_wm_report


class ModelArm(Arm):
    name = 'model'
    arm_key = 'explore'

    @property
    def arm_cfg(self):
        return getattr(self.config, self.arm_key)

    def model_kwargs(self):
        return {}

    def build_model(self):
        self.model = TDMPC2Model(self.obs_dim, self.action_dim, self.device,
                                 self.config.tdmpc, self.gamma, **self.model_kwargs())

    def model_update(self, replay, metrics_on):
        t = self.config.tdmpc
        w = replay.sample_model_windows(t.batch_size, t.horizon, self.device, self.rng,
                                        online_frac=t.online_frac)
        if w is None:
            return {}
        return self.model.update(w['obs'], w['next_obs'], w['action'],
                                 w['reward'], w['mask'], w['valid'],
                                 metrics_on=metrics_on)

    def report(self, replay):
        t = self.config.tdmpc
        if t.diag_windows <= 0:
            return {}
        m = model_report(self.model, self.policy, replay, self.chunk_len,
                         t.diag_depth, self.gamma, self.device, self.rng,
                         num_windows=t.diag_windows)
        print_wm_report(m, t.diag_depth)
        return m
