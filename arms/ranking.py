""" RANKING arm: TD-MPC2-scored best-of-N.

    QC-FQL training is untouched. At every chunk boundary the latent model
    imagines each of the select_n candidate chunks and the argmax of

        sum_t gamma^t * r_model(z_t, a_t) + gamma^H * Q_model(z_H, pi(z_H))

    is executed. Its comparison partner is the control (critic best-of-N with
    the same select_n): the two differ only in what ranks the candidates.

    Every other model-based arm subclasses this one, so the model, its
    training and its diagnostics are defined here once. """

import torch

from sac_chunked.experiment import Arm
from tdmpc.agent import TDMPC2Model
from tdmpc.diagnostics import model_report, print_wm_report
from wm.chunk_selector import ChunkSelector


class RankingArm(Arm):
    name = 'ranking'
    arm_key = 'ranking'

    @property
    def arm_cfg(self):
        return getattr(self.config, self.arm_key)

    def model_kwargs(self):
        return {}

    def build_model(self):
        self.model = TDMPC2Model(self.obs_dim, self.action_dim, self.device,
                                 self.config.tdmpc, self.gamma, **self.model_kwargs())

    def build_selector(self):
        return ChunkSelector(self.model, self.policy, self.action_dim,
                             self.chunk_len, self.chunk.select_n, self.gamma,
                             self.device, score_mode='model',
                             rollout_chunks=self.arm_cfg.rollout_chunks)

    def describe(self):
        n, k = self.chunk.select_n, self.arm_cfg.rollout_chunks
        return (f'{self.name}: TD-MPC2 best-of-{n}, {k} chunk(s) = '
                f'{k * self.chunk_len} latent steps, no decode')

    def model_update(self, replay, metrics_on):
        t = self.config.tdmpc
        w = replay.sample_model_windows(t.batch_size, t.horizon, self.device, self.rng)
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
