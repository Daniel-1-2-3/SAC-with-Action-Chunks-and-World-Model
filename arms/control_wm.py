""" CONTROL + TD-MPC2 latent model trained alongside it.

    Same QC-FQL training and critic best-of-N as the control
    (arms/control.py) -- the policy side is byte-identical. A TD-MPC2 latent
    model (tdmpc/agent.py) trains on replay in parallel. With
    tdmpc.q_mode=chunk its Q ensemble reads whole action chunks and
    regresses onto the same target family as the QC critic (pooled chunk
    reward + gamma^h * mask * bootstrap), on the SAME chunk transitions --
    a latent twin of the chunk critic.

    wm_control.score_source picks who scores the act-time candidates:
      critic  the QC critic (the control's behaviour). The model's chunk-Q
              scores are still computed and logged against it every decision
              (select/model_pick_agree, model_rank_corr,
              model_pick_regret_q) -- shadow mode, zero behavioural change.
      model   the model's chunk Q executes the argmax; the critic becomes
              the shadow. Requires tdmpc.q_mode=chunk. This is the "can the
              latent twin replace the critic at act time" run; everything
              else, including the policy's training, is unchanged.

    NOTE the model's updates draw from the same numpy rng as the policy's
    batches, so runs from this file are not bit-comparable to
    train_control.py runs at the same seed. Two runs of THIS file differing
    only in score_source are a matched pair. """

from arms.control import ControlArm
from tdmpc.agent import TDMPC2Model
from tdmpc.diagnostics import chunk_q_report, model_report
from wm.chunk_selector import ChunkSelector


class WMControlArm(ControlArm):
    name = 'control_wm'

    def build_model(self):
        self.model = TDMPC2Model(self.obs_dim, self.action_dim, self.device,
                                 self.config.tdmpc, self.gamma,
                                 chunk_len=self.chunk_len)
        return self.model

    def build_selector(self):
        return ChunkSelector(self.model, self.policy, self.action_dim,
                             self.chunk_len, self.chunk.select_n, self.gamma,
                             self.device,
                             candidate_source=self.chunk.candidate_source,
                             score_source=self.config.wm_control.score_source)

    def describe(self):
        return (f'{super().describe()} + tdmpc model '
                f'(q_mode={self.config.tdmpc.q_mode}, '
                f'scores: {self.config.wm_control.score_source})')

    def model_update(self, replay, metrics_on):
        t = self.config.tdmpc
        w = replay.sample_model_windows(t.batch_size, t.horizon, self.device,
                                        self.rng, online_frac=t.online_frac)
        if w is None:
            return {}
        chunk_batch = None
        if self.model.q_mode == 'chunk':
            chunk_batch = replay.sample_chunks(t.batch_size, self.device,
                                               self.rng, self.gamma)
            if chunk_batch is None:
                return {}
        return self.model.update(w['obs'], w['next_obs'], w['action'],
                                 w['reward'], w['mask'], w['valid'],
                                 chunk_batch=chunk_batch,
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
                                  select_n=self.chunk.select_n))
        return rep
