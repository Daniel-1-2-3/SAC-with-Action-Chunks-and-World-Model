""" EXPLORE arm: critic best-of-N with an uncertainty-scaled novelty bonus.

    The CONTROL ranks candidates by the critic. This arm adds, per candidate,

        beta * s_i * nu_i

    s_i  = the critic's own uncertainty about candidate i (std across its
           ensemble heads, in Q units)
    nu_i = latent-model novelty of candidate i's imagined path, clamped to
           [0, 1] (dynamics-ensemble disagreement relative to the data's,
           Pathak et al. 2019)

    so novelty is measured in units of the critic's error bar and can never
    move a pick by more than beta of it. When the critic is sure (heads
    agree), the bonus is zero and this arm IS the control; when it is not
    (states it has never valued), novelty picks. No threshold, no schedule --
    the critic's convergence is the anneal. beta=0 is exactly the control.

    Comparison partner: the CONTROL arm (train_sac_chunked.py). The two
    differ only in beta. The latent model trains as in the ranking arm but is
    never used to score value -- only its dynamics ensemble is read. """

from arms.ranking import RankingArm
from wm.chunk_selector import ChunkSelector


class ExploreArm(RankingArm):
    name = 'explore'
    arm_key = 'explore'

    def model_kwargs(self):
        # The ensemble size is this arm's own knob; the shared tdmpc block
        # keeps num_dyn=1 for every other arm.
        return {'num_dyn': self.arm_cfg.num_dyn}

    def build_selector(self):
        return ChunkSelector(self.model, self.policy, self.action_dim,
                             self.chunk_len, self.chunk.select_n, self.gamma,
                             self.device, score_mode='critic',
                             rollout_chunks=self.arm_cfg.rollout_chunks,
                             bonus_beta=self.arm_cfg.beta)

    def log_extra(self):
        out = super().log_extra()
        # The novelty denominator. If it drifts up, "novel" is getting harder
        # to earn and nu collapses; if it drifts down, nu saturates.
        out['diagnosis/wm_data_disagreement'] = float(self.model.data_disagreement)
        return out

    def describe(self):
        c = self.arm_cfg
        return (f'{self.name}: critic best-of-{self.chunk.select_n} + '
                f'{c.num_dyn}-head novelty scaled by critic uncertainty, '
                f'beta {c.beta}')