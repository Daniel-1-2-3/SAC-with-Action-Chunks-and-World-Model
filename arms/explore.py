""" EXPLORE arm: critic best-of-N with an uncertainty-scaled novelty bonus.

    The CONTROL ranks candidates by the critic. This arm adds, per candidate
    (bonus_scale=spread, the default),

        beta * g * sigma_Q * s~_i * nu_i

    sigma_Q = spread of the critic's values across the n candidates
    s~_i    = the critic's relative doubt about candidate i (its ensemble
              std over the mean ensemble std, clamped to [0, 1])
    nu_i    = latent-model novelty of candidate i's imagined path, clamped
              to [0, nu_cap] (dynamics-ensemble disagreement relative to the
              data's, Pathak et al. 2019)
    g       = controller output in [0, 1]. explore.controller=gate: the
              learning-progress gate, 1 while real online returns are flat,
              -> 0 while they improve, so exploration switches itself off
              when plain learning is working. explore.controller=bandit: 0
              or 1 for a whole episode, chosen per episode by a
              sliding-window UCB over the two arms' returns

    so the bonus is bounded by beta * g * sigma_Q * nu_cap: novelty can
    reorder candidates the critic rates within about beta spreads of each
    other and never overrides a clear critic preference. beta=0 is exactly
    the control. bonus_scale=unc reproduces the earlier beta * s_i * nu_i.

    Comparison partner: the CONTROL arm (train_qcfql_bon.py). The two
    differ only in beta. The latent model (arms/model_arm.py) is never used
    to score value -- only its dynamics ensemble is read. """

from arms.model_arm import ModelArm
from wm.chunk_selector import ChunkSelector


class ExploreArm(ModelArm):
    name = 'explore'
    arm_key = 'explore'

    def model_kwargs(self):
        # The ensemble size is this arm's own knob; the shared tdmpc block
        # keeps num_dyn=1 for every other arm.
        c = self.arm_cfg
        return {'num_dyn': c.num_dyn,
                'novelty': 'reward' if c.novelty == 'reward' else 'mean',
                'novelty_at': c.novelty_at,
                'rollout_chunks': c.rollout_chunks,
                'chunk_len': self.chunk_len}

    def build_selector(self):
        return ChunkSelector(self.model, self.policy, self.action_dim,
                             self.chunk_len, self.chunk.select_n, self.gamma,
                             self.device,
                             rollout_chunks=self.arm_cfg.rollout_chunks,
                             bonus_beta=self.arm_cfg.beta,
                             novelty='none' if self.arm_cfg.novelty == 'none' else 'model',
                             novelty_at=self.arm_cfg.novelty_at,
                             nu_cap=self.arm_cfg.nu_cap,
                             bonus_scale=self.arm_cfg.bonus_scale,
                             progress_gate=self.arm_cfg.progress_gate,
                             progress_window=self.arm_cfg.progress_window,
                             progress_tau=self.arm_cfg.progress_tau,
                             use_rel_unc=self.arm_cfg.use_rel_unc,
                             controller=self.arm_cfg.controller,
                             bandit_window=self.arm_cfg.bandit_window,
                             bandit_c=self.arm_cfg.bandit_c)

    def log_extra(self):
        out = super().log_extra()
        # The novelty denominator. If it drifts up, "novel" is getting harder
        # to earn and nu collapses; if it drifts down, nu saturates.
        out['diagnosis/wm_data_disagreement'] = float(self.model.data_disagreement)
        return out

    def describe(self):
        c = self.arm_cfg
        if c.controller == 'bandit':
            gate = f'bandit controller W={c.bandit_window} c={c.bandit_c}'
        else:
            gate = (f'progress gate W={c.progress_window} tau={c.progress_tau}'
                    if c.progress_gate else 'no progress gate')
        t = self.config.tdmpc
        return (f'{self.name}: critic best-of-{self.chunk.select_n} + '
                f'{c.num_dyn}-head novelty[{c.novelty}@{c.novelty_at}, cap {c.nu_cap}, '
                f'ref {t.ref_mode}, shrink {t.reward_weight_shrink}, '
                f'model online_frac {t.online_frac}] '
                f'bonus_scale={c.bonus_scale}, rel_unc={c.use_rel_unc}, '
                f'beta {c.beta}, {gate}')
