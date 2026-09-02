""" EXPLORE arm: ranking + a disagreement bonus (Pathak et al. 2019,
    "Self-Supervised Exploration via Disagreement").

    The latent model carries an ENSEMBLE of dynamics heads (num_dyn). The
    rollout runs through their mean; their variance along the candidate's
    imagined path is the bonus:

        score = sum_t gamma^t r_model + gamma^H Q_model + beta * disagreement / scale

    scale is a running mean of the disagreement over decisions, so beta is
    in reward units: beta = 1 adds one reward unit per typical disagreement.

    This is optimism in the face of model uncertainty applied to chunk
    selection: among candidates the model rates similarly, prefer the one
    whose outcome the model is least sure of. Pathak's differentiable policy
    update is not needed -- best-of-N already optimises the score directly
    over candidates, and QC-FQL training stays untouched.

    Comparison partner: the RANKING arm. The two differ only in beta. """

from arms.ranking import RankingArm
from wm.chunk_selector import ChunkSelector


class ExploreArm(RankingArm):
    name = 'explore'
    arm_key = 'explore'

    def build_model(self):
        # The ensemble size is an arm parameter, injected into the model
        # config for this arm only.
        self.config = self.config.update({'tdmpc': {'num_dyn': self.arm_cfg.num_dyn}})
        super().build_model()

    def build_selector(self):
        return ChunkSelector(self.model, self.policy, self.action_dim,
                             self.chunk_len, self.chunk.select_n, self.gamma,
                             self.device, score_mode='model',
                             rollout_chunks=self.arm_cfg.rollout_chunks,
                             bonus_beta=self.arm_cfg.beta)

    def describe(self):
        c = self.arm_cfg
        return (f'{self.name}: TD-MPC2 best-of-{self.chunk.select_n} + '
                f'{c.num_dyn}-head disagreement bonus, beta {c.beta}')
