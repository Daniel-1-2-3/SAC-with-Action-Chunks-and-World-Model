""" OPTIMISTIC arm: ranking with an Optimistic World Model (Mete et al.
    2026, RBMLE for deep MBRL).

    The latent model's training gets one extra loss, eq. 10 of the paper,
    applied to the SimNorm latent read as G categoricals:

        L_opt = -alpha * sum_l A_l log p(z_{l+1} | z_l, a_l) - eta * sum_l H(p)

    on imagined trajectories rolled by the policy prior from real encoded
    states. Transitions whose imagined outcome beat the value's expectation
    get their likelihood raised, so the model's imagination drifts toward
    outcomes that are better than its data says -- which is exactly the bias
    RBMLE argues an MLE model needs to escape the closed-loop identification
    trap. The ranking selector then prefers chunks the optimistic model
    likes. alpha must be tiny (paper: 1e-4; 0.1 collapses learning).

    Everything else is the RANKING arm, so the two differ only in the
    optimistic loss. """

from arms.ranking import RankingArm


class OptimisticArm(RankingArm):
    name = 'optimistic'
    arm_key = 'optimistic'

    def model_kwargs(self):
        return {'optimism': self.arm_cfg}

    def describe(self):
        c = self.arm_cfg
        return (f'{self.name}: TD-MPC2 best-of-{self.chunk.select_n} with '
                f'optimistic dynamics loss alpha {c.alpha}, eta {c.eta}')
