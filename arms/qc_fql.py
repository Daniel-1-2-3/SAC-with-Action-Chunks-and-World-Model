""" QC-FQL arm -- Li et al. 2025, Alg. 2, exactly: the distilled one-step
    actor's single output is executed at act and eval time. No best-of-N,
    no critic in the loop at act time. This is the control with
    chunk.select_n forced to 1; it exists as its own script so the baseline
    row in the paper is a named, un-flagged run.

    The paper's alpha for cube-triple QC-FQL is 100 (Table 3); pass
    --chunk.alpha=100 to match it (the config default is 300). """

from sac_chunked.experiment import Arm


class QCFQLArm(Arm):
    name = 'qc_fql'

    def __init__(self, config, obs_dim, action_dim, device, rng):
        # elements.Config is immutable; derive a copy with select_n=1.
        config = config.update({'chunk.select_n': 1})
        super().__init__(config, obs_dim, action_dim, device, rng)

    def describe(self):
        return f'{self.name}: QC-FQL, single actor sample (alpha {self.chunk.alpha})'
