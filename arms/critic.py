""" CONTROL arm: QC-FQL with the critic ranking select_n candidate chunks.
    This is QC's own best-of-N, no learned model anywhere. select_n=1 is
    plain QC-FQL. The base Arm class in sac_chunked/experiment.py is this
    arm; this file exists so every train script imports an arm the same
    way. """

from sac_chunked.experiment import Arm


class CriticArm(Arm):
    name = 'critic_best_of_n'
