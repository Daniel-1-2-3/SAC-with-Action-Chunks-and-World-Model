""" CONTROL arm: QC-FQL (Li et al. 2025, Alg. 2) with the online critic
    picking the best of select_n chunks sampled from the distilled one-step
    actor, at act and eval time. No learned model anywhere. select_n=1 is
    plain QC-FQL. The base Arm class in sac_chunked/experiment.py is this
    arm; this file exists so every train script imports an arm the same
    way. """

from sac_chunked.experiment import Arm


class QCFQLBestOfNArm(Arm):
    name = 'qcfql_bon'
