""" CONTROL arm: QC-FQL (Li et al. 2025, Alg. 2) with the online critic
    picking the best of select_n chunks sampled from the distilled one-step
    actor, at act and eval time. No learned model anywhere. select_n=1 is
    plain QC-FQL. The base Arm class in sac_chunked/experiment.py is this
    arm; this file adds only chunk.candidate_source (actor | bc: where the
    candidates are drawn from; the critic picks either way). """

from sac_chunked.experiment import Arm
from wm.chunk_selector import ChunkSelector


class QCFQLBestOfNArm(Arm):
    name = 'qcfql_bon'

    def build_selector(self):
        return ChunkSelector(None, self.policy, self.action_dim, self.chunk_len,
                             self.chunk.select_n, self.gamma, self.device,
                             candidate_source=self.chunk.candidate_source)

    def describe(self):
        return f'{super().describe()} (candidates: {self.chunk.candidate_source})'
