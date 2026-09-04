""" CONTROL arm -- the paper's method, "QC-FQL + critic best-of-N".

    QC-FQL training (Li et al. 2025, Alg. 2: flow BC policy, distilled
    one-step actor, chunk critic). At act and eval time the online critic
    picks the best of chunk.select_n chunks. select_n=1 is plain QC-FQL
    (train_qc_fql.py). No learned model anywhere. The base Arm class in
    sac_chunked/experiment.py is this arm; this file adds only
    chunk.candidate_source (actor | bc: where the candidates are drawn from;
    the critic picks either way). """

from sac_chunked.experiment import Arm
from wm.chunk_selector import ChunkSelector


class ControlArm(Arm):
    name = 'control'

    def build_selector(self):
        return ChunkSelector(None, self.policy, self.action_dim, self.chunk_len,
                             self.chunk.select_n, self.gamma, self.device,
                             candidate_source=self.chunk.candidate_source)

    def describe(self):
        return f'{super().describe()} (candidates: {self.chunk.candidate_source})'
