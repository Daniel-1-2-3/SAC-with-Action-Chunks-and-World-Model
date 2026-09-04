""" QC arm: Li et al. 2025 Algorithm 1, best-of-N from the flow BC policy
    under the critic, at act time, eval time and inside the TD target. The
    port lives in sac_chunked/qc_agent.py; this arm only wires it into the
    shared loop.

    QC's best-of-N is INSIDE the agent (its policy is the argmax), so the
    selector is disabled: n=1 passthrough, select() is exactly policy.act.
    chunk.select_n and chunk.alpha are ignored; chunk.qc_num_samples is N.

    Comparison: train_control.py (QC-FQL with critic best-of-N over the
    distilled actor) differs in what is sampled (one-step actor vs. the
    flow) and in the actor loss (distillation + Q term vs. BC only). """

import torch

from sac_chunked.experiment import Arm
from sac_chunked.qc_agent import QCAgent
from wm.chunk_selector import ChunkSelector


class QCArm(Arm):
    name = 'qc'

    def build_policy(self):
        c = self.chunk
        return QCAgent(
            repr_dim=self.obs_dim, action_dim=self.action_dim, chunk_len=self.chunk_len,
            device=self.device, lr=c.lr, hidden_dim=c.hidden_dim,
            num_layers=c.num_layers, critic_target_tau=c.critic_target_tau,
            ensemble=c.ensemble, num_samples=c.qc_num_samples,
            flow_steps=c.flow_steps, q_agg=c.q_agg, compile_nets=c.compile_nets,
            target_mode=c.qc_target)

    def build_selector(self):
        # Disabled on purpose: best-of-N is the agent's own policy.
        return ChunkSelector(None, self.policy, self.action_dim, self.chunk_len,
                             1, self.gamma, self.device)

    def describe(self):
        return (f'{self.name}: QC Alg. 1, best-of-{self.chunk.qc_num_samples} from '
                f'the flow BC policy (chunk.select_n and chunk.alpha ignored)')

    @torch.no_grad()
    def critic_target(self, next_obs, reward, mask, metrics_on=False):
        """ Their eq. 11: R_h + gamma^h * mask * mean_k Q_target_k(s', a*),
            a* the agent's own best-of-N at s' (QCAgent.chunk_target_values). """
        return reward + self.gamma_h * mask * self.policy.chunk_target_values(next_obs)
