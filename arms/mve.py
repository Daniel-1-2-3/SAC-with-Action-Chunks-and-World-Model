""" MVE arm: Model-based Value Expansion (Feinberg et al. 2018) for the
    QC-FQL critic.

    QC's critic target is

        R_real(chunk) + gamma^h * mask * Q_target(s', mu(s', z))

    where everything after the real chunk is a single bootstrap. MVE replaces
    that bootstrap with a short model rollout and a value at its end. Here
    the rollout is the QC actor's OWN next chunk imagined in latent space,
    optionally continued by the policy prior for `chunks - 1` more chunks,
    and the value at the end is the model's latent Q:

        V_mve(s') = sum_j gamma^{jh} R_model(chunk_j) + gamma^{Kh} Q_model(z_K, pi(z_K))
        target    = R_real + gamma^h * mask * [ (1 - w) Q_target(s', a') + w V_mve(s') ]

    `weight` w blends the two bootstraps; w = 1 is pure MVE, w = 0 is QC.
    `chunks` = 0 skips the rollout and uses the latent Q at s' directly.

    Acting is the CONTROL's: critic best-of-N with the same select_n. So the
    only difference from the control is the critic target -- this arm tests
    whether model-based targets speed up value learning, nothing else.

    What is NOT ported: MVE's TD-k trick trains the critic on imagined
    states too. The QC critic lives in observation space and this model does
    not decode, so imagined states cannot be fed to it. Every critic row is a
    real state; only its TARGET carries imagination. """

import torch

from arms.ranking import RankingArm
from wm.chunk_selector import ChunkSelector


class MVEArm(RankingArm):
    name = 'mve'
    arm_key = 'mve'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._tstats = {}

    def build_selector(self):
        return ChunkSelector(None, self.policy, self.action_dim, self.chunk_len,
                             self.chunk.select_n, self.gamma, self.device,
                             score_mode='critic')

    def describe(self):
        c = self.arm_cfg
        return (f'{self.name}: QC target blended with {c.chunks}-chunk latent '
                f'expansion, weight {c.weight}; acting = critic best-of-'
                f'{self.chunk.select_n}')

    @torch.no_grad()
    def critic_target(self, next_obs, reward, mask):
        c = self.arm_cfg
        # The SAME next chunk drives both bootstraps, so the blend compares
        # two estimates of the same quantity rather than two samples.
        next_chunk = self.policy.sample_chunk(next_obs)
        q_qc = self.policy._agg(self.policy.critic_target(next_obs, next_chunk))

        z = self.model.encode(next_obs)
        if c.chunks >= 1:
            actions = next_chunk.reshape(-1, self.chunk_len, self.action_dim)
            z, r_pool, disc, _ = self.model.rollout_chunk(z, actions)
            for _ in range(c.chunks - 1):
                z, r_j, disc, _ = self.model.rollout_pi_chunk(z, self.chunk_len, disc)
                r_pool = r_pool + r_j
            v_mve = r_pool + disc * self.model.terminal_value(z)
        else:
            v_mve = self.model.terminal_value(z)

        v = (1.0 - c.weight) * q_qc + c.weight * v_mve
        self._acc(q_qc, v_mve)
        return reward + self.gamma_h * mask * v

    def _acc(self, q_qc, v_mve):
        """ Is the expansion informative, or a noisier copy of the QC
            bootstrap? Logged under mve/. """
        a, b = q_qc.squeeze(-1), v_mve.squeeze(-1)
        s = self._tstats
        s['q_qc_mean'] = s.get('q_qc_mean', 0.0) + a.mean().item()
        s['v_mve_mean'] = s.get('v_mve_mean', 0.0) + b.mean().item()
        s['abs_gap'] = s.get('abs_gap', 0.0) + (a - b).abs().mean().item()
        if a.std() > 1e-6 and b.std() > 1e-6:
            corr = ((a - a.mean()) * (b - b.mean())).mean() / (a.std(correction=0) * b.std(correction=0))
            s['corr'] = s.get('corr', 0.0) + corr.item()
        s['_n'] = s.get('_n', 0) + 1

    def log_extra(self):
        out = super().log_extra()
        n = self._tstats.pop('_n', 0)
        if n:
            out.update({f'mve/{k}': v / n for k, v in self._tstats.items()})
        self._tstats = {}
        return out
