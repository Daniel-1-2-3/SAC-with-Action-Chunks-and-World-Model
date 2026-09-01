import jax
import numpy as np
import torch

from sac_chunked.chunk_utils import pool_chunk
from helpers.interop import jax_to_torch

def decode_obs(bridge, carry, obs_key, device):
    """ Latent -> predicted observation, as a torch tensor.

        The actor and critic live in observation space, so every point where
        an imagined latent has to talk to them goes through here. The result
        is a PREDICTION, not a real state: it carries decoder error on top of
        whatever dynamics error accumulated to reach this latent. wm_report
        measures both against real replay states. """
    decoded = bridge.decode_state(carry)[obs_key]
    return torch.as_tensor(np.asarray(decoded, dtype=np.float32), device=device)

class ChunkSelector:
    """ Model-scored best-of-N chunk selection. The ONLY place the world model
        influences behavior.

        At every chunk boundary: sample n candidate chunks from the one-step
        policy at the raw observation, imagine each through the world model
        from the current posterior latent, score it as

            pooled imagined reward + gamma^h * cont * Q(decoded end state)

        and execute the argmax. Training never sees any of this -- the critic
        and actor update on real replay chunks with the plain QC-FQL target.

        Failure containment: candidates are i.i.d. samples from the policy, so
        if the scores are uninformative noise the argmax is distributed like a
        single policy sample and the agent degrades to QC-FQL rather than
        below it. The exception is a model that SYSTEMATICALLY prefers chunks
        it predicts well (e.g. do-nothing chunks); watch select/score_gap
        together with eval/coherence for that.

        n <= 1 disables everything here: observe/record_action are no-ops and
        select() is exactly policy.act. That is the QC-FQL control run.

        The posterior latent is maintained by one encode_step per env step
        (observe). That is scorer plumbing, not a policy input -- the policy
        reads the raw observation. A history-less single-step encode is not a
        substitute: the RSSM's deter state starts empty and the rollout from
        it is meaningless.

        Attribution: every decision also scores the same candidates with the
        online critic alone, Q(s, chunk) -- which is the QC paper's own
        best-of-N method. select/pick_agreement near 1.0 means the model adds
        nothing beyond the critic and this run reduces to QC. """

    def __init__(self, bridge, policy, action_dim, chunk_len, n, gamma, device,
                 obs_key='state', reward_shift=0.0, score_mode='model',
                 rollout_chunks=1):
        """ score_mode 'model': the docstring above (needs a bridge).
            score_mode 'critic': the SAME best-of-N mechanics, but the score
            is Q(s, chunk) from the online critic -- QC's own best-of-N. No
            bridge, no encoding, no imagination; the world model is not
            involved at all. This is the critic-selection control arm.
            rollout_chunks (model mode): imagine this many chunks ahead. The
            first is the candidate; each further chunk is sampled from the
            policy at the decoded intermediate state. Lookahead depth is
            rollout_chunks * chunk_len steps; score is the discounted pooled
            imagined reward over all of it plus the discounted terminal
            Q(decoded end state). """
        assert score_mode in ('model', 'critic'), score_mode
        self.score_mode = score_mode
        self.rollout_chunks = max(1, int(rollout_chunks))
        self.bridge = bridge
        self.policy = policy
        self.action_dim = action_dim
        self.chunk_len = chunk_len
        self.n = int(n)
        self.enabled = self.n > 1
        self.gamma = gamma
        self.gamma_h = gamma ** chunk_len
        self.device = device
        self.obs_key = obs_key
        self.reward_shift = reward_shift
        self._stats = {}
        self.reset()

    def reset(self):
        """ Call at every episode start (env.reset). """
        if not self.enabled or self.score_mode == 'critic':
            return
        self.enc, self.dyn = self.bridge.init_encode(1)
        self.prevact = np.zeros((1, self.action_dim), dtype=np.float32)
        self.is_first = np.array([True])

    def observe(self, state_1d):
        """ Call once per env step with the CURRENT raw observation, before
            acting. Filters the posterior so select() imagines from an
            up-to-date latent. Encoding continues on every step of an
            executing chunk -- committing to actions is not a reason to stop
            looking. """
        if not self.enabled or self.score_mode == 'critic':
            return
        s = np.asarray(state_1d, dtype=np.float32).reshape(1, -1)
        self.enc, self.dyn, _ = self.bridge.encode_step(
            self.enc, self.dyn, s, self.prevact, self.is_first)
        self.is_first = np.array([False])

    def record_action(self, action_1d):
        """ Call after env.step with the action that was EXECUTED, so the next
            observe() conditions on it. """
        if not self.enabled or self.score_mode == 'critic':
            return
        self.prevact = np.asarray(action_1d, dtype=np.float32).reshape(1, -1)

    def _acc(self, key, value):
        s, c = self._stats.get(key, (0.0, 0))
        self._stats[key] = (s + float(value), c + 1)

    def pop_stats(self):
        """ Means since the last pop, prefixed select/. Empty when disabled or
            no decisions happened. """
        out = {f'select/{k}': s / c for k, (s, c) in self._stats.items() if c > 0}
        self._stats = {}
        return out

    @torch.no_grad()
    def select(self, state_1d):
        """ state_1d: (obs_dim,) raw observation, the same one observe() just
            saw. Returns (chunk_len, action_dim). """
        if not self.enabled:
            return self.policy.act(np.asarray(state_1d, dtype=np.float32),
                                   eval_mode=False)

        feat = torch.as_tensor(np.asarray(state_1d, dtype=np.float32),
                               device=self.device).reshape(1, -1)
        feat_n = feat.repeat(self.n, 1)
        cands = self.policy.sample_chunk(feat_n) # (n, chunk_len * action_dim)

        # Critic score of the SAME candidates -- QC's own best-of-N ranking.
        # In critic mode it picks; in model mode it is logged for attribution.
        critic_score = self.policy._agg(
            self.policy.critic(feat_n, cands)).squeeze(-1) # (n,)

        if self.score_mode == 'critic':
            idx = int(torch.argmax(critic_score).item())
            self._acc('score_gap',
                      (critic_score[idx] - critic_score.mean()).item())
            self._acc('score_std', critic_score.std().item())
            return cands[idx].detach().cpu().numpy().reshape(
                self.chunk_len, self.action_dim)

        carry_h = {k: np.repeat(np.asarray(jax.device_get(v)), self.n, axis=0)
                   for k, v in self.dyn.items()}
        carry = self.bridge.place_seed(carry_h)
        # Discounted pooled reward over rollout_chunks imagined chunks:
        #   score = sum_j disc_j * pooled_r_j + disc_end * Q(decoded end)
        # with disc advancing by gamma^chunk_len * pooled_cont per chunk, so
        # imagined termination cuts everything after it.
        score = torch.zeros(self.n, 1, device=self.device)
        disc = torch.ones(self.n, 1, device=self.device)
        chunk_np = cands.detach().cpu().numpy()
        for j in range(self.rollout_chunks):
            if j > 0:
                mid_obs = decode_obs(self.bridge, carry, self.obs_key,
                                     self.device)
                chunk_np = self.policy.sample_chunk(
                    mid_obs).detach().cpu().numpy()
            carry, _, reward_j, cont_j = self.bridge.img_chunk(
                carry, chunk_np, self.chunk_len)
            # (n, chunk_len) -> (chunk_len, n, 1), step-major (pool_chunk).
            r = jax_to_torch(reward_j, self.device).transpose(0, 1) \
                .unsqueeze(-1) + self.reward_shift
            c = jax_to_torch(cont_j, self.device).transpose(0, 1).unsqueeze(-1)
            pooled_r, pooled_c = pool_chunk(r, c, self.gamma)
            score = score + disc * pooled_r
            disc = disc * self.gamma_h * pooled_c

        end_obs = decode_obs(self.bridge, carry, self.obs_key, self.device)
        end_v = self.policy.chunk_target_values(end_obs)
        score = (score + disc * end_v).squeeze(-1) # (n,)

        idx = int(torch.argmax(score).item())
        self._acc('score_gap', (score[idx] - score.mean()).item())
        self._acc('score_std', score.std().item())
        self._acc('end_v_std', end_v.squeeze(-1).std().item())
        self._acc('pick_agreement',
                  float(idx == int(torch.argmax(critic_score).item())))
        # Pearson over the n candidates. correction=0 (population std)
        # because the products are averaged over n -- the default sample std
        # would scale the result by (n-1)/n.
        s_std = score.std(correction=0)
        c_std = critic_score.std(correction=0)
        if s_std.item() > 1e-8 and c_std.item() > 1e-8:
            sn = (score - score.mean()) / s_std
            cn = (critic_score - critic_score.mean()) / c_std
            self._acc('model_critic_corr', (sn * cn).mean().item())

        return cands[idx].detach().cpu().numpy().reshape(
            self.chunk_len, self.action_dim)