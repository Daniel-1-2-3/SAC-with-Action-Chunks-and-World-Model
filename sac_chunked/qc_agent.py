""" QC (Li et al. 2025, "Reinforcement Learning with Action Chunking",
    Algorithm 1): action-chunked Q-learning with a flow behavior policy and
    critic best-of-N. NOT QC-FQL: there is no distilled one-step actor, no
    distillation loss, no alpha.

    PyTorch port of github.com/ColinQiyangLi/qc at 48283b4, agents/acfql.py
    with actor_type='best-of-n'. Line numbers below are that file's.

      networks       L262-282: critic Value(hidden 512x4, layer_norm, num_qs=2)
                     and actor_bc_flow ActorVectorField(hidden 512x4, no
                     layer_norm). Shared with ChunkAgent (ChunkCritic,
                     ActorVectorField in sac_chunk_agent.py).
      flow sampling  L207-223 compute_flow_actions: Euler from noise,
                     flow_steps steps on the grid t = i / flow_steps, each
                     step adds velocity / flow_steps, ONE clip to [-1, 1] at
                     the end. Reused verbatim (ChunkAgent.compute_flow_actions)
                     by sample_chunks_bc.
      policy         L180-203 sample_actions: N noise vectors per state,
                     flow, clip, q = mean over the critic ensemble of the
                     ONLINE critic (q_agg 'mean', L193-194), argmax over N.
                     best_of_n. Used at act time, at eval time
                     (evaluation.py L88 calls the same sample_actions) and
                     inside the TD target.
      critic loss    L22-52, their eq. 11: a* = sample_actions(s') (L32) --
                     best-of-N under the ONLINE critic -- then the TARGET
                     critic at (s', a*) (L34), mean over heads (L38);
                     target = R_h + gamma^h * mask_h * Q_target (L40-41);
                     loss = mean over (heads, batch) of valid_h * (Q - target)^2
                     (L45). chunk_target_values + ChunkAgent.update_critic.
      actor loss     L54-108: the rectified-flow BC loss L63-79 (their eq.
                     18, the same one QC-FQL trains), masked per position by
                     valid; distill_loss = q_loss = 0 in best-of-n mode
                     (L97-99), so actor_loss = bc_flow_loss (L102).
                     ChunkAgent.bc_flow_loss.
      update         L110-152: critic + actor step, then polyak tau on the
                     critic (L129-136, L147). Same optimizer (adam, lr 3e-4,
                     L299-301), tau 0.005, batch 256, discount 0.99, as
                     ChunkAgent.

    Deviations from acfql.py, all listed:
      1. Two Adam optimizers (critic, BC flow) instead of one Adam over the
         summed loss (L126, L301). Identical updates: Adam is per-parameter
         and the two losses touch disjoint parameters once the actor's Q
         term is zero (L97-99).
      2. ChunkAgent.update_critic sums the per-head masked means instead of
         averaging over (heads, batch) (L45): the loss is num_qs times
         theirs. Adam is invariant to a constant gradient scale except
         through eps=1e-8, so the update is the same to that precision.
      3. Weight init is PyTorch's nn.Linear default, not flax's
         variance_scaling(1.0, 'fan_avg', 'uniform') with zero bias
         (utils/networks.py L8-10). Shared with ChunkAgent.
      4. actor_onestep_flow is not built. The reference builds it (L277-282)
         but in best-of-n mode it receives zero gradient and is never read.
      5. Replay: the reference clips dataset actions to +-(1 - 1e-5)
         (envs/env_utils.py L149-153); ours stores them as loaded. Window
         sampling, masks, valid and the pooled reward follow
         utils/datasets.py sample_sequence (sac_chunked/replay.py).
      6. chunk_diversity (a metric only) is the std across the N flow
         samples at a state; the reference has no such metric.
    Everything the loop does around the agent (offline then online phases,
    start_training, utd_ratio, chunk executed open loop) is main.py's, see
    sac_chunked/experiment.py. """

import torch

from sac_chunked.sac_chunk_agent import ActorVectorField, ChunkAgent, ChunkCritic


class QCAgent(ChunkAgent):
    """ Shares with ChunkAgent: _agg, noise, compute_flow_actions,
        update_critic, update_target, bc_flow_loss. Everything that touched
        the one-step actor is replaced by best-of-N over the flow policy. """

    def __init__(self, repr_dim, action_dim, chunk_len, device, lr, hidden_dim,
                 num_layers, critic_target_tau, ensemble=2, num_samples=32,
                 flow_steps=10, q_agg='mean', actor_layer_norm=False,
                 critic_layer_norm=True, compile_nets=False):
        # ChunkAgent.__init__ is deliberately NOT called: QC has no
        # actor_onestep_flow and no alpha.
        self.device = device
        self.action_dim = action_dim
        self.chunk_len = chunk_len
        self.chunk_dim = action_dim * chunk_len
        self.critic_target_tau = critic_target_tau
        self.flow_steps = flow_steps
        self.q_agg = q_agg
        self.num_samples = int(num_samples)
        self.alpha = None   # QC has no distillation term; chunk.alpha is ignored

        self.actor_bc_flow = ActorVectorField(
            repr_dim, self.chunk_dim, hidden_dim, num_layers, actor_layer_norm, with_time=True).to(device)
        self.critic = ChunkCritic(
            repr_dim, self.chunk_dim, hidden_dim, num_layers, ensemble, critic_layer_norm).to(device)
        self.critic_target = ChunkCritic(
            repr_dim, self.chunk_dim, hidden_dim, num_layers, ensemble, critic_layer_norm).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor_bc_flow.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        if compile_nets:
            try:
                opts = dict(mode='default', fullgraph=False)
                self.actor_bc_flow = torch.compile(self.actor_bc_flow, **opts)
                self.critic = torch.compile(self.critic, **opts)
                self.critic_target = torch.compile(self.critic_target, **opts)
                print('[compile] torch.compile(default) enabled on QC flow/critic')
            except Exception as e:
                print(f'[compile] FAILED, running uncompiled: {e}')

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor_bc_flow.train(training)
        self.critic.train(training)

    # ------------------------------------------------------------- policy

    def _repeat(self, feat, n):
        """ (B, repr) -> (B * n, repr), row i of feat repeated n times in a
            block, matching jnp.repeat(observations[..., None, :], N, axis=-2)
            at L190 after the flatten at L202. """
        B = feat.shape[0]
        return feat.unsqueeze(1).expand(B, n, feat.shape[-1]).reshape(B * n, -1)

    @torch.no_grad()
    def sample_chunks_bc(self, feat, n):
        """ L183-192: n noise vectors per state through the behavior flow,
            Euler with flow_steps steps on t = i / flow_steps, clipped once
            at the end (compute_flow_actions). (B, repr) -> (B * n,
            chunk_dim), the n samples of state b in rows b*n .. b*n+n-1 (the
            reference's flat layout before its reshape at L202); for one
            state that is (n, chunk_dim). """
        B = feat.shape[0]
        noises = torch.randn(B * n, self.chunk_dim, device=feat.device, dtype=feat.dtype)
        return self.compute_flow_actions(self._repeat(feat, n), noises)

    @torch.no_grad()
    def best_of_n(self, feat, n=None):
        """ L180-203: argmax_i agg_k Q_theta_k(s, a_i) over n flow samples,
            ONLINE critic, q_agg over the ensemble. (B, repr) -> (B, chunk_dim). """
        n = self.num_samples if n is None else int(n)
        B = feat.shape[0]
        cands = self.sample_chunks_bc(feat, n)                        # (B * n, D)
        q = self._agg(self.critic(self._repeat(feat, n), cands))      # (B * n, 1)
        idx = q.reshape(B, n).argmax(-1)
        return cands.reshape(B, n, self.chunk_dim)[torch.arange(B, device=feat.device), idx]

    def sample_chunk(self, feat, noises=None):
        """ The policy IS best-of-N. `noises` is accepted for interface
            compatibility and ignored (N fresh draws are made). """
        return self.best_of_n(feat)

    @torch.no_grad()
    def act(self, feat, eval_mode, step=None):
        """ feat: (repr_dim,) numpy. Returns (chunk_len, action_dim). The
            reference uses the same sample_actions at train and eval time
            (evaluation.py L63, L88), so eval_mode is ignored. """
        feat_t = torch.as_tensor(feat, device=self.device).float().unsqueeze(0)
        chunk = self.best_of_n(feat_t)
        return chunk.cpu().numpy()[0].reshape(self.chunk_len, self.action_dim)

    @torch.no_grad()
    def chunk_target_values(self, next_feats):
        """ Their eq. 11 bootstrap: a* = best-of-N at s' under the ONLINE
            critic (L32 -> L194), then the TARGET critic at (s', a*) (L34),
            aggregated with q_agg (L35-38). (B, 1). """
        a_star = self.best_of_n(next_feats)
        return self._agg(self.critic_target(next_feats, a_star))

    # ----------------------------------------------------------- training

    def update_actor(self, feat, weight, bc_feat, bc_chunk, bc_valid=None,
                     actor_batch=None, metrics_on=True):
        """ L54-108 in best-of-n mode: actor_loss = bc_flow_loss, nothing
            else. feat / weight / actor_batch are the ChunkAgent interface
            and unused here. """
        bc_flow_loss = self.bc_flow_loss(bc_feat, bc_chunk, bc_valid)
        self.actor_opt.zero_grad(set_to_none=True)
        bc_flow_loss.backward()
        self.actor_opt.step()
        if not metrics_on:
            return {}
        return {'actor_loss': bc_flow_loss.item(),
                'bc_flow_loss': bc_flow_loss.item()}

    @torch.no_grad()
    def chunk_diversity(self, feat, n_samples=None):
        """ Std across the N flow samples at the same state, averaged over
            chunk dims and states. Same metric name as ChunkAgent's so the
            dashboards line up. """
        n = self.num_samples if n_samples is None else int(n_samples)
        sub = feat[:min(256, feat.shape[0])]
        return self.sample_chunks_bc(sub, n).reshape(sub.shape[0], n, -1).std(1).mean().item()

    def state_dict_all(self):
        return {
            'actor_bc_flow': self.actor_bc_flow.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
        }

    def load_state_dict_all(self, state):
        self.actor_bc_flow.load_state_dict(state['actor_bc_flow'])
        self.critic.load_state_dict(state['critic'])
        self.critic_target.load_state_dict(state['critic_target'])
