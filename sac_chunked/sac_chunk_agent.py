import torch
import torch.nn as nn

def build_mlp(in_dim, hidden_dim, num_layers, out_dim, layer_norm):
    """ utils/networks.py MLP. Dense -> activation -> LayerNorm per hidden
        layer, activation is gelu, no activation on the output layer. """
    layers = []
    d = in_dim
    for _ in range(num_layers):
        layers.append(nn.Linear(d, hidden_dim))
        layers.append(nn.GELU())
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        d = hidden_dim
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)

class ActorVectorField(nn.Module):
    """ utils/networks.py ActorVectorField. Used for BOTH flow networks:
        actor_bc_flow takes a time argument, actor_onestep_flow does not.

        No output squashing. QC clips to [-1, 1] at the call site instead,
        and the distillation loss is computed on the UNCLIPPED output. """

    def __init__(self, repr_dim, chunk_dim, hidden_dim, num_layers, layer_norm, with_time):
        super().__init__()
        in_dim = repr_dim + chunk_dim + (1 if with_time else 0)
        self.with_time = with_time
        self.mlp = build_mlp(in_dim, hidden_dim, num_layers, chunk_dim, layer_norm)

    def forward(self, feat, chunk, t=None):
        if self.with_time:
            return self.mlp(torch.cat([feat, chunk, t], dim=-1))
        return self.mlp(torch.cat([feat, chunk], dim=-1))

class ChunkCritic(nn.Module):
    """ utils/networks.py Value with num_ensembles=num_qs, layer_norm=True.
        Returns all ensemble members stacked as (ensemble, batch, 1). """

    def __init__(self, repr_dim, chunk_dim, hidden_dim, num_layers, ensemble, layer_norm=True):
        super().__init__()
        self.qs = nn.ModuleList([
            build_mlp(repr_dim + chunk_dim, hidden_dim, num_layers, 1, layer_norm)
            for _ in range(ensemble)])

    def forward(self, feat, chunk):
        h = torch.cat([feat, chunk], dim=-1)
        return torch.stack([q(h) for q in self.qs], dim=0)

class ChunkAgent:
    """ QC-FQL: action-chunked Flow Q-learning, ported from
        github.com/ColinQiyangLi/qc agents/acfql.py (actor_type=distill-ddpg).

        Three networks. actor_bc_flow is a rectified-flow velocity field that
        models the behavior distribution over chunks. actor_onestep_flow maps
        a noise vector to a chunk in one pass and is distilled against it.
        The critic scores whole chunks.

        Both flow networks share one optimizer and one loss, exactly as the
        reference does -- there is no separate behavior-model update step.

        Shared by both arms of the comparison: the world-model arm feeds RSSM
        features and imagined chunk transitions, the baseline arm feeds raw
        observations and real chunk transitions from replay. """

    def __init__(self, repr_dim, action_dim, chunk_len, device, lr, hidden_dim,
                 num_layers, critic_target_tau, ensemble=2, alpha=300.0,
                 flow_steps=10, q_agg='mean', actor_layer_norm=False,
                 critic_layer_norm=True, compile_nets=False):
        self.device = device
        self.action_dim = action_dim
        self.chunk_len = chunk_len
        self.chunk_dim = action_dim * chunk_len
        self.critic_target_tau = critic_target_tau
        self.alpha = alpha
        self.flow_steps = flow_steps
        self.q_agg = q_agg

        self.actor_bc_flow = ActorVectorField(
            repr_dim, self.chunk_dim, hidden_dim, num_layers, actor_layer_norm, with_time=True).to(device)
        self.actor_onestep_flow = ActorVectorField(
            repr_dim, self.chunk_dim, hidden_dim, num_layers, actor_layer_norm, with_time=False).to(device)
        self.critic = ChunkCritic(
            repr_dim, self.chunk_dim, hidden_dim, num_layers, ensemble, critic_layer_norm).to(device)
        self.critic_target = ChunkCritic(
            repr_dim, self.chunk_dim, hidden_dim, num_layers, ensemble, critic_layer_norm).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # One optimizer over both flow networks, matching the reference's
        # single ModuleDict + single optax.adam. No gradient clipping: QC uses
        # plain adam, and a clip on a loss scaled by alpha silently rescales
        # every update back to the same magnitude, cancelling alpha entirely.
        self.actor_opt = torch.optim.Adam(
            list(self.actor_bc_flow.parameters()) + list(self.actor_onestep_flow.parameters()), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # A single actor update fires ~400 tiny CUDA kernels: flow_steps
        # sequential passes through a 4-layer MLP, times three networks, plus
        # backward. Each does microseconds of arithmetic, so wall-clock is
        # dominated by launch cost, not FLOPs. torch.compile fuses them.
        #
        # mode='default', NOT 'reduce-overhead'. The latter enables CUDA
        # graphs, which capture a fixed memory layout -- incompatible with two
        # separate backward passes per step (critic, then actor) and with the
        # critic being called under three different grad contexts. It fails at
        # runtime, not compile time.
        if compile_nets:
            try:
                opts = dict(mode='default', fullgraph=False)
                self.actor_bc_flow = torch.compile(self.actor_bc_flow, **opts)
                self.actor_onestep_flow = torch.compile(self.actor_onestep_flow, **opts)
                self.critic = torch.compile(self.critic, **opts)
                self.critic_target = torch.compile(self.critic_target, **opts)
                print('[compile] torch.compile(default) enabled on actor/critic')
            except Exception as e:
                print(f'[compile] FAILED, running uncompiled: {e}')

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor_bc_flow.train(training)
        self.actor_onestep_flow.train(training)
        self.critic.train(training)

    def noise(self, batch):
        return torch.randn(batch, self.chunk_dim, device=self.device)

    def _agg(self, qs):
        return qs.min(0).values if self.q_agg == 'min' else qs.mean(0)

    def sample_chunk(self, feat, noises=None):
        """ acfql.sample_actions for actor_type='distill-ddpg': one forward
            pass through the one-step policy, then clip. """
        if noises is None:
            noises = self.noise(feat.shape[0])
        return torch.clamp(self.actor_onestep_flow(feat, noises), -1.0, 1.0)

    @torch.no_grad()
    def compute_flow_actions(self, feat, noises):
        """ acfql.compute_flow_actions: Euler integration of the behavior flow
            from noise to a chunk. Note the reference divides the velocity by
            flow_steps -- the paper's Algorithm 1 writes a bare assignment and
            drops the step size, so the code is right and the paper is not.
            Clipping happens once at the end, not per step. """
        chunk = noises
        for i in range(self.flow_steps):
            t = torch.full((chunk.shape[0], 1), i / self.flow_steps,
                           device=chunk.device, dtype=chunk.dtype)
            chunk = chunk + self.actor_bc_flow(feat, chunk, t) / self.flow_steps
        return torch.clamp(chunk, -1.0, 1.0)

    @torch.no_grad()
    def act(self, feat, eval_mode, step=None):
        """ feat: (repr_dim,) numpy array. Returns (chunk_len, action_dim).

            eval_mode is ignored. The one-step policy is a noise-to-chunk map
            with no meaningful mean to take -- feeding it zeros would be off
            distribution. The reference samples noise for evaluation too
            (evaluation.py calls the same sample_actions). """
        feat_t = torch.as_tensor(feat, device=self.device).float().unsqueeze(0)
        chunk = self.sample_chunk(feat_t)
        return chunk.cpu().numpy()[0].reshape(self.chunk_len, self.action_dim)

    @torch.no_grad()
    def chunk_target_values(self, next_feats):
        """ Value at a chunk boundary: target critic scored on the chunk the
            one-step policy would take from there, aggregated with q_agg
            ('mean' in the reference, for every agent including its SAC one). """
        next_chunk = self.sample_chunk(next_feats)
        return self._agg(self.critic_target(next_feats, next_chunk))

    def update_critic(self, feat, chunk, target_Q, weight, n_real=None,
                      metrics_on=True):
        """ acfql.critic_loss. weight is `valid` for the replay arm and the
            imagined survival weight for the world-model arm. The reference
            multiplies by the mask and takes a plain mean -- it does NOT
            renormalize by the mask's sum.

            n_real: number of leading rows that are REAL transitions. When the
            caller concatenates [real, imagined], pass the real count so the
            diagnostics below can be reported per source.

            Why this matters: the blended metrics are NOT comparable between
            the two arms. The baseline arm passes 256 real rows; the
            world-model arm passes 256 real + ~12288 imagined, so a blended
            mean is ~98% imagined and will look different for reasons that
            have nothing to do with learning quality. The `_real` variants
            below are computed on the same 256-row real population in BOTH
            arms and are the only critic numbers safe to compare directly.

            The training loss itself is unchanged -- this only splits how the
            already-computed errors get reported. """
        metrics = {}
        target_Q = target_Q.detach()

        qs = self.critic(feat, chunk)
        sq = (qs - target_Q.unsqueeze(0)) ** 2 # (ensemble, batch, 1)
        critic_loss = sum((weight * sq[i]).mean() for i in range(sq.shape[0]))

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        if not metrics_on:
            # Each .item() below is a blocking GPU->CPU sync. With
            # log_every=100 these ran on all 100 steps and were used on one,
            # stalling the pipeline ~24 times per step for nothing. Training
            # math is untouched; only reporting is skipped.
            return metrics
        q_mean = qs.mean(0)
        spread = qs.std(0)
        metrics['critic_loss'] = critic_loss.item()
        metrics['critic_target_q'] = target_Q.mean().item()
        metrics['critic_q'] = q_mean.mean().item()
        metrics['diagnosis/critic_q_max'] = q_mean.max().item()
        metrics['diagnosis/critic_q_min'] = q_mean.min().item()
        metrics['diagnosis/critic_ensemble_spread'] = spread.mean().item()
        metrics['diagnosis/critic_target_q_range'] = (target_Q.max() - target_Q.min()).item()

        # Per-source split. In the baseline arm n_real covers the whole batch,
        # so the _real metrics equal the blended ones -- which is exactly what
        # makes them comparable across arms.
        n = feat.shape[0] if n_real is None else int(n_real)
        sq_det = sq.detach()
        metrics['diagnosis/critic_mse_real'] = sq_det[:, :n].mean().item()
        metrics['diagnosis/critic_spread_real'] = spread[:n].mean().item()
        metrics['diagnosis/critic_q_real'] = q_mean[:n].mean().item()
        metrics['diagnosis/critic_target_q_real'] = target_Q[:n].mean().item()
        if n < feat.shape[0]:
            metrics['diagnosis/critic_mse_imagined'] = sq_det[:, n:].mean().item()
            metrics['diagnosis/critic_spread_imagined'] = spread[n:].mean().item()
            metrics['diagnosis/critic_q_imagined'] = q_mean[n:].mean().item()
            metrics['diagnosis/critic_target_q_imagined'] = target_Q[n:].mean().item()
        return metrics

    def update_actor(self, feat, weight, bc_feat, bc_chunk, bc_valid=None,
                     actor_batch=None, metrics_on=True):
        """ acfql.actor_loss:

              actor_loss = bc_flow_loss + alpha * distill_loss + q_loss

            One backward, one optimizer step, both flow networks. No division
            by (1 + alpha) and no gradient clipping -- the reference has
            neither, and those two were only ever cancelling each other.

            feat drives the distillation and Q terms; bc_feat / bc_chunk drive
            the flow-matching term. In the baseline arm they are the same
            batch, exactly as the reference. In the world-model arm feat is
            imagined and bc_feat is real latents, because imagined chunks are
            not behavior data.

            actor_batch: if set, randomly subsample `feat` to this many rows
            before the distill/Q terms. compute_flow_actions runs flow_steps
            SEQUENTIAL passes through a 4x512 MLP, so its cost is linear in
            the row count and it dominates the actor update once the caller
            concatenates a large imagined batch. The reference computes these
            terms on batch_size (256) rows; the extra imagined coverage is
            what the CRITIC needs, not what the actor's gradient estimate
            needs. Leave as None to use every row. """
        metrics = {}

        # BC flow loss: interpolate noise -> real chunk at a random time and
        # predict the straight-line velocity.
        z0 = torch.randn_like(bc_chunk)
        t = torch.rand(bc_chunk.shape[0], 1, device=bc_chunk.device, dtype=bc_chunk.dtype)
        x_t = (1.0 - t) * z0 + t * bc_chunk
        vel = bc_chunk - z0
        pred = self.actor_bc_flow(bc_feat, x_t, t)
        sq = (pred - vel) ** 2
        if bc_valid is not None:
            # Masked per position inside the chunk, then a plain mean over all
            # elements including the masked ones -- not a masked mean.
            sq = sq.reshape(-1, self.chunk_len, self.action_dim) * bc_valid[..., None]
        bc_flow_loss = sq.mean()

        if actor_batch is not None and actor_batch < feat.shape[0]:
            sel = torch.randperm(feat.shape[0], device=feat.device)[:actor_batch]
            feat = feat[sel]
            weight = weight[sel]

        # Distillation: the same noise through the one-step policy and through
        # the integrated behavior flow. The flow target is detached.
        noises = self.noise(feat.shape[0])
        target_flow_chunk = self.compute_flow_actions(feat, noises)
        actor_chunk = self.actor_onestep_flow(feat, noises)
        distill_loss = ((actor_chunk - target_flow_chunk) ** 2).mean()

        # The Q term is computed on the CLIPPED chunk; the distillation term
        # above is computed on the unclipped one.
        clipped = torch.clamp(actor_chunk, -1.0, 1.0)
        q = self._agg(self.critic(feat, clipped))
        q_loss = -(weight * q).mean()

        actor_loss = bc_flow_loss + self.alpha * distill_loss + q_loss

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if not metrics_on:
            return metrics
        metrics['actor_loss'] = actor_loss.item()
        metrics['bc_flow_loss'] = bc_flow_loss.item()
        metrics['distill_loss'] = distill_loss.item()
        metrics['actor_q_term'] = q_loss.item()
        # Fraction of the actor loss coming from the behavior terms. Near 1.0
        # the policy is pure imitation; a share that RISES over training is
        # the signature of alpha set too high for this task's Q scale.
        _bc = bc_flow_loss.detach() + self.alpha * distill_loss.detach()
        metrics['diagnosis/actor_bc_share'] = (_bc / (q_loss.detach().abs() + _bc).clamp_min(1e-8)).item()
        metrics['diagnosis/actor_chunk_abs_mean'] = clipped.detach().abs().mean().item()
        metrics['diagnosis/actor_chunk_clip_frac'] = (actor_chunk.detach().abs() > 1.0).float().mean().item()
        metrics['diagnosis/actor_bc_gap'] = (actor_chunk.detach() - target_flow_chunk).abs().mean().item()
        # Within-chunk jerk: mean absolute difference between consecutive
        # actions inside a chunk. Direct readout of whether the chunk is a
        # committed motion or open-loop noise.
        c = clipped.detach().reshape(-1, self.chunk_len, self.action_dim)
        metrics['diagnosis/actor_intra_chunk_jerk'] = (c[:, 1:] - c[:, :-1]).abs().mean().item()
        return metrics

    @torch.no_grad()
    def chunk_diversity(self, feat, n_samples=8):
        """ Spread across noise draws at the same state. The one-step policy
            can in principle learn to ignore its noise input; the distillation
            term against a multimodal flow model is what prevents it. Watch
            this -- a trend toward zero means alpha is too low. """
        sub = feat[:min(256, feat.shape[0])]
        chunks = torch.stack([self.sample_chunk(sub) for _ in range(n_samples)], dim=0)
        return chunks.std(0).mean().item()

    def update_target(self):
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.mul_(1.0 - self.critic_target_tau).add_(self.critic_target_tau * p.data)

    def state_dict_all(self):
        return {
            'actor_bc_flow': self.actor_bc_flow.state_dict(),
            'actor_onestep_flow': self.actor_onestep_flow.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
        }

    def load_state_dict_all(self, state):
        self.actor_bc_flow.load_state_dict(state['actor_bc_flow'])
        self.actor_onestep_flow.load_state_dict(state['actor_onestep_flow'])
        self.critic.load_state_dict(state['critic'])
        self.critic_target.load_state_dict(state['critic_target'])