import torch
import torch.nn as nn
from helpers.sac_wm_utils import soft_update_params, weight_init

GRAD_CLIP_NORM = 10.0

class ChunkActor(nn.Module):
    """ Noise-conditioned chunk policy: one noise vector in, the whole chunk
        out in a single forward pass.

        The randomness is an INPUT, not something added to the output. That is
        the difference that makes chunks coherent: with a per-dimension
        Gaussian head, each of the chunk_len actions gets its own independent
        draw and the arm shakes; here the network sees the whole draw at once
        and can spend it on a single consistent motion.

        There is no log_std, no log_prob and no entropy coefficient. Nothing
        downstream of this class needs them. The unbounded pre-tanh mean that
        MU_CLIP / MU_REG existed to contain also cannot arise, because the
        distillation term in update_actor is a direct opposing force on the
        output. """

    def __init__(self, repr_dim, chunk_dim, feature_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh())
        self.net = nn.Sequential(
            nn.Linear(feature_dim + chunk_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, chunk_dim))
        self.apply(weight_init)

    def forward(self, feat, z):
        h = self.trunk(feat)
        return torch.tanh(self.net(torch.cat([h, z], dim=-1)))

class ChunkCritic(nn.Module):
    """ Scores a whole chunk: Q(feat, a_t ... a_t+h-1). Returns all ensemble
        members stacked as (ensemble, batch, 1). """

    def __init__(self, repr_dim, chunk_dim, feature_dim, hidden_dim, ensemble):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh())
        self.qs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim + chunk_dim, hidden_dim), nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1))
            for _ in range(ensemble)])
        self.apply(weight_init)

    def forward(self, feat, chunk):
        h = torch.cat([self.trunk(feat), chunk], dim=-1)
        return torch.stack([q(h) for q in self.qs], dim=0)

class ChunkAgent:
    """ Action-chunked off-policy actor-critic, QC-FQL style. Shared by both
        arms of the comparison: the world-model arm feeds it RSSM features and
        imagined chunk transitions, the baseline arm feeds it raw observations
        and real chunk transitions from replay. Everything below is identical
        across the two, which is what keeps the comparison controlled. """

    def __init__(self, repr_dim, action_dim, chunk_len, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, ensemble=10, bc_alpha=300.0,
                 normalize_q=True):
        self.device = device
        self.action_dim = action_dim
        self.chunk_len = chunk_len
        self.chunk_dim = action_dim * chunk_len
        self.critic_target_tau = critic_target_tau
        self.bc_alpha = bc_alpha
        self.normalize_q = normalize_q

        self.actor = ChunkActor(repr_dim, self.chunk_dim, feature_dim, hidden_dim).to(device)
        self.critic = ChunkCritic(repr_dim, self.chunk_dim, feature_dim, hidden_dim, ensemble).to(device)
        self.critic_target = ChunkCritic(repr_dim, self.chunk_dim, feature_dim, hidden_dim, ensemble).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def noise(self, batch):
        return torch.randn(batch, self.chunk_dim, device=self.device)

    @torch.no_grad()
    def act(self, feat, eval_mode, step=None):
        """ feat: (repr_dim,) numpy array. Returns (chunk_len, action_dim).

            Eval uses z = 0, the noise-conditioned analogue of taking the mean
            of a Gaussian policy. """
        feat_t = torch.as_tensor(feat, device=self.device).float().unsqueeze(0)
        z = torch.zeros(1, self.chunk_dim, device=self.device) if eval_mode else self.noise(1)
        chunk = self.actor(feat_t, z)
        return chunk.cpu().numpy()[0].reshape(self.chunk_len, self.action_dim)

    @torch.no_grad()
    def chunk_target_values(self, next_feats):
        """ Value at a chunk boundary: the target critic's score for the chunk
            the current actor would take from there.

            The ensemble is averaged rather than min-reduced, matching the
            critic loss in the QC-FQL paper. With 10 members a min would be
            severely pessimistic. """
        z = self.noise(next_feats.shape[0])
        next_chunk = self.actor(next_feats, z)
        return self.critic_target(next_feats, next_chunk).mean(0)

    def update_critic(self, feat, chunk, target_Q, weight):
        metrics = {}
        target_Q = target_Q.detach()

        qs = self.critic(feat, chunk)
        wsum = weight.sum().clamp_min(1e-6)
        critic_loss = sum(
            (weight * (q - target_Q) ** 2).sum() / wsum for q in qs)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), GRAD_CLIP_NORM)
        self.critic_opt.step()

        q_mean = qs.mean(0)
        metrics['critic_loss'] = critic_loss.item()
        metrics['critic_target_q'] = target_Q.mean().item()
        metrics['critic_q'] = q_mean.mean().item()
        metrics['diagnosis/critic_grad_norm'] = critic_grad_norm.item()
        metrics['diagnosis/critic_q_std'] = q_mean.std().item()
        metrics['diagnosis/critic_ensemble_spread'] = qs.std(0).mean().item()
        metrics['diagnosis/critic_target_q_min'] = target_Q.min().item()
        metrics['diagnosis/critic_target_q_max'] = target_Q.max().item()
        metrics['diagnosis/critic_target_q_range'] = (target_Q.max() - target_Q.min()).item()
        return metrics

    def update_actor(self, feat, weight, flow_bc):
        """ Two terms:
              -Q(feat, chunk)                       push toward high value
              alpha * ||chunk - flow_chunk||^2      stay near real behavior

            Both the actor and the flow model are given the SAME z. Without
            that, the second term would pull the actor toward the average of
            every valid motion, and the average of "reach left" and "reach
            right" is a motion that does neither. Sharing z asks instead: for
            this particular draw, what would the behavior model have done? """
        metrics = {}
        z = self.noise(feat.shape[0])
        chunk = self.actor(feat, z)

        q = self.critic(feat, chunk).mean(0)
        wsum = weight.sum().clamp_min(1e-6)
        q_term = (weight * (-q)).sum() / wsum
        # Dividing by |Q| keeps alpha comparable across reward scales and
        # across training, so a value tuned early does not silently become
        # weak once Q grows.
        if self.normalize_q:
            q_term = q_term / q.abs().mean().detach().clamp_min(1e-6)

        bc_chunk = flow_bc.sample(feat, z)
        bc_term = (weight * ((chunk - bc_chunk) ** 2).mean(-1, keepdim=True)).sum() / wsum
        # Dividing by (1 + alpha) leaves the BC-to-Q influence ratio at exactly
        # alpha:1 -- the tuning knob is unchanged -- but keeps the loss scale
        # near O(1) for every alpha. Without this the gradient norm at alpha in
        # the hundreds is far above GRAD_CLIP_NORM, so clip_grad_norm_ rescales
        # every update back to the same magnitude and alpha stops having any
        # effect at all. Clipping should catch spikes, not silently undo the
        # hyperparameter being swept.
        actor_loss = (q_term + self.bc_alpha * bc_term) / (1.0 + self.bc_alpha)

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), GRAD_CLIP_NORM)
        self.actor_opt.step()

        metrics['actor_loss'] = actor_loss.item()
        metrics['actor_q_term'] = q_term.item()
        metrics['actor_bc_term'] = bc_term.item()
        # Fraction of the actor loss coming from the behavior constraint. This
        # is the number to sweep against: near 1.0 the policy is pure imitation
        # and the critic cannot improve on the data, near 0.0 the chunk is
        # unconstrained. Note bc_term shrinks over training while the
        # normalized q_term stays O(1), so this share drifts down on its own --
        # read it late in the run, not at step 0.
        _bc = self.bc_alpha * bc_term.detach()
        metrics['diagnosis/actor_bc_share'] = (_bc / (q_term.detach().abs() + _bc).clamp_min(1e-8)).item()
        metrics['diagnosis/actor_grad_norm'] = actor_grad_norm.item()
        metrics['diagnosis/actor_chunk_abs_mean'] = chunk.detach().abs().mean().item()
        metrics['diagnosis/actor_chunk_sat_frac'] = (chunk.detach().abs() > 0.95).float().mean().item()
        metrics['diagnosis/actor_bc_gap'] = (chunk.detach() - bc_chunk).abs().mean().item()
        # Within-chunk jerk: mean absolute difference between consecutive
        # actions inside a chunk. This is the direct readout of whether the
        # chunk is a committed motion or open-loop noise, measured on the
        # actor's own output rather than on the executed trajectory.
        c = chunk.detach().reshape(-1, self.chunk_len, self.action_dim)
        metrics['diagnosis/actor_intra_chunk_jerk'] = (c[:, 1:] - c[:, :-1]).abs().mean().item()
        return metrics

    @torch.no_grad()
    def chunk_diversity(self, feat, n_samples=8):
        """ The failure mode Option A has instead of entropy collapse: the
            actor learns to ignore z, every noise draw maps to the same chunk,
            and exploration dies. Nothing auto-corrects this, so it has to be
            watched. Trending toward zero means bc_alpha is too low. """
        sub = feat[:min(256, feat.shape[0])]
        chunks = torch.stack([
            self.actor(sub, self.noise(sub.shape[0])) for _ in range(n_samples)], dim=0)
        return chunks.std(0).mean().item()

    def update_target(self):
        soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

    def state_dict_all(self):
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
        }

    def load_state_dict_all(self, state):
        self.actor.load_state_dict(state['actor'])
        self.critic.load_state_dict(state['critic'])
        self.critic_target.load_state_dict(state['critic_target'])