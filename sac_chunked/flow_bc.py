import torch
import torch.nn as nn
from helpers.sac_wm_utils import weight_init

GRAD_CLIP_NORM = 10.0

class FlowVelocity(nn.Module):
    """ Velocity field for a rectified-flow model over action chunks.
        Takes (state feature, intermediate chunk, flow time) and predicts the
        direction the intermediate chunk should move in. """

    def __init__(self, repr_dim, chunk_dim, feature_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh())
        self.net = nn.Sequential(
            nn.Linear(feature_dim + chunk_dim + 1, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, chunk_dim))
        self.apply(weight_init)

    def forward(self, feat, m, u):
        h = self.trunk(feat)
        return self.net(torch.cat([h, m, u], dim=-1))

class FlowBC:
    """ Behavior model: learns the distribution of real chunk_len-step action
        sequences in the replay data. It knows nothing about the task -- it
        only knows what plausible motion looks like. The actor is kept near it
        by a distillation term, which is what makes chunks coherent instead of
        open-loop jitter. """

    def __init__(self, repr_dim, chunk_dim, device, lr, feature_dim, hidden_dim,
                 flow_steps=10):
        self.device = device
        self.chunk_dim = chunk_dim
        self.flow_steps = flow_steps
        self.net = FlowVelocity(repr_dim, chunk_dim, feature_dim, hidden_dim).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def train(self, training=True):
        self.net.train(training)

    def _integrate(self, feat, z):
        # Euler integration of dm/du = v(feat, m, u) from u=0 (pure noise) to
        # u=1 (a chunk). Note the paper's Algorithm 1 writes this as a plain
        # assignment m <- f(s, m, u), which drops the step size; the correct
        # rectified-flow update accumulates dt * velocity, which is what this
        # does.
        m = z
        dt = 1.0 / self.flow_steps
        for i in range(self.flow_steps):
            u = torch.full((m.shape[0], 1), i * dt, device=m.device, dtype=m.dtype)
            m = m + dt * self.net(feat, m, u)
        return m.clamp(-1.0, 1.0)

    @torch.no_grad()
    def sample(self, feat, z):
        """ The chunk this model would produce from noise z. Used as the
            distillation target, so it must be a constant -- no gradient
            flows back into the flow network from the actor's loss. """
        return self._integrate(feat, z)

    def update(self, feat, chunk):
        """ Flow-matching loss: interpolate between noise and a real chunk at
            a random time u, and predict the straight-line direction from the
            noise to the chunk. """
        z = torch.randn_like(chunk)
        u = torch.rand(chunk.shape[0], 1, device=chunk.device, dtype=chunk.dtype)
        m = u * chunk + (1.0 - u) * z
        pred = self.net(feat, m, u)
        loss = ((pred - (chunk - z)) ** 2).mean()

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), GRAD_CLIP_NORM)
        self.opt.step()

        return {
            'bc_flow_loss': loss.item(),
            'diagnosis/bc_flow_grad_norm': grad_norm.item(),
            'diagnosis/bc_chunk_abs_mean': chunk.abs().mean().item(),
            'diagnosis/bc_pairs': float(chunk.shape[0]),
        }

    def state_dict_all(self):
        return {'flow': self.net.state_dict()}

    def load_state_dict_all(self, state):
        self.net.load_state_dict(state['flow'])
