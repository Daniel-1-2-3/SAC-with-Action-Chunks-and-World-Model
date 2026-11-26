""" Chunk proposer: a small network that copies the flow behaviour-cloning
    policy, but reads a LATENT instead of a raw observation.

    Deeper in the tree, chunks must be proposed from imagined latents. The
    actor only reads raw observations, and the TD-MPC2 prior is known to
    drift, so this learns latent -> chunk by distilling the flow policy:
    at each model update, noise and real states give flow chunks as
    targets, and the proposer regresses onto them from the states'
    (detached) latents. At act time the tree queries it at imagined
    latents with noise from the selector's own generator. """

import torch
import torch.nn as nn

from tdmpc.model import weight_init


class ProposerNet(nn.Module):
    """ concatenate(latent, noise) -> 3 hidden layers of proposer_hidden,
        Mish (the activation tdmpc/model.py uses) -> chunk -> tanh. """

    def __init__(self, latent_dim, chunk_dim, hidden):
        super().__init__()
        layers, d = [], latent_dim + chunk_dim
        for _ in range(3):
            layers += [nn.Linear(d, hidden), nn.Mish(inplace=False)]
            d = hidden
        layers.append(nn.Linear(d, chunk_dim))
        self.mlp = nn.Sequential(*layers)
        self.apply(weight_init)

    def forward(self, latent, noise):
        return torch.tanh(self.mlp(torch.cat([latent, noise], dim=-1)))


class ChunkProposer:
    """ Owns the network and its optimizer. train_step runs once per model
        update on the same windows the model trains on. """

    def __init__(self, latent_dim, chunk_dim, device, hidden=512, lr=3e-4):
        self.chunk_dim = int(chunk_dim)
        self.device = device
        self.net = ProposerNet(latent_dim, self.chunk_dim, int(hidden)).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=float(lr))

    def train_step(self, policy, model, states):
        """ states (B, obs_dim): the first observation of each model window.
            target = the flow behaviour-cloning chunk at those states (no
            gradient); latent = encode(states), detached (no gradient into
            the encoder). Returns the proposer loss as a float. """
        with torch.no_grad():
            noise = policy.noise(states.shape[0])
            target = policy.compute_flow_actions(states, noise)
            latent = model.encode(states).detach()
        pred = self.net(latent, noise)
        loss = ((pred - target) ** 2).mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return float(loss.item())

    @torch.no_grad()
    def propose(self, latents, generator):
        """ latents (B, latent_dim) -> chunks (B, chunk_dim) in [-1, 1],
            noise drawn from the given torch.Generator so act-time tree
            sampling doesn't consume the global RNG. """
        noise = torch.randn(latents.shape[0], self.chunk_dim,
                            device=latents.device, generator=generator)
        return self.net(latents, noise)

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, state):
        self.net.load_state_dict(state)
