import numpy as np
import torch
import torch.nn as nn


class RNDNovelty:
    """ Random Network Distillation over world-model feature vectors.

        Two identical MLPs on the SAME features the selector already scores
        (feat2tensor of an RSSM carry): a frozen randomly-initialized target
        and a trained predictor. Novelty of a feature = squared error of the
        predictor against the target. The predictor trains only on features
        of states the policy actually visited (fed in by observe()), so the
        error decays wherever the policy has been and stays high where it
        hasn't -- a novelty signal that self-anneals per state with no
        schedule, no floor, and no sampling noise (one deterministic score
        per feature, unlike the std-over-draws signal).

        Networks are built lazily on the first feature seen, because the
        feature dim is only known at runtime. """

    def __init__(self, device, hidden=256, out=128, lr=1e-4,
                 buffer_size=2048, batch=256, train_every=4):
        self.device = device
        self.hidden = int(hidden)
        self.out = int(out)
        self.lr = float(lr)
        self.buffer_size = int(buffer_size)
        self.batch = int(batch)
        self.train_every = int(train_every)
        self.target = None
        self.predictor = None
        self.opt = None
        self._buf = []
        self._buf_pos = 0
        self._step = 0
        self.last_loss = float('nan')

    def _build(self, dim):
        def mlp():
            return nn.Sequential(
                nn.Linear(dim, self.hidden), nn.ReLU(),
                nn.Linear(self.hidden, self.hidden), nn.ReLU(),
                nn.Linear(self.hidden, self.out))
        self.target = mlp().to(self.device)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.predictor = mlp().to(self.device)
        self.opt = torch.optim.Adam(self.predictor.parameters(), lr=self.lr)

    def add(self, feat_np):
        """ feat_np: (1, D) numpy, a visited state's posterior feature.
            Stored on CPU; training batches move to device on demand. """
        f = torch.as_tensor(np.asarray(feat_np, dtype=np.float32)).reshape(-1)
        if self.target is None:
            self._build(f.shape[0])
        if len(self._buf) < self.buffer_size:
            self._buf.append(f)
        else:
            self._buf[self._buf_pos] = f
            self._buf_pos = (self._buf_pos + 1) % self.buffer_size
        self._step += 1
        if self._step % self.train_every == 0 and len(self._buf) >= self.batch:
            self._train_step()

    def _train_step(self):
        idx = np.random.randint(0, len(self._buf), size=self.batch)
        x = torch.stack([self._buf[i] for i in idx]).to(self.device)
        with torch.no_grad():
            y = self.target(x)
        loss = ((self.predictor(x) - y) ** 2).mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        self.last_loss = float(loss.item())

    @torch.no_grad()
    def score(self, feats):
        """ feats: (n, D) torch tensor on any device. Returns (n,) novelty =
            per-row mean squared predictor error. Deterministic given feats. """
        x = feats.to(self.device).float()
        if self.target is None:
            self._build(x.shape[1])
        return ((self.predictor(x) - self.target(x)) ** 2).mean(-1)