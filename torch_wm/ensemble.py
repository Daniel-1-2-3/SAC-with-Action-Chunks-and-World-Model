""" Plan2Explore's disagreement ensemble, faithful to the paper:

      "we train a bootstrap ensemble to predict, from each model state,
       the next encoder features"  (P2E Sec. 3.1, Eq. 3)

    K one-step models q(h_{t+1} | w_k, s_t, a_t): input = latent model
    state (deter + stoch features) and action; target = the encoder's next
    embedding. Trained on posterior features from the world-model's own
    training batches. Intrinsic reward is Eq. 10, the variance of the
    ensemble means, computed ENTIRELY IN LATENT SPACE -- imagination never
    decodes. Self-annealing: heads converge wherever data accumulates, so
    disagreement -> 0 even under stochastic dynamics (their Sec. 3.1).

    No observation slicing, no shaping: the signal is aimed by epistemics
    alone, exactly as published.

    All K heads are fused in stacked einsum passes (no per-head loop).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentDisagreementEnsemble(nn.Module):
    def __init__(self, feat_dim, act_dim, embed_dim, k=5, hidden=256,
                 lr=3e-4, device='cpu'):
        super().__init__()
        self.k = k
        d_in = feat_dim + act_dim

        def w(fan_in, *shape):
            t = torch.empty(*shape)
            nn.init.uniform_(t, -1 / math.sqrt(fan_in), 1 / math.sqrt(fan_in))
            return nn.Parameter(t)

        # 2-hidden-layer MLPs, as in P2E's appendix
        self.w1, self.b1 = w(d_in, k, d_in, hidden), w(d_in, k, hidden)
        self.w2, self.b2 = w(hidden, k, hidden, hidden), w(hidden, k, hidden)
        self.w3, self.b3 = (w(hidden, k, hidden, embed_dim),
                            w(hidden, k, embed_dim))
        self.to(device)
        self.device = device
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

    def _forward_all(self, x):
        """ x (B, feat+act) -> ensemble means (K, B, embed_dim), fused. """
        h = F.silu(torch.einsum('bi,kih->kbh', x, self.w1)
                   + self.b1.unsqueeze(1))
        h = F.silu(torch.einsum('kbh,khj->kbj', h, self.w2)
                   + self.b2.unsqueeze(1))
        return torch.einsum('kbj,kjo->kbo', h, self.w3) + self.b3.unsqueeze(1)

    def train_from_wm(self, feats, actions, embeds):
        """ feats (B, T, F), actions (B, T, A), embeds (B, T, E), all
            detached from the WM's training pass. Pairs are
            (feat_t, action_{t+1}) -> embed_{t+1}: action[t+1] is the
            action leading INTO step t+1 in the replay layout. """
        x = torch.cat([feats[:, :-1], actions[:, 1:]], -1).flatten(0, 1)
        tgt = embeds[:, 1:].flatten(0, 1).unsqueeze(0)        # (1, M, E)
        preds = self._forward_all(x)                          # (K, M, E)
        # independent bootstrap masks per head (Breiman bagging, per P2E)
        mask = (torch.rand(self.k, x.shape[0], 1,
                           device=self.device) < 0.8).float()
        loss = (mask * (preds - tgt) ** 2).mean() * self.k
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return {'ensemble_loss': float(loss.item()) / self.k}

    def disagreement(self, feat, act):
        """ Eq. 10: variance across ensemble means, averaged over embed
            dims. feat (B, F), act (B, A) -> (B,). Differentiable (the
            explorer's imagined reward backprops through this into the
            dynamics), so no no_grad here; callers detach as needed. """
        preds = self._forward_all(torch.cat([feat, act], -1))
        return preds.var(dim=0, unbiased=True).mean(-1)