""" MPPI planner -- a faithful port of nicklashansen/tdmpc2 `_plan` /
    `_estimate_value` (tdmpc2.py), adapted to CHUNK execution: the planner
    optimizes a sequence of `plan_chunks * chunk_len` SINGLE-STEP actions
    exactly as the reference does, but the caller commits the FIRST CHUNK
    (chunk_len steps) open-loop and replans at the next chunk boundary, so
    the warm start shifts by chunk_len rows instead of one.

    Ported line-for-line where the reference applies: prior seeding
    (num_pi_trajs rollouts of the model's policy prior), Gaussian sampling
    around (mean, std), value = sum of discounted predicted rewards plus
    the discounted step-Q at the horizon under the prior's action,
    elite softmax reweighting at `temperature`, std clamped to
    [min_std, max_std], Gumbel sampling of the returned elite, exploration
    noise std added outside eval. No termination head (our env has none)
    and no multitask masks -- those reference branches are dropped.

    Deviations, deliberate: chunk commitment (above); warm-start persists
    across episode boundaries unless reset() is called (the arm's selector
    shim calls it at online episode starts; eval episodes replan from the
    shifted mean -- noted, not load-bearing). """

import torch


class ChunkMPPI:
    def __init__(self, model, cfg, chunk_len, action_dim, device):
        """ model: TDMPC2Model (q_mode='step'). cfg: the tdmpc config
            block (num_samples, num_elites, iterations, min_std, max_std,
            temperature, num_pi_trajs, plan_chunks). """
        self.m = model
        self.chunk_len = int(chunk_len)
        self.action_dim = int(action_dim)
        self.horizon = int(cfg.plan_chunks) * self.chunk_len
        self.num_samples = int(cfg.num_samples)
        self.num_elites = int(cfg.num_elites)
        self.iterations = int(cfg.iterations)
        self.min_std = float(cfg.min_std)
        self.max_std = float(cfg.max_std)
        self.temperature = float(cfg.temperature)
        self.num_pi_trajs = int(cfg.num_pi_trajs)
        self.device = device
        self._prev_mean = None
        self.last_plan_value = float('nan')
        self.last_plan_std = float('nan')

    def reset(self):
        """ Episode boundary: forget the warm start (reference t0=True). """
        self._prev_mean = None

    @torch.no_grad()
    def _estimate_value(self, z, actions):
        """ Reference _estimate_value: G = sum_t gamma^t r(z_t, a_t), plus
            gamma^H Q(z_H, pi(z_H)) with the reference's 'avg' head rule. """
        G, discount = 0.0, 1.0
        for t in range(self.horizon):
            reward = self.m.net.reward_pred(z, actions[t])
            z = self.m.net.next(z, actions[t])
            G = G + discount * reward
            discount = discount * self.m.gamma
        pi_a = self.m.net.pi(z)[1]
        return G + discount * self.m.net.q_subset(z, pi_a, reduce='avg')

    @torch.no_grad()
    def plan(self, obs_1d, eval_mode=False):
        """ obs_1d: (obs_dim,) tensor. Returns (chunk_len, action_dim) in
            [-1, 1] -- the first chunk of the planned sequence, with the
            reference's exploration noise on it outside eval. """
        H, A, N, P = self.horizon, self.action_dim, self.num_samples, self.num_pi_trajs
        z0 = self.m.encode(obs_1d.reshape(1, -1))

        if P > 0:
            pi_actions = torch.empty(H, P, A, device=self.device)
            _z = z0.repeat(P, 1)
            for t in range(H - 1):
                pi_actions[t] = self.m.net.pi(_z)[1]
                _z = self.m.net.next(_z, pi_actions[t])
            pi_actions[-1] = self.m.net.pi(_z)[1]

        z = z0.repeat(N, 1)
        mean = torch.zeros(H, A, device=self.device)
        std = torch.full((H, A), self.max_std, device=self.device)
        if self._prev_mean is not None:
            # chunk commitment: the executed chunk_len rows fall off the front
            mean[:-self.chunk_len] = self._prev_mean[self.chunk_len:]
        actions = torch.empty(H, N, A, device=self.device)
        if P > 0:
            actions[:, :P] = pi_actions

        for _ in range(self.iterations):
            r = torch.randn(H, N - P, A, device=self.device)
            actions[:, P:] = (mean.unsqueeze(1) + std.unsqueeze(1) * r).clamp(-1, 1)
            value = self._estimate_value(z, actions).nan_to_num(0)
            elite_idxs = torch.topk(value.squeeze(1), self.num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]
            max_value = elite_value.max(0).values
            score = torch.exp(self.temperature * (elite_value - max_value))
            score = score / score.sum(0)
            mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
            std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2
                    ).sum(dim=1) / (score.sum(0) + 1e-9)).sqrt()
            std = std.clamp(self.min_std, self.max_std)

        # Gumbel sample of the returned elite (reference gumbel_softmax_sample)
        g = -torch.log(-torch.log(torch.rand_like(score.squeeze(1)).clamp_min(1e-10)))
        rand_idx = torch.argmax(torch.log(score.squeeze(1).clamp_min(1e-10)) + g)
        chosen = elite_actions[:, rand_idx]              # (H, A)
        chunk = chosen[:self.chunk_len]
        if not eval_mode:
            chunk = chunk + std[:self.chunk_len] * torch.randn_like(chunk)
        self._prev_mean = mean
        self.last_plan_value = float(elite_value[rand_idx])
        self.last_plan_std = float(std.mean())
        return chunk.clamp(-1, 1)
