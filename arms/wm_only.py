""" WM-ONLY arm: the critic is deleted; the TD-MPC2 latent model's chunk Q
    is the ONLY value function. Gated by check_sibling_resolution.py on the
    trained t4_mve_fixed_s0 checkpoints: sibling spread 0.90x the critic's,
    rank corr 0.40, pick agreement 0.32 (5x chance), action-gradient ratio
    0.85 -- the two-hot head can express and gradient-carry sibling-scale
    differences, so the closed loop (wm picks -> its picks execute -> it
    trains on their real outcomes) is worth one run.

    What changes vs the control (arms/control.py), and what does not:

      value function   the wm's chunk Q, trained in model_update by its own
                       chunk-TD loss on the SAME replay chunk transitions
                       the critic used to consume:
                         target = pooled chunk reward
                                  + gamma^h * chunk_mask
                                    * agg(TARGET heads at (z', actor chunk))
                       with agg = tdmpc.chunk_boot_agg (mean by default --
                       the equal-optimism rule; 'min' reproduces TD-MPC2's).
                       The wm's own bootstrap chunk is ALWAYS the actor's
                       (the proven configuration; the drifting prior stays
                       out of the loop).
      best-of-N        unchanged selector, unchanged n (chunk.select_n): it
                       reads policy.critic, which here is a facade onto the
                       wm's Q heads. The wm's picks are what execute -- the
                       closed loop Daniel asked for.
      actor            unchanged QC-FQL losses (BC flow + alpha * distill +
                       Q term); the Q term's gradient flows through the
                       wm's net path (encode + q_values), the same path its
                       own training loss uses.
      critic           GONE. update_critic is a no-op, no ChunkCritic is
                       trained, saved or read. arm.critic_target returns
                       None.

    Cadence: the value function now trains in model_update, which the loop
    runs every tdmpc.train_every steps (default 2). As the ONLY value
    function that halved cadence is a handicap the critic never had -- run
    with --tdmpc.train_every=1.

    Stray-gradient note: the actor's backward pushes gradients into the
    wm's parameters through the Q term; the wm's optimizer zero_grads at
    the start of its own update (tdmpc/agent.py), so they never step. """

import torch

from arms.control import ControlArm
from sac_chunked.sac_chunk_agent import ChunkAgent
from tdmpc.agent import TDMPC2Model
from tdmpc.diagnostics import chunk_q_report, model_report


class _WMValueFacade:
    """ Stands where ChunkAgent.critic (the ChunkCritic module) stood.
        Returns per-head values (num_q, B, 1) from the wm via the
        grad-capable net path, so update_actor's Q term, the selector's
        best-of-N and every q_scale-style diagnostic run through the wm
        without any of them changing. _agg with q_agg=mean then averages
        the heads, the same reduction the critic used. """

    def __init__(self, agent):
        self.agent = agent

    def __call__(self, feat, chunk):
        m = self.agent.model
        assert m is not None, 'wm not attached to the agent yet'
        return m.net.q_values(m.net.encode(feat), chunk)


class WMOnlyAgent(ChunkAgent):
    """ ChunkAgent minus its critic. Actor machinery is inherited
        unchanged; self.critic is the wm facade; value TRAINING lives in
        the arm's model_update, not here. """

    def __init__(self, **kw):
        super().__init__(**kw)
        self.model = None                     # attached by WMOnlyArm.build_model
        # The critic modules super() built are never trained or read here;
        # drop them so checkpoints and optimizers carry the actors only.
        for attr in ('critic', 'critic_target', 'critic_opt'):
            if hasattr(self, attr):
                delattr(self, attr)
        self.critic = _WMValueFacade(self)

    def update_critic(self, feat, chunk, targets, valid, metrics_on=True):
        """ No-op: the wm's chunk-TD loss (arm.model_update) is the value
            update. targets is None (arm.critic_target). """
        return {}

    def update_target(self):
        """ No-op: the wm maintains its own target network (tdmpc.tau). """

    def chunk_target_values(self, next_feats):
        raise RuntimeError('wm_only has no QC critic target -- '
                           'arm.critic_target returns None by design')

    def state_dict_all(self):
        return {
            'actor_bc_flow': self.actor_bc_flow.state_dict(),
            'actor_onestep_flow': self.actor_onestep_flow.state_dict(),
        }

    def load_state_dict_all(self, state):
        self.actor_bc_flow.load_state_dict(state['actor_bc_flow'])
        self.actor_onestep_flow.load_state_dict(state['actor_onestep_flow'])


class WMOnlyArm(ControlArm):
    name = 'wm_only'

    def build_policy(self):
        return WMOnlyAgent(
            repr_dim=self.obs_dim, action_dim=self.action_dim, chunk_len=self.chunk_len,
            device=self.device, lr=self.chunk.lr, hidden_dim=self.chunk.hidden_dim,
            num_layers=self.chunk.num_layers,
            critic_target_tau=self.chunk.critic_target_tau,
            ensemble=self.chunk.ensemble, alpha=self.chunk.alpha,
            flow_steps=self.chunk.flow_steps, q_agg=self.chunk.q_agg,
            compile_nets=self.chunk.compile_nets)

    def build_model(self):
        self.model = TDMPC2Model(self.obs_dim, self.action_dim, self.device,
                                 self.config.tdmpc, self.gamma,
                                 chunk_len=self.chunk_len, q_mode='chunk')
        self.policy.model = self.model
        return self.model

    def describe(self):
        t = self.config.tdmpc
        return (f'{self.name}: NO critic -- wm chunk-Q is the value function '
                f'(agg: {t.chunk_boot_agg}, wm bootstrap: actor, '
                f'train_every: {t.train_every}) | best-of-{self.chunk.select_n} '
                f'scored by the wm | alpha {self.chunk.alpha}')

    def critic_target(self, next_obs, reward, mask, metrics_on=False):
        return None                           # consumed by the no-op update_critic

    def model_update(self, replay, metrics_on):
        t = self.config.tdmpc
        w = replay.sample_model_windows(t.batch_size, t.horizon, self.device,
                                        self.rng, online_frac=t.online_frac)
        if w is None:
            return {}
        chunk_batch = replay.sample_chunks(t.batch_size, self.device, self.rng,
                                           self.gamma)
        if chunk_batch is None:
            return {}
        with torch.no_grad():
            next_chunk = self.policy.sample_chunk(chunk_batch[6])
        return self.model.update(w['obs'], w['next_obs'], w['action'],
                                 w['reward'], w['mask'], w['valid'],
                                 chunk_batch=chunk_batch, next_chunk=next_chunk,
                                 metrics_on=metrics_on)

    def report(self, replay):
        t = self.config.tdmpc
        if t.diag_windows <= 0:
            return {}
        rep = model_report(self.model, self.policy, replay, self.chunk_len,
                           t.diag_depth, self.gamma, self.device, self.rng,
                           num_windows=t.diag_windows)
        # chunk_q_report's "critic" side reads the facade, i.e. the wm
        # itself: chunkq_* vs criticq_* will agree by construction here.
        # Kept for the hit-gap panels' comparability across arms.
        rep.update(chunk_q_report(self.model, self.policy, replay,
                                  self.chunk_len, self.device, self.rng,
                                  num_windows=t.diag_windows,
                                  hit_thresh=t.diag_hit_thresh))
        return rep