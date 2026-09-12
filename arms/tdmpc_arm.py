""" TDMPC2 arm -- the reference algorithm itself, with chunk execution:
    the baseline row "try tdmpc with action chunks".

    Nothing of QC-FQL runs here: no critic, no flow policy, no distilled
    actor, no best-of-N. The agent IS TD-MPC2 (nicklashansen/tdmpc2):
      - the latent model (tdmpc/model.py, q_mode='step' -- the faithful
        reference configuration) trains on replay windows every
        tdmpc.train_every steps: consistency + reward + step-TD value
        (min-of-2 target heads, the reference rule) + policy prior;
      - acting = MPPI in latent space (tdmpc/planner.py), which improves
        actions by SAMPLING and SCORING under the model -- no gradient
        of the value with respect to the action anywhere, the mechanism
        the wm_only arm died of;
      - chunk adaptation: the planner optimizes plan_chunks * chunk_len
        single-step actions and the env executes the first chunk_len
        open-loop, replanning at each chunk boundary.

    Offline-to-online per the repo protocol: 1M model updates on the
    seeded buffer, then online collection via the planner. Known risk,
    on the record: the policy prior trains BC-less on offline data (the
    drift measured Sep 8); MPPI is the reference's own corrector for a
    mediocre prior -- this row measures whether that is enough.

    Run with --tdmpc.train_every=1 (the value function and prior live on
    the model-update cadence). MPPI makes eval expensive: ~200 plans per
    episode x num_samples x horizon dynamics steps. """

import numpy as np
import torch

from sac_chunked.experiment import Arm
from tdmpc.agent import TDMPC2Model
from tdmpc.diagnostics import model_report
from tdmpc.planner import ChunkMPPI


class _PlanSelector:
    """ Selector-shaped shim: routes act-time decisions to the planner,
        forwarding eval_mode (the stock selector's n<=1 path drops it),
        resets the warm start at online episode boundaries, and reports
        planner gauges as select/ stats. """

    def __init__(self, agent):
        self.agent = agent
        self._stats = {}

    def begin_episode(self):
        self.agent.planner.reset()
        self._await_first = True
        self._first_plan_value = None

    def select(self, state_1d, eval_mode=False):
        chunk = self.agent.act(state_1d, eval_mode=eval_mode)
        p = self.agent.planner
        if getattr(self, '_await_first', False) and not eval_mode:
            self._await_first = False
            self._first_plan_value = p.last_plan_value
        for k, v in (('plan_value', p.last_plan_value),
                     ('plan_std', p.last_plan_std)):
            s, c = self._stats.get(k, (0.0, 0))
            if np.isfinite(v):
                self._stats[k] = (s + v, c + 1)
        return chunk

    def report_episode_discounted(self, disc_return):
        """ Planner optimism: the first plan's promised value minus the
            episode's realized discounted return. Persistently positive =
            the model's promises exceed what execution delivers (prior
            drift / model error, quantified); near zero = planning is
            trustworthy and the row's result is about the method. """
        v = getattr(self, '_first_plan_value', None)
        if v is None or not np.isfinite(v):
            return
        self._first_plan_value = None
        for k, val in (('plan_value_first', v),
                       ('plan_optimism', v - float(disc_return)),
                       ('realized_disc_return', float(disc_return))):
            s, c = self._stats.get(k, (0.0, 0))
            self._stats[k] = (s + val, c + 1)

    def pop_stats(self):
        out = {f'select/{k}': s / c for k, (s, c) in self._stats.items() if c}
        self._stats = {}
        return out


class TDMPC2ChunkAgent:
    """ The loop's policy interface, implemented by the planner + model.
        Every QC-FQL update hook is a no-op: training lives entirely in
        the arm's model_update. """

    def __init__(self, chunk_len, action_dim, device):
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self.device = device
        self.model = None          # attached by the arm
        self.planner = None

    @torch.no_grad()
    def act(self, state_1d, eval_mode=False):
        feat = torch.as_tensor(np.asarray(state_1d, dtype=np.float32),
                               device=self.device)
        return self.planner.plan(feat, eval_mode=eval_mode) \
                   .cpu().numpy().reshape(self.chunk_len, self.action_dim)

    @torch.no_grad()
    def sample_chunk(self, obs):
        """ The prior's chunk at obs -- diagnostics-only here. """
        z = self.model.encode(obs)
        out = []
        for _ in range(self.chunk_len):
            a = self.model.net.pi(z)[1]
            out.append(a)
            z = self.model.net.next(z, a)
        return torch.cat(out, dim=-1)

    @torch.no_grad()
    def chunk_target_values(self, next_feats):
        """ model_report's value probe: the reference step-Q at the
            prior's action ('avg' rule, the planner's own terminal). """
        z = self.model.encode(next_feats)
        return self.model.net.q_subset(z, self.model.net.pi(z)[1], reduce='avg')

    # --- QC-FQL hooks, all inert -------------------------------------
    def update_critic(self, *a, **k):
        return {}

    def update_actor(self, *a, **k):
        return {}

    def update_target(self):
        pass

    def chunk_diversity(self, obs):
        return 0.0

    def state_dict_all(self):
        return {}                  # the model checkpoints via arm.save

    def load_state_dict_all(self, state):
        pass


class TDMPC2Arm(Arm):
    name = 'tdmpc2_chunks'

    def build_policy(self):
        return TDMPC2ChunkAgent(self.chunk_len, self.action_dim, self.device)

    def build_model(self):
        self.model = TDMPC2Model(self.obs_dim, self.action_dim, self.device,
                                 self.config.tdmpc, self.gamma,
                                 chunk_len=self.chunk_len, q_mode='step')
        self.policy.model = self.model
        self.policy.planner = ChunkMPPI(self.model, self.config.tdmpc,
                                        self.chunk_len, self.action_dim,
                                        self.device)
        return self.model

    def build_selector(self):
        return _PlanSelector(self.policy)

    def describe(self):
        t = self.config.tdmpc
        return (f'{self.name}: TD-MPC2 (reference, step-mode) + MPPI over '
                f'{t.plan_chunks} chunk(s) x {self.chunk_len} steps, commit 1 '
                f'chunk | samples {t.num_samples} elites {t.num_elites} '
                f'iters {t.iterations} pi_trajs {t.num_pi_trajs} | '
                f'train_every {t.train_every}')

    def critic_target(self, next_obs, reward, mask, metrics_on=False):
        return None                # consumed by the inert update_critic

    def model_update(self, replay, metrics_on):
        t = self.config.tdmpc
        w = replay.sample_model_windows(t.batch_size, t.horizon, self.device,
                                        self.rng, online_frac=t.online_frac)
        if w is None:
            return {}
        return self.model.update(w['obs'], w['next_obs'], w['action'],
                                 w['reward'], w['mask'], w['valid'],
                                 metrics_on=metrics_on)

    def report(self, replay):
        t = self.config.tdmpc
        if t.diag_windows <= 0:
            return {}
        return model_report(self.model, self.policy, replay, self.chunk_len,
                            t.diag_depth, self.gamma, self.device, self.rng,
                            num_windows=t.diag_windows)
