""" TREE-EXPLORE arm: the combined method + a world-model tree at act-time
    selection during ONLINE COLLECTION only.

    Training is the combined arm's, byte for byte: QC-FQL losses, the
    critic's own target, best-of-N execution. The one behavioral
    difference is which chunk gets executed while collecting online data:
    in an exploring episode, every root candidate is imagined several
    chunks deep (wm/tree_selector.py), each imagined end point scored by
    how far it is from anywhere actually visited (wm/visit_memory.py),
    untrustworthy branches cut against a per-depth real-data reference,
    and the critic's score gets a bounded bonus for the best "unvisited
    but trusted" place reachable through that candidate. The bonus never
    touches a target, a reward, a gradient, or eval.

    Whether an episode explores is chosen per episode by the same
    sliding-window upper-confidence-bound bandit the old explorer used
    (wm/episode_bandit.py): exploit episodes are exactly the combined arm.
    tree.beta = 0 makes every decision the combined method's.

    The TD-MPC2 model trains alongside on replay windows (consistency +
    reward + the reference step-mode value/prior), exactly as before; the
    chunk proposer (wm/chunk_proposer.py) distills the flow
    behaviour-cloning policy into latent space with one extra step per
    model update, and the trust reference is re-measured on real action
    sequences every tree.ref_every model updates, in the offline phase
    too, so it is warm when online starts. """

import torch

from arms.combined import CombinedArm
from tdmpc.agent import TDMPC2Model
from tdmpc.diagnostics import model_report
from wm.chunk_proposer import ChunkProposer
from wm.episode_bandit import EpisodeBandit
from wm.tree_selector import TreeSelector
from wm.visit_memory import VisitMemory


class TreeExploreArm(CombinedArm):
    name = 'tree_explore'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        t = self.config.tree
        assert t.num_dyn >= 2 or t.beta == 0.0, \
            'disagreement needs a dynamics ensemble: set tree.num_dyn >= 2'
        self._model_updates = 0
        self._last_ref_metrics = {}
        # Loading a model checkpoint goes through
        # arm.model.load_state_dict_all in the loop; wrap it so the
        # proposer's weights ride along when the file has them (old
        # checkpoints don't -- tolerated).
        original_load = self.model.load_state_dict_all

        def load_with_proposer(state):
            original_load(state)
            if 'proposer' in state:
                self.proposer.load_state_dict(state['proposer'])
        self.model.load_state_dict_all = load_with_proposer

    def build_model(self):
        # num_dyn passed as a constructor override so the shared tdmpc
        # block keeps the reference default (1 head) for every other arm.
        self.model = TDMPC2Model(self.obs_dim, self.action_dim, self.device,
                                 self.config.tdmpc, self.gamma,
                                 chunk_len=self.chunk_len,
                                 num_dyn=self.config.tree.num_dyn)
        return self.model

    def build_selector(self):
        t = self.config.tree
        self.proposer = ChunkProposer(
            self.config.tdmpc.latent_dim, self.chunk_len * self.action_dim,
            self.device, hidden=t.proposer_hidden, lr=t.proposer_lr)
        self.memory = VisitMemory(
            self.obs_dim, self.device, offline_size=t.memory_offline,
            online_size=t.memory_online, k=t.memory_k)
        self.bandit = EpisodeBandit(t.bandit_window, t.bandit_c) if t.bandit else None
        return self._make_selector()

    def _make_selector(self):
        t = self.config.tree
        return TreeSelector(
            self.model, self.policy, self.action_dim, self.chunk_len,
            self.chunk.select_n, self.gamma, self.device,
            candidate_source=self.chunk.candidate_source, tree=t,
            proposer=self.proposer, memory=self.memory, bandit=self.bandit,
            seed=self.config.seed)

    def attach_replay(self, replay):
        n = self.memory.fill_offline(replay, self.rng)
        print(f'Visit memory seeded with {n} offline observations')

    def describe(self):
        t = self.config.tree
        return (f'{self.name}: combined best-of-{self.chunk.select_n} + tree '
                f'(beta {t.beta}, depth {t.depth}, beam {t.beam}, '
                f'branch {t.branch}, root_bc {t.root_bc}, '
                f'trust_cap {t.trust_cap}, reach_cap {t.reach_cap}, '
                f'{t.num_dyn} dyn heads, bandit '
                + ('on' if t.bandit else 'off') + ')')

    def model_update(self, replay, metrics_on):
        # Note build_model runs before build_selector in Arm.__init__, so
        # the selector always holds the real model.
        t = self.config.tdmpc
        w = replay.sample_model_windows(t.batch_size, t.horizon, self.device,
                                        self.rng, online_frac=t.online_frac)
        if w is None:
            return {}
        metrics = self.model.update(w['obs'], w['next_obs'], w['action'],
                                    w['reward'], w['mask'], w['valid'],
                                    metrics_on=metrics_on)
        proposer_loss = self.proposer.train_step(self.policy, self.model,
                                                 w['obs'][:, 0])
        self._model_updates += 1
        if self._model_updates % int(self.config.tree.ref_every) == 1:
            ref = self.selector.update_trust_reference(
                replay, self.rng, t.online_frac)
            if ref:
                self._last_ref_metrics = ref
        if not metrics_on:
            return metrics
        metrics = dict(metrics)
        metrics['proposer_loss'] = proposer_loss
        metrics['visit_memory_size'] = float(len(self.memory))
        if self.memory.typical_distance is not None:
            metrics['visit_typical_distance'] = self.memory.typical_distance
        metrics.update(self._last_ref_metrics)
        return metrics

    def report(self, replay):
        t = self.config.tdmpc
        if t.diag_windows <= 0:
            return {}
        return model_report(self.model, self.policy, replay, self.chunk_len,
                            t.diag_depth, self.gamma, self.device, self.rng,
                            num_windows=t.diag_windows)

    def save(self, out_dir, tag):
        torch.save(self.policy.state_dict_all(), out_dir / f'chunk_{tag}.pt')
        state = self.model.state_dict_all()
        state['proposer'] = self.proposer.state_dict()
        torch.save(state, out_dir / f'model_{tag}.pt')
