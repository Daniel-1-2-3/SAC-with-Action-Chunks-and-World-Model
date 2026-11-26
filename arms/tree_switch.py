""" TREE-SWITCH arm: the same tree as tree_explore, but the strength of
    the bonus is set PER DECISION by a stuck score read from the critic,
    instead of being fully on for a whole episode.

    The per-episode bandit chooses between "switch disabled" (amount 0
    everywhere: the pure combined method, which also gives the switch
    clean, unbiased statistics) and "switch enabled" (amount =
    tree.beta x stuck score, recomputed at every chunk boundary). The
    stuck score ranks two critic readouts against recent history:
    expected progress (current value minus the value of making no
    progress forever) and broken promise (what the critic promised
    lookback_chunks decisions ago minus what was delivered since). See
    wm/tree_selector.py TreeSwitchSelector. switch.signal=constant with a
    fixed stuck score is the timing control. Everything else --
    model, proposer, memory, tree, training, eval -- is tree_explore's. """

from arms.tree_explore import TreeExploreArm
from wm.tree_selector import TreeSwitchSelector


class TreeSwitchArm(TreeExploreArm):
    name = 'tree_switch'

    def _make_selector(self):
        t = self.config.tree
        return TreeSwitchSelector(
            self.model, self.policy, self.action_dim, self.chunk_len,
            self.chunk.select_n, self.gamma, self.device,
            candidate_source=self.chunk.candidate_source, tree=t,
            proposer=self.proposer, memory=self.memory, bandit=self.bandit,
            seed=self.config.seed, switch=self.config.switch)

    def describe(self):
        sw = self.config.switch
        return (f'{super().describe()[:-1]}, switch signal {sw.signal}, '
                f'lookback {sw.lookback_chunks} chunks, '
                f'rank window {sw.rank_window})')
