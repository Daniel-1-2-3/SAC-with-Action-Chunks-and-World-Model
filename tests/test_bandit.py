""" explore.controller=bandit: sliding-window UCB over {exploit, explore}
    picked once per real online episode (wm/chunk_selector.py). Pure
    bookkeeping, no networks: the selector is built with model=None,
    policy=None and never asked to select. """

import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from wm.chunk_selector import ChunkSelector  # noqa: E402

BANDIT_KEYS = ['select/bandit_arm', 'select/bandit_n_window', 'select/bandit_mean_exploit',
               'select/bandit_mean_explore', 'select/bandit_ucb_gap']


def make(controller='bandit', window=10, c=1.0, **kw):
    return ChunkSelector(None, None, action_dim=2, chunk_len=5, n=16, gamma=0.99,
                         device='cpu', controller=controller, bandit_window=window,
                         bandit_c=c, **kw)


def play(sel, payoff, episodes, rng):
    """ payoff(arm) -> episode return; returns the list of arms pulled. """
    arms = []
    for _ in range(episodes):
        sel.begin_episode()
        arms.append(sel.bandit_arm)
        sel.report_episode_return(payoff(sel.bandit_arm) + rng.normal(0, 0.1))
    return arms


def test_untried_arms_first_then_gate_follows_arm():
    sel = make()
    sel.begin_episode()
    assert sel.bandit_arm == 0 and sel.gate == 0.0
    sel.report_episode_return(-100.0)
    sel.begin_episode()
    assert sel.bandit_arm == 1 and sel.gate == 1.0


def test_prefers_the_better_arm():
    rng = np.random.default_rng(0)
    sel = make(window=20)
    arms = play(sel, lambda a: -50.0 if a == 1 else -100.0, 60, rng)
    assert np.mean(arms[20:]) > 0.8, arms


def test_sliding_window_forgets():
    rng = np.random.default_rng(1)
    sel = make(window=10)
    play(sel, lambda a: -50.0 if a == 1 else -100.0, 30, rng)
    # payoff flips: exploit is now better; within ~2 windows the pick flips
    arms = play(sel, lambda a: -100.0 if a == 1 else -50.0, 30, rng)
    assert np.mean(arms[15:]) < 0.3, arms


def test_ucb_keeps_sampling_the_worse_arm():
    rng = np.random.default_rng(2)
    sel = make(window=20, c=1.0)
    arms = play(sel, lambda a: -50.0 if a == 1 else -60.0, 200, rng)
    assert 0 < np.mean(arms[20:]) < 1.0, 'UCB should never lock out an arm entirely'


def test_five_keys_logged_and_reset():
    rng = np.random.default_rng(3)
    sel = make()
    play(sel, lambda a: -50.0, 5, rng)
    stats = sel.pop_stats()
    for k in BANDIT_KEYS:
        assert k in stats, (k, sorted(stats))
    assert 0.0 <= stats['select/bandit_arm'] <= 1.0
    assert sel.pop_stats() == {}


def test_gate_controller_untouched():
    sel = make(controller='gate', progress_window=5)
    sel.begin_episode()               # no-op under the gate
    assert sel.gate == 1.0
    for r in [-100.0] * 5 + [-50.0] * 5:
        sel.report_episode_return(r)
    assert sel.gate < 1.0             # returns improved -> gate fell
    assert not any(k.startswith('select/bandit') for k in sel.pop_stats())
