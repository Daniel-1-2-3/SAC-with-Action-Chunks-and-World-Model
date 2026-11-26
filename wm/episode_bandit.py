""" Sliding-window upper-confidence-bound bandit over two per-episode arms.

    Moved out of the old selector's begin_episode unchanged in behaviour:

        score_k = mean return of arm k in window / range of returns in window
                  + c * sqrt(2 ln N / n_k)

    mean_k the mean return of the episodes that ran arm k in the window,
    range = max - min of all returns in the window (so c is dimensionless),
    N the window size in use, n_k arm k's pulls in it. An arm with no pulls
    in the window is taken first (arm 0, then arm 1). """

import numpy as np


class EpisodeBandit:

    def __init__(self, window=40, c=1.0):
        self.window = int(window)
        self.c = float(c)
        self._pulls = []            # (arm, return) of the last finished episodes
        self.arm = 1                # arm in force until the first choose()

    def choose(self, acc=None):
        """ Pick this episode's arm. acc, when given, is the selector's
            _acc; every call logs five bandit_* keys through it (prefixed
            select/ by pop_stats). """
        window = self._pulls[-self.window:]
        n_total = len(window)
        means, counts = {}, {}
        for k in (0, 1):
            rs = [r for a, r in window if a == k]
            counts[k] = len(rs)
            if rs:
                means[k] = float(np.mean(rs))
        untried = [k for k in (0, 1) if counts[k] == 0]
        gap = None
        if untried:
            arm = untried[0]
        else:
            rets = np.asarray([r for _, r in window])
            rng_ = max(float(rets.max() - rets.min()), 1e-8)
            score = {k: means[k] / rng_ + self.c * np.sqrt(2.0 * np.log(n_total) / counts[k])
                     for k in (0, 1)}
            arm = 1 if score[1] >= score[0] else 0
            gap = score[1] - score[0]
        self.arm = int(arm)
        if acc is not None:
            acc('bandit_arm', float(arm))
            acc('bandit_n_window', float(n_total))
            if 0 in means:
                acc('bandit_mean_exploit', means[0])
            if 1 in means:
                acc('bandit_mean_explore', means[1])
            if gap is not None:
                acc('bandit_ucb_gap', float(gap))
        return self.arm

    def report(self, ep_return):
        """ Called once per finished REAL online episode with its return. """
        self._pulls.append((self.arm, float(ep_return)))
