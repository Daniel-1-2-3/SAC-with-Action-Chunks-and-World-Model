""" EXPLORE arm: the combined method (control) + a bandit-gated novelty
    bonus at act-time selection during ONLINE COLLECTION only.

    Training is the control's, byte for byte: QC-FQL losses, the critic's
    own eq. 15 target, best-of-N execution. The one behavioral difference
    is the score that picks which of the N candidate chunks gets executed
    while collecting online data:

        score_i = critic value_i
                  + beta_active * spread of critic values * novelty_i

        novelty_i = clamp(imagined disagreement of candidate i's path
                          / disagreement on real data  -  1,  0, nu_cap)

    Disagreement comes from a dynamics ENSEMBLE inside the TD-MPC2 model
    (explore.num_dyn heads; the wm output with the only measured POSITIVE
    this project has: Phase-4 found successes ~2x earlier on triple
    collection and ~300k earlier on quadruple). The bonus never touches a
    target, a reward, a gradient, or eval -- the four measured wm failure
    modes are structurally absent. Eval always scores with the bonus off.

    beta_active is chosen PER EPISODE by a sliding-window UCB bandit over
    the two arms' real episode returns (wm/chunk_selector.py,
    controller='bandit' -- the v5 fix for Phase-4's endpoint tax, built
    Sep 3 and never run): exploit episodes are exactly the control, explore
    episodes carry the full bonus, and whichever is EARNING more return
    keeps getting picked. The loop hooks (selector.begin_episode at online
    episode start, selector.report_episode_return at episode end) already
    live in sac_chunked/experiment.py.

    The model trains alongside on replay windows only (consistency + reward
    + the reference step-mode value/prior; q_mode stays 'step' -- nothing
    consumes the wm's value here, and the chunk-mode head is not needed).
    tdmpc.ref_mode=rollout (v5 fix A) measures the novelty reference by
    imagining real replay actions, matched to how candidates are measured.

    Acceptance property (inherited from the selector): an exploit episode
    (bandit arm 0) scores candidates identically to the control -- gate 0
    zeroes the bonus, leaving the critic's argmax. explore.beta=0 disables
    the machinery entirely. """

from arms.control import ControlArm
from tdmpc.agent import TDMPC2Model
from tdmpc.diagnostics import model_report
from wm.chunk_selector import ChunkSelector


class ExploreArm(ControlArm):
    name = 'explore'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        e = self.config.explore
        assert e.num_dyn >= 2 or e.beta == 0.0, \
            'novelty needs a dynamics ensemble: set explore.num_dyn >= 2'

    def build_model(self):
        # num_dyn passed as a constructor override so the shared tdmpc
        # block keeps the reference default (1 head) for every other arm.
        self.model = TDMPC2Model(self.obs_dim, self.action_dim, self.device,
                                 self.config.tdmpc, self.gamma,
                                 chunk_len=self.chunk_len,
                                 num_dyn=self.config.explore.num_dyn)
        return self.model

    def build_selector(self):
        e = self.config.explore
        return ChunkSelector(
            self.model, self.policy, self.action_dim, self.chunk_len,
            self.chunk.select_n, self.gamma, self.device,
            rollout_chunks=e.rollout_chunks, bonus_beta=e.beta,
            novelty=e.novelty, novelty_at=e.novelty_at, nu_cap=e.nu_cap,
            bonus_scale=e.bonus_scale, progress_gate=e.progress_gate,
            progress_window=e.progress_window, progress_tau=e.progress_tau,
            use_rel_unc=e.use_rel_unc, controller=e.controller,
            bandit_window=e.bandit_window, bandit_c=e.bandit_c,
            candidate_source=self.chunk.candidate_source)

    def describe(self):
        e = self.config.explore
        return (f'{self.name}: control best-of-{self.chunk.select_n} '
                f'+ novelty bonus (beta {e.beta}, {e.bonus_scale}, '
                f'novelty {e.novelty}@{e.novelty_at}, nu_cap {e.nu_cap}, '
                f'{e.num_dyn} dyn heads, controller {e.controller}'
                + (f', window {e.bandit_window}, c {e.bandit_c})'
                   if e.controller == 'bandit' else ')'))

    def model_update(self, replay, metrics_on):
        # Note build_model runs before build_selector in Arm.__init__, so
        # the selector always holds the real model.
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
