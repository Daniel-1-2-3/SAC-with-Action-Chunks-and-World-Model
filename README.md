# World models for action-chunked RL (QC-FQL), cube-triple

Three arms. Every arm trains the SAME policy: QC-FQL (chunked flow
Q-learning, ported from ColinQiyangLi/qc), observation-space actor and
critic, real chunk transitions from one shared replay buffer. The explore
arm also trains a TD-MPC2 latent model on that same replay (encoder ->
latent, dynamics in latent, reward head and Q both reading the latent,
latent policy prior; nothing decodes to observation space) and uses it in
exactly ONE way:

| script | arm | what happens at a chunk boundary | partner |
|---|---|---|---|
| `train_qcfql_bon.py` | **qcfql_bon** | QC-FQL (Li et al. 2025, Alg. 2) with the online critic picking the best of `select_n` chunks from the distilled one-step actor, at act and eval time. `--chunk.select_n=1` is plain QC-FQL. The paper's control | -- |
| `train_explore.py` | **explore** | the same critic best-of-N, plus a novelty bonus scaled by the critic's own doubt: `Q_i + beta * g * sigma_Q * s~_i * nu_i`, novelty `nu` in `[0, nu_cap]` from dynamics-ensemble disagreement (Pathak et al. 2019), gate `g` from learning progress. Zero where the critic is sure, so it converges to the control by itself | qcfql_bon |

The method of each arm is in `arms/<name>.py`; the shared loop is
`sac_chunked/experiment.py`; the model is `tdmpc/` with its training and
diagnostics plumbing in `arms/model_arm.py`; the buffer is
`sac_chunked/replay.py`; the selector is `wm/chunk_selector.py`. One
`configs.yaml` holds the QC block and the model block ONCE, so arms cannot
drift apart; the explore arm's own knobs sit in its `explore` block.

## Is the model contributing anything?

Read these before anything else on an explore run:

- critic side: `select/frac_within_unc` (candidates the critic cannot
  separate from its favourite; 1/n = sure, 1 = no opinion),
  `select/unc_over_gap` (error bar over its own margin). Both should fall
  over training; that fall IS the anneal. Cross-check with
  `diagnosis/critic_calibration` (ensemble spread over real TD error): near
  0 means the heads agree but are wrong, so the bonus is silenced for the
  wrong reason.
- bonus side: `select/pick_changed`, `select/bonus_over_gap` (how much of
  the critic's margin novelty could buy), `select/picked_unc`,
  `select/picked_novelty`, `select/progress_g` (the gate).
- novelty side: `select/novelty_mean`, `select/novelty_frac_active` (~0 =
  bonus dead), `select/novelty_frac_saturated` (~1 = bonus is a constant),
  `diagnosis/wm_data_disagreement` (the denominator; drift moves both).
  `select/unc_novelty_corr` > 0 means critic doubt and model novelty point
  at the same candidates. `select/model_changed` is the pick changing
  because of the model, over and above the critic's own doubt.
- `wm/reward_corr`, `wm/latent_drift_rel`, `wm/value_critic_corr`  model
  accuracy against replay at every eval, zero env steps
  (`tdmpc/diagnostics.py`).

With wandb disabled, every eval prints the `select/` means since the last
log step as an `attribution:` line.

Four knobs decide what "novel" means; their defaults are the fixed
versions, and the earlier behaviour is one flag each:

| knob | default | earlier | what it fixes |
|---|---|---|---|
| `tdmpc.ref_mode` | `rollout` | `step` | the reference disagreement is measured on real windows the same way a candidate is (path or end of an imagined chunk), instead of one step from a real state. Imagined drift is in both numerator and denominator, so an in-distribution candidate scores ~1, not >1 everywhere |
| `tdmpc.reward_weight_shrink` | `0.5` | `0.0` | `novelty=reward` weights blended toward uniform, so a peaked reward head cannot hand novelty to a few dims' ensemble noise |
| `tdmpc.online_frac` | `0.5` | `0.0` | the model's own balanced sampling: half of every model batch starts in the online region, so visited states stop being novel. The policy's `chunk.online_frac` is untouched |
| `explore.use_rel_unc` | `false` | `true` | drops the critic's relative-doubt factor from the bonus; with two heads it halved the bonus on random candidates |

Passing all four earlier values reproduces the pre-fix run bit for bit.

## Running

Commands in COMMANDS.txt. Always pass `--general.run_name`. Static
checks, no training: `python -m pyflakes .` (outside `embodied/`) and
`python -m pytest tests/`.

Metric decided in advance for cube: steps until eval/mean_return first
crosses a fixed threshold both arms clearly pass. Log the 2/3-cube crossing
and the 3/3 crossing separately.
