# Tree exploration for the combined method — hand-back report

Branch: `tree-explore` (from `b9981bf`). Final commit: the commit that adds
this report (the tip of `tree-explore`; `main` is fast-forwarded to it, as
Daniel asked in-session after the smoke runs passed).
Overall: **everything passed** — A1 through A12, with the machine caveat in
section 3 (no GPU on this container; every check ran on CPU).

To use the branch on a pod:

```
git fetch origin && git checkout tree-explore && git reset --hard origin/tree-explore
```

(`main` carries the same commit.)

## 1. Files

New files:

* `arms/tree_explore.py` — the tree_explore arm
* `arms/tree_switch.py` — the tree_switch arm
* `train_tree_explore.py`, `train_tree_switch.py` — entry points
* `wm/tree_selector.py` — TreeSelector and TreeSwitchSelector
* `wm/visit_memory.py` — the visit memory
* `wm/chunk_proposer.py` — the latent-space chunk proposer
* `wm/episode_bandit.py` — the sliding-window UCB bandit (moved out of the old selector, behaviour unchanged)
* `scripts/check_tree_drift.py` — the drift pre-check
* `tests/test_tree_explore.py` — unit tests A2, A4–A9

Deleted files:

* `arms/wm_explore.py`, `train_wm_explore.py`
* `combined_jax/wm/__init__.py`, `combined_jax/wm/selector.py`, `combined_jax/wm/tdmpc2.py`

Changed files, two lines each:

* `wm/chunk_selector.py` — `score_candidates` split out of `select()` (identical draw, scoring and RNG use); everything belonging only to the old explorer removed. Kept: critic best-of-N, `candidate_source`, `eval_mode`, episode-start value capture, `begin_episode` (reduced), `pop_stats`, `_acc`, the `model` attribute.
* `tdmpc/agent.py` — old novelty machinery removed (`reward_weights`, `reduce_disagreement`, `path_disagreement`, `update_novelty_reference`, `ref_mode`, `reward_weight_shrink`, `data_disagreement`, the `var0` plumbing in `update()`/`_losses`); `imagine_chunk` added (rollout_chunk without the reward pooling, built on `_step`, compiled with the other scoring paths). `tdmpc/model.py` untouched.
* `tdmpc/diagnostics.py` — `wm/unc_err_corr` now measured with `imagine_chunk` over the window's real actions in chunk_len pieces; same mean-over-steps meaning, nothing else changed.
* `sac_chunked/experiment.py` — two hasattr-guarded hooks (`arm.attach_replay`, `selector.observe_step`); with wandb disabled every logged dictionary is also written as one JSON line to `out_dir/metrics.jsonl` (pure output, kept deliberately).
* `configs.yaml` — `wm_explore` block, `ref_mode`, `reward_weight_shrink` removed; `tree:` and `switch:` blocks added; the `q_mode` comment names `arms/tree_explore.py` now.
* `COMMANDS.txt` — each `train_wm_explore.py` line replaced by a `train_tree_explore.py` and a `train_tree_switch.py` line, same flags, run names `..._tree_explore_s{seed}` / `..._tree_switch_s{seed}`.
* `combined_jax/main.py`, `combined_jax/agents/acfql.py`, `combined_jax/utils/datasets.py`, `combined_jax/README.md` — restored to their pre-explorer state (commit `84f96a5`), so the non-explorer JAX behaviour is unchanged line for line; `acfql.sample_candidates` had no remaining reference and went with it.

## 2. Acceptance results

* **A1 — pass.** The two before-runs (`b9981bf`, pristine worktree) were
  bit-identical to each other, and the after-run matched them on every
  `eval_log.csv` column except wall time (coherence digit for digit):
  `step 1300: return -3000.0, success 0.0, coherence 0.390964` and
  `step 2300: return -3000.0, success 0.0, coherence 0.390964` in all three.
  Run-to-run noise on this CPU is zero, and the bar was met exactly.
* **A2 — pass** (pytest): same pick, same eval pick, and identical global
  torch RNG state between `TreeSelector` (amount 0) and `ChunkSelector`.
* **A3 — pass**: `smoke_tree_beta0` logged `select/tree_grown_frac` with
  mean 0 and max 0 over the whole run.
* **A4–A8 — pass** (pytest; 11 tests total, all green, ~3 s).
* **A9 — pass**: an old-shaped `{'net': ...}` state dictionary saved from
  the pristine `b9981bf` code loads into the new `TDMPC2Model(num_dyn=4)`;
  the drift pre-check also loaded `smoke_tree_nobandit/model_final.pt`,
  which carries the new `proposer` key, through the same path.
* **A10 — pass.** All eight runs finished exit 0 on the real OGBench
  simulator (CPU — see section 3). Required metrics, all non-empty and
  finite in runs 2–8: `wm/tree_reference_d1` (~5e-5 by run end),
  `wm/tree_drift_d1` before the first memory refresh and
  `wm/tree_drift_ratio_d1` after (~0.8), `wm/proposer_loss` (~0.025 by run
  end), `wm/visit_typical_distance` (~0.63). `select/tree_grown_frac`
  means: run 2 = 0.50, run 3 = 0.50, run 4 = 0.4975, run 6 = 0.2475, run 8
  = 1.00 (all > 0 as required), run 7 = exactly 0. `select/reach_picked`,
  `select/trusted_frac_d1` (0.93–0.98) and `select/tree_milliseconds`
  present in runs 2, 3, 4, 6, 8; `select/switch_stuck` in runs 4 (mean
  0.939 — flat smoke-run returns rank almost everything as stuck, as
  expected) and 6 (exactly 0.500, the constant); `select/bandit_arm` in
  runs 2–7 (mean 0.5, the two arms alternating). Run 8's
  `select/eval_tree_grown` stayed 0 and its eval rows evaluate on the
  plain path. The compiled run proved `imagine_chunk` and the proposer
  compile (log lines present). Median `select/tree_milliseconds` of run 3:
  **157.7 ms on CPU** (mean 814 ms, dominated by the first tree, which
  pays the inductor compilation). The 30 ms target is a GPU target and
  could not be assessed here. Throughput (online env steps per second from
  `eval_log.csv` wall times): `smoke_combined` 8.14, `smoke_tree_nobandit`
  1.54 — again CPU numbers. Drift pre-check on run 8's checkpoint printed
  the full table (4,000 model updates only, so the numbers are not
  meaningful; the disagreement-drift correlation was already 0.50–0.57):
  depth 1..4 drift ratio means 1.16 / 1.74 / 2.28 / 2.74, deepest depth
  with mean < 1: 0. No old `wm_explore` checkpoint exists on this fresh
  container, so the real "how deep can the tree trust the model" table
  still needs one run of `scripts/check_tree_drift.py` on a pod that has
  an old `model_*.pt`.
* **A11 — pass**: the Part 2.8 grep returns nothing.
* **A12 — pass**: `combined_jax/main.py` parses; no `use_wm` anywhere under
  `combined_jax/`.

## 3. Machine setup (read this first)

The brief describes a RunPod pod with an NVIDIA RTX PRO 4500. **This
session ran in a cloud container with no GPU at all** (`nvidia-smi` not
present; `torch.cuda.is_available()` False). Everything — unit tests, the
A1 runs, all eight smoke runs, the drift pre-check — therefore ran **on
CPU** (4 cores, 15 GB RAM). They are real OGBench simulator runs of the
real training code; only the device differs, so treat every wall-clock
number above as a CPU number.

Setup that mattered (the container's `pip` targeted Python 3.13 while
`python` was 3.11, hence the relink):

```
ln -sf /usr/local/bin/python3.13 /usr/local/bin/python
ln -sf /usr/local/bin/python3.13 /usr/local/bin/python3
pip install 'numpy<2' 'elements>=3.17.0' ruamel.yaml mujoco gymnasium \
    opencv-python-headless requests imageio pytest
pip install dm-tree dm-env glfw lxml pyopengl pyparsing absl-py protobuf tqdm
pip install 'scipy==1.14.1'        # the preinstalled scipy needed numpy 2
pip install --no-deps dm_control ogbench   # labmaze has no wheel here; dm_control's
                                           # mjcf (all ogbench needs) works without it
pip install --no-cache-dir torch   # download.pytorch.org is blocked by the
                                   # container's network policy; the PyPI wheel works on CPU
apt-get update && apt-get install -y libegl1 libosmesa6 libgl1
```

* `bash install.sh` could not be used as written: `download.pytorch.org`
  is blocked by this container's network policy. A cloud environment named
  `SAC-with-Action-Chunks-and-World-Model` was created allowing that host
  and `rail.eecs.berkeley.edu` for future sessions; this running session
  could not be moved onto it.
* `MUJOCO_GL`: **egl works** once `libegl1` is installed (without it,
  import fails with an EGL AttributeError). osmesa and glfw also import.
  All runs used the scripts' default egl; every `setdefault` line is in
  place, the new entry points included.
* Dataset: `rail.eecs.berkeley.edu` is blocked here, so per the brief's
  fallback the dataset was **regenerated** with OGBench's own
  `data_gen_scripts/generate_manipspace.py` (scripted plan oracle, no
  downloads): 150 train + 5 val episodes of `cube-triple-play-v0`
  (150,000 / 5,000 transitions), stored in `/workspace/ogbench_data`,
  symlinked at `~/.ogbench/data`, and loaded through
  `helpers/ogbench_methods.py` exactly as the loop does (rewards −3..0,
  masks present). **All smoke runs used this regenerated file**, not the
  official 3M-transition dataset.
* Smoke outputs live under `/workspace/smoke/` (eval logs and
  `metrics.jsonl` kept; checkpoints deleted after the drift check, per the
  disk rule). No stray python processes remain.

## 4. Bugs found in existing code

None. (Two test failures during development were bugs in the new tests
themselves — an RNG reseeding mistake and a miscounted accumulator — fixed
in place; no existing code was wrong.)

## 5. Launch commands for the five task 4 runs (seed 0, official dataset protocol)

```
OMP_NUM_THREADS=2 nohup python train_combined.py --general.env_name=cube-triple-play-singletask-task4-v0 --general.num_offline_steps=1000000 --general.run_name=t4_combined_s0 --general.out_dir=runs/seed-task-testing/t4_combined_s0 --seed=0 > t4_combined_s0.log 2>&1 &
OMP_NUM_THREADS=2 nohup python train_tree_explore.py --general.env_name=cube-triple-play-singletask-task4-v0 --general.num_offline_steps=1000000 --general.run_name=t4_tree_explore_s0 --general.out_dir=runs/seed-task-testing/t4_tree_explore_s0 --seed=0 > t4_tree_explore_s0.log 2>&1 &
OMP_NUM_THREADS=2 nohup python train_tree_explore.py --tree.depth=1 --general.env_name=cube-triple-play-singletask-task4-v0 --general.num_offline_steps=1000000 --general.run_name=t4_tree_depth1_s0 --general.out_dir=runs/seed-task-testing/t4_tree_depth1_s0 --seed=0 > t4_tree_depth1_s0.log 2>&1 &
OMP_NUM_THREADS=2 nohup python train_tree_switch.py --general.env_name=cube-triple-play-singletask-task4-v0 --general.num_offline_steps=1000000 --general.run_name=t4_tree_switch_s0 --general.out_dir=runs/seed-task-testing/t4_tree_switch_s0 --seed=0 > t4_tree_switch_s0.log 2>&1 &
OMP_NUM_THREADS=2 nohup python train_tree_switch.py --switch.signal=constant --switch.constant_stuck=FILL_ME --general.env_name=cube-triple-play-singletask-task4-v0 --general.num_offline_steps=1000000 --general.run_name=t4_tree_switch_constant_s0 --general.out_dir=runs/seed-task-testing/t4_tree_switch_constant_s0 --seed=0 > t4_tree_switch_constant_s0.log 2>&1 &
```

`FILL_ME`: the logged mean of `select/switch_stuck` from `t4_tree_switch_s0`
once that run has gone long enough. Do not use the smoke run's 0.939 — it
comes from a 6,000-step run on a regenerated dataset whose returns are
flat, which ranks nearly every decision as stuck.

## 6. Decisions made where the brief was ambiguous

* No GPU exists on this machine, so every check ran on CPU, declared here rather than silently.
* The dataset host is blocked, so the smoke dataset was regenerated with OGBench's scripted oracle (the brief's named fallback).
* `COMMANDS.txt` was rewritten in the removal commit (it is Part 2.7) rather than the final commit.
* A depth with no trust-reference measurement yet trusts nothing (the tree stays silent rather than trusting an unmeasured model).
* The trust reference is first measured at the first model update, then every `tree.ref_every` updates; the latest measurement is re-emitted with every logged metrics batch so it is always visible offline.
* Switch ranks are computed against the window excluding the current decision, which is appended afterwards; the `min_history` warm-up check runs after the append.
* The first decision of an episode records its value for later lookbacks even though its amount is 0.
* The proposer is a plain Linear+Mish stack with tdmpc's truncated-normal(0.02)/zero-bias init; checkpoints go through the raw module so compiled and uncompiled runs exchange files.
* TreeSelector's episode-start value capture saves and restores the global torch RNG around the model's value estimate, so the plain path consumes exactly the combined arm's random numbers (the combined arm has no model and consumes none there).
* `select/tree_milliseconds` is measured on every 50th tree (the brief's allowed option).
* The carried chunk is cleared explicitly on eval decisions as the brief words it, although `begin_episode` already made a stale carry unreachable.
* No old `wm_explore` run exists on this fresh container, so the A9 old-shaped checkpoint was synthesized from the pristine `b9981bf` worktree before the edits, exactly as the brief prescribes.
* The bandit's five log keys moved into `EpisodeBandit.choose`, written through the selector's accumulator, keeping the old names.
* The branch was pushed mid-work because a repository stop hook demands uncommitted work be pushed; `main` was fast-forwarded at the end because Daniel asked for that in-session, which supersedes the brief's "never push to main".
