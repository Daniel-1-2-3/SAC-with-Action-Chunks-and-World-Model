# HER roadmap

One change per step. Each step is measured against the previous step's
curve. Nothing is added until the step below it works.

### 1. Check that HER works (DONE)

Their documented task: bitflipping, DQN + HER, checking the relabeling
implementation is correct.

```
python bitflip_check.py
```

Verified: 10 bits reaches 100% with HER and 0% without; 20 bits reaches
88% and still climbing with HER, 0% without. This reproduces the paper's
Figure 1 using the relabeling code imported from `train_her.py`.

### 2. HER + DDPG on OGBench cube-single

```
python train_her.py --env_name cube-single-play-singletask-v0 --out_dir sear_runs/her_ddpg_single --wandb_project sear-her-wm
```

Expected failure mode: the arm never touches the cube, so relabeled goals
are satisfied on arrival and there is no signal. Confirm from the metrics
below. Three branches:

- `her/cube_moved_frac` near 0 -> contact is the problem -> step 3
- cube moves but `her/relabeled_reward_zero_frac` near 1 -> `--thresh`
  (0.04 m) is too coarse for how far the cube travels; lower it
- cube moves and rewards vary -> HER is working, just slow; let it run

Metrics log every cycle (~3,200 steps); eval logs every epoch (160,000
steps). `her/q_max` should sit in the middle of [-50, 0]; pinned at 0
means the agent thinks it is already winning, pinned at -50 means no
relabeled goal is ever reached.

Note this is the paper's single-goal setting (their Sec 4.3, Fig 4),
which is their slow regime: success lifts off around epoch 10-25 and
reaches ~60% by epoch 200. `--epochs` defaults to 30, so raise it before
calling a null result.

### 3. SEAR exploration + DDPG + HER

If the cube not getting bumped was the issue. Take only SEAR's collection
(MaxEnt actions, random-prefix replanning), leave the DDPG+HER learner
alone. The bar is low: SEAR only has to make the arm bump the cube; HER
turns the bump into signal. If it does not work, look for other
exploration solutions.

### 4. SEAR + action-chunked SAC + HER

`train_sear.py` with `use_her=True`. Test the chunk relabeling
independently first, the way bitflip tested single-transition relabeling.

### 5. World model for best-of-N selection at act time

Sample N candidate chunks, imagine each one's cube trajectory, score by
predicted distance to goal, execute the best. No learned reward head --
reward is computed from the predicted cube position with the same
function used everywhere else. `select_n=1` is the exact control.

Precondition to measure first: does the model predict cube position
better than "the cube stays put"? If not, selection cannot work.

## Files

```
train_her.py         steps 1-2: HER + DDPG, standalone
bitflip_check.py     step 1: the paper's Fig 1 check
train_sear.py        steps 3-5: SEAR trainer
configs.yaml         config for train_sear.py
helpers/her.py       step 4: chunk-window relabeling
helpers/common.py    config loading, seeding, metrics
helpers/sear_replay.py
sear/                SEAR agent, critic, window builder
torch_wm/            world model + ensemble (step 5)
```

## Fixes carried in from earlier runs

- `helpers/her.py` — the end-effector was appended to the goal vector, a
  deviation from the paper. Goals are cube-only again.
- `helpers/her.py` — relabeled future goals already satisfied at the
  window start are now rejected. Verified: on a frozen cube the useless
  zero-reward fraction drops from 0.76 to 0.00.
- `helpers/her.py` — goals are raw metres, not rescaled into observation
  space. Removes a second coordinate frame that had to be kept in sync.
- `configs.yaml` — `reward_batch_frac` 0.5 -> 0.15. At 0.5, a single
  early reward episode was over-represented roughly 100x on every update.
- `configs.yaml` — `success_reward_thresh` -1.5 -> -0.5, correct for
  cube-single's 0/-1 reward. Use -1.5 for cube-double.

## Removed

- `explore/` (Plan2Explore explorer) — benched after it lagged SEAR at
  finding partials; not in the roadmap
- `helpers/curriculum.py` (spawn randomization, reset-to-pooled-state)
  — cube spawning was rejected as a direction
- `plan_eval.py` (MPPI planner) — step 5 is best-of-N selection at act
  time, which is a different mechanism
