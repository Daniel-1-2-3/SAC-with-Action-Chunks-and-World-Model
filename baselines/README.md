# Official QC code, for validating the port

`baselines/qc` is github.com/ColinQiyangLi/qc as a git submodule, pinned
at commit 48283b4 (Feb 2026). Nothing inside it is edited. Its one job here
is to check `train_qc.py` (our PyTorch port of `agents/acfql.py` with
`actor_type=best-of-n`) and, as a second reference point, the QC-FQL policy
behind `train_control.py --chunk.select_n=1`.

## Setup

    bash baselines/setup_qc_venv.sh        # submodule + .venv-qc (JAX stack)

The official requirements pin `jax==0.6.0` with the CUDA 12 plugin and
`numpy==2.2.5`, which clash with this repo's PyTorch/`numpy<2` stack, so
they live in their own interpreter. Both stacks read OGBench datasets from
`~/.ogbench/data`, so the data is downloaded once.

## Running

    python train_official.py --baseline qc    --general.env_name=cube-triple-play-singletask-task4-v0 --general.num_offline_steps=1000000 --general.eval_episodes=50 --general.run_name=t4_official_qc_s0 --seed=0
    python train_official.py --baseline qcfql --general.env_name=cube-triple-play-singletask-task4-v0 --general.num_offline_steps=1000000 --general.eval_episodes=50 --general.run_name=t4_official_qcfql_s0 --seed=0

`--dry_run` prints the `main.py` command without running it. Smoke test
(2k steps, minutes, both baselines):

    for b in qc qcfql; do python train_official.py --baseline $b --general.env_name=cube-triple-play-singletask-task4-v0 --general.num_offline_steps=1000 --general.num_online_steps=1000 --general.start_training=100 --general.eval_every=1000 --general.eval_episodes=2 --general.wandb_mode=disabled --general.run_name=smoke_$b --general.out_dir=runs/official_smoke; done

## Flag mapping

Our flags are read exactly as `train_*.py` read them (configs.yaml
`defaults`, `--configs` presets, dotted overrides) and mapped to `main.py`
and `agents/acfql.py::get_config()`:

| ours | official | note |
|---|---|---|
| `--baseline qc` | `--agent.actor_type=best-of-n --agent.actor_num_samples=<chunk.qc_num_samples>` | the paper's QC; `agent.alpha` unused |
| `--baseline qcfql` | `--agent.actor_type=distill-ddpg --agent.alpha=<chunk.alpha>` | the paper's QC-FQL |
| `seed` | `--seed` | |
| `general.env_name` | `--env_name` | OGBench `*-singletask-*` names only |
| `general.run_name` | `--run_group` | `reproduce` if empty |
| `general.out_dir` | `--save_dir` | main.py appends `qc/<run_group>/<env>/sd<seed>_<time>/` |
| `general.num_offline_steps` | `--offline_steps` | |
| `general.num_online_steps` | `--online_steps` | |
| `general.eval_every` | `--eval_interval` | main.py also evals at `offline_steps - 1` and `online_steps - 1` |
| `general.eval_episodes` | `--eval_episodes` | |
| `general.log_every` | `--log_interval` | |
| `general.start_training` | `--start_training` | |
| `general.wandb_mode` | `WANDB_MODE` env var | `disabled` / `offline` / `online` |
| `general.wandb_project` | `WANDB_PROJECT` env var | main.py hardcodes project `qc`; the env var is informational |
| `chunk.chunk_len` | `--horizon_length` | |
| `chunk.gamma` | `--discount` | |
| `chunk.utd_ratio` | `--utd_ratio` | |
| `chunk.batch_size` | `--agent.batch_size` | |
| `chunk.lr` | `--agent.lr` | |
| `chunk.hidden_dim`, `chunk.num_layers` | `--agent.actor_hidden_dims`, `--agent.value_hidden_dims` | `(hidden_dim,) * num_layers` |
| `chunk.critic_target_tau` | `--agent.tau` | |
| `chunk.ensemble` | `--agent.num_qs` | |
| `chunk.q_agg` | `--agent.q_agg` | |
| `chunk.flow_steps` | `--agent.flow_steps` | |
| (always) | `--sparse=False` | cube tasks are dense in both code bases |
| `chunk.replay_capacity` | not mapped | main.py keeps its `--buffer_size=2000000`, `max(buffer_size, dataset + 1)`; ours is `max(capacity, dataset + online + 1)`. Both never evict offline data on a 1M + 1M run |
| `chunk.online_frac`, `chunk.select_n`, `chunk.compile_nets`, `chunk.eef_slice`, `tdmpc.*`, `explore.*` | none | not in the official code |

## Metric mapping

Official `eval.csv` (one row per evaluation, `step` = offline + online
steps so far) against our `eval_log.csv`:

| official `eval.csv` | ours `eval_log.csv` | note |
|---|---|---|
| `step` | `env_step` | same counter: offline updates, then online env steps |
| `success` | `success_rate` | mean over eval episodes of OGBench `info['success']` |
| `episode.return` | `mean_return` | raw env rewards, `{-n..0}` per step |
| `episode.length` | `mean_episode_len` | |
| -- | `coherence` | ours only (end-effector displacement over 5 steps) |
| `avg_gripper_contact_length`, `num_gripper_contacts` | -- | official only |

Training logs: official `offline_agent.csv` / `online_agent.csv`
(`critic/critic_loss`, `critic/q_mean`, `actor/bc_flow_loss`,
`actor/distill_loss`) against ours under `sac/` (`critic_loss`,
`critic_q`, `bc_flow_loss`, `distill_loss`; wandb only).

`scripts/plot_vs_paper.py` reads our CSV layout; to overlay an official
run, convert its `eval.csv` with the two column renames above.

## Known differences of the official loop

- main.py writes `token.tk` with `wandb.run.url` at the very end; with
  `WANDB_MODE=disabled` that value is `None` and the write raises after
  all training and every `eval.csv` row are done. Ignore that traceback.
- Video and `save_interval` are off; pass `--dry_run` and append flags by
  hand if you need them.
