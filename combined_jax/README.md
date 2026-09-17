# combined arm, fully in JAX

The official QC codebase ([ColinQiyangLi/qc](https://github.com/ColinQiyangLi/qc)
at `48283b4`, MIT -- see LICENSE) with one addition: the combined method's
best-of-N candidate selection at act/eval time. Training is byte-for-byte
the official QC-FQL; the only new code is in `agents/acfql.py` under
`COMBINED PATCH` comments:

- `--agent.actor_num_candidates=N` (this repo's `chunk.select_n`): in
  distill-ddpg mode, N candidate chunks are drawn per decision, the ONLINE
  critic scores each (`q_agg` over the ensemble) and the argmax is executed
  -- at act time and at eval time. `1` (the default) is plain QC-FQL,
  upstream-identical.
- `--agent.candidate_source=actor|bc` (`chunk.candidate_source`): candidates
  from the distilled one-step actor (default) or the flow BC policy.
- The TD target always bootstraps on ONE plain one-step-actor sample at s'
  (never the act-time best-of-N), so training stays exactly QC-FQL's.

## What was vendored, and how it differs from upstream

`main.py`, `evaluation.py`, `log_utils.py`, `agents/acfql.py`,
`envs/{env_utils,ogbench_utils}.py`, `utils/{datasets,flax_utils,encoders,networks}.py`.
Differences, all marked with `COMBINED PATCH` comments:

- `agents/acfql.py`: the combined-arm patch described above.
- `main.py`: `is_robomimic_env` inlined (robomimic/d4rl utils not vendored;
  OGBench is the only supported env family here).
- `agents/__init__.py`: registers acfql only (acrlpd not vendored).
- `utils/networks.py`: the distrax-dependent classes acfql never uses are
  deleted, so distrax/tensorflow-probability are not dependencies.

## Install

```bash
pip install -U "jax[cuda12]" flax optax ml_collections wandb tqdm pillow ogbench
```

## Run (from inside combined_jax/)

```bash
# combined arm (this repo's train_combined.py, alpha 300 = its config default)
MUJOCO_GL=egl python main.py --run_group=combined --agent.alpha=300 \
  --agent.actor_num_candidates=16 \
  --env_name=cube-triple-play-singletask-task3-v0 --sparse=False --horizon_length=5

# plain QC-FQL control (upstream-identical; paper alpha for cube-triple is 100)
MUJOCO_GL=egl python main.py --run_group=qcfql --agent.alpha=100 \
  --env_name=cube-triple-play-singletask-task3-v0 --sparse=False --horizon_length=5
```

Protocol flags (`--offline_steps`, `--online_steps`, `--eval_episodes`, ...)
are upstream's, with upstream defaults (1M offline, 1M online, 50 eval
episodes, eval every 100k). Note the PyTorch loop in the repo root differs
from these defaults in eval cadence (10k) and episode count (20), so
compare curves at matching eval settings.
