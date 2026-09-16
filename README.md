# QC-FQL + critic best-of-N (combined), + novelty bonus (wm_explore), QC, QC-FQL

Four offline-to-online RL arms on OGBench cube tasks, one training loop
(`sac_chunked/experiment.py`), one replay, same protocol as Li et al. 2025
(*Reinforcement Learning with Action Chunking*): 1M offline updates on the
play dataset, 1M online env steps, 50 eval episodes.

| script | arm | what it does | published number (cube-triple task4) |
|---|---|---|---|
| `train_combined.py` | **combined** | QC-FQL training; at act and eval time the online critic picks the best of `chunk.select_n` chunks sampled from the distilled one-step actor (`candidate_source=actor`) or the flow BC policy (`=bc`). This is our method. | -- |
| `train_qc_fql.py` | **qc_fql** | QC-FQL exactly (Alg. 2): the one-step actor's single output. The combined arm with `select_n=1`. Paper alpha for triple is 100. | 26 |
| `train_qc.py` | **qc** | QC (Alg. 1): flow BC policy, best-of-32 by the critic at act, eval and TD-target time. PyTorch port of `ColinQiyangLi/qc`; deviations listed in `sac_chunked/qc_agent.py`. `chunk.qc_target=single` bootstraps on one BC sample instead. | 54 |
| `train_wm_explore.py` | **wm_explore** | the combined arm plus a bandit-gated dynamics-disagreement bonus on act-time selection during online collection only (was `train_explore.py`). Training, targets and eval are the combined arm's; `wm_explore.beta=0` is exactly the combined arm. | -- |
| `train_official.py` | -- | runs the vendored official JAX code (`baselines/qc`) with the same flags, to validate the port. | -- |

`tdmpc/` is the latent world-model architecture (trained by `wm_explore`
only, for its dynamics-ensemble disagreement) and `wm/` is the chunk
selector.

## Commands
See `COMMANDS.txt`. Every arm takes the same `--general.*` and `--chunk.*`
flags; `--seed` sets init, batch order and env resets. Runs land in
`runs/seed-task-testing/<run_name>` and in the `seed-task-testing` wandb
project.

## Speed
Eval is most of the wall clock: each eval is `eval_episodes` x 1000 env steps
plus a critic call per chunk, plus the video render. Defaults are unchanged
(`eval_every: 10000`, `eval_episodes: 20`); `--general.eval_every=100000` is
QC's cadence if a run needs to be fast. First line of a run must read
`PyTorch device: cuda`; a CPU warning is printed otherwise (see `install.sh`).

## Install
`bash install.sh` -- installs torch from the CUDA index (never plain PyPI,
never next to `jax[cuda12]`) and the MuJoCo rendering libs.

## Static checks
`python -m pyflakes .`, `python -m pytest -q tests`.
