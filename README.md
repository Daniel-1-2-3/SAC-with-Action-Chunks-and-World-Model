# World models for action-chunked RL (QC-FQL), cube-triple

Five arms. Every arm trains the SAME policy: QC-FQL (chunked flow
Q-learning, ported from ColinQiyangLi/qc), observation-space actor and
critic, real chunk transitions from one shared replay buffer. Four of them
also train a TD-MPC2 latent model on that same replay (encoder -> latent,
dynamics in latent, reward head and Q both reading the latent, latent
policy prior; nothing decodes to observation space). Each arm uses the
model in exactly ONE way, and differs from its comparison partner in
exactly that:

| script | arm | what the model does | partner |
|---|---|---|---|
| `train_sac_chunked.py` | **control** | nothing. Critic ranks `select_n` candidate chunks (QC's own best-of-N). `--chunk.select_n=1` = plain QC-FQL | -- |
| `train_sac_chunked_ranking.py` | **ranking** | ranks the same candidates by `sum gamma^t r_model + gamma^H Q_model` | control |
| `train_sac_chunked_mve.py` | **mve** | replaces the critic's bootstrap with a latent value expansion (Feinberg et al. 2018). Acting is the control's | control |
| `train_sac_chunked_explore.py` | **explore** | ranking + dynamics-ensemble disagreement bonus (Pathak et al. 2019) | ranking |
| `train_sac_chunked_optimistic.py` | **optimistic** | ranking with an optimistic (RBMLE) dynamics loss (Mete et al. 2026) | ranking |

The method of each arm is in `arms/<name>.py`; the shared loop is
`sac_chunked/experiment.py`; the model is `tdmpc/`; the buffer is
`sac_chunked/replay.py`; the selector is `wm/chunk_selector.py`. One
`configs.yaml` holds the QC block and the model block ONCE, so arms cannot
drift apart; each arm's own knobs sit in its own block.

## Is the model contributing anything?

Read these before anything else on a model-arm run:

- `select/term_r_share`  reward term's share of the score's spread across
  candidates. Near 0: the reward head is irrelevant to the ranking.
- `select/term_q_std`  spread of the terminal latent value.
- `select/pick_agreement`, `select/model_critic_corr`  how often / how
  closely the model picks what the critic would. Near 1: the arm has
  reduced to the control.
- `mve/corr`, `mve/abs_gap` (mve arm)  is the expansion informative, or a
  noisier copy of the QC bootstrap.
- `select/bonus_only_agree` (explore arm)  is the bonus driving the pick.
- `wm/reward_corr`, `wm/latent_drift_rel`, `wm/value_critic_corr`  model
  accuracy against replay at every eval, zero env steps
  (`tdmpc/diagnostics.py`).

## Running

Commands in COMMANDS.txt. Always pass `--general.run_name`.

Fast sanity race on the toy pusher (`toy/point_push.py`, CPU, minutes):

    python toy/race.py --arms critic ranking mve explore optimistic --seeds 0 1

Metric decided in advance for cube: steps until eval/mean_return first
crosses a fixed threshold both arms clearly pass. Log the 2/3-cube crossing
and the 3/3 crossing separately.
