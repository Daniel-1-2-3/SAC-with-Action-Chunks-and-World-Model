# Critic best-of-N vs TD-MPC2 best-of-N, cube-triple

Two arms, seeds 0 and 1 each. Both pretrain 250k offline then run 1M
online. Both sample select_n=16 candidate chunks per decision from the
same QC-FQL policy. The ONLY difference is what scores the candidates:

- Arm 1 (`train_sac_chunked.py`): the online critic, Q(s, chunk).
  This is QC's own best-of-N. No learned model anywhere. CONTROL.
- Arm 2 (`train_sac_chunked_wm.py`): a TD-MPC2 latent model, trained
  alongside on the same replay and used for scoring only:

      sum_t gamma^t * r_model(z_t, a_t) + gamma^H * Q_model(z_H, pi(z_H))

  Nothing decodes back to observation space. The encoder produces a
  latent, the dynamics stay in it, and the reward head and Q both read
  the latent directly. Reward and value are symlog two-hot, so both
  terms of the score are in the same units as each other and as the QC
  critic. select_rollout_chunks stays 1.

QC-FQL training is IDENTICAL in both arms: observation-space actor and
critic, real chunk transitions from replay, target
R_real + gamma^h * mask * Q(s_next). `_agent_update` takes no model
argument by signature, so the model cannot touch training. Its entire
effect flows through which chunks get executed, i.e. through the data.

Because both arms sample the same 16 candidates from the same policy and
differ only in the scorer, any gap is attributable to the model.

Is the model contributing anything? Four selection metrics answer that
directly, and they are the first thing to read on any run:

- `select/term_r_share`   reward term's share of the score's spread
                          across candidates. Near 0 means the reward head
                          is irrelevant to the ranking.
- `select/term_q_std`     spread of the terminal latent value.
- `select/pick_agreement` how often the model picks the same chunk the
                          critic would. Near 1.0 means this arm has
                          reduced to the control.
- `select/model_critic_corr` the same question as a correlation.

Model accuracy is checked offline against replay at every eval
(`tdmpc/diagnostics.py`, zero env steps): pooled imagined chunk reward
vs real, latent drift relative to the spread between encoded states, and
the model value's correlation with the QC critic.

Metric decided in advance: steps until eval/mean_return first crosses a
fixed threshold both arms clearly pass. Log the 2/3-cube crossing and the
3/3 crossing separately -- the previous single-seed run reached 2/3
earlier than the critic arm but never got the third cube.

Commands in COMMANDS.txt. Always pass run_name.
