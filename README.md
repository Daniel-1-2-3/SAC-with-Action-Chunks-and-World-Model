# Critic best-of-N vs world-model best-of-N, cube-triple

Two arms, seeds 0 and 1 each. Both pretrain 250k offline then run 1M
online. Both sample select_n=16 candidate chunks per decision. The ONLY
difference is what scores the candidates:

- Arm 1 (`train_sac_chunked.py`): the online critic, Q(s, chunk).
  This is QC's own best-of-N. No world model anywhere.
- Arm 2 (`train_sac_chunked_wm.py`): the world model, scoring exactly as
  this repo already did -- imagine the candidate chunk in latent space,
  pooled imagined reward + gamma^h * cont * Q(decoded end state). One
  decode per candidate, at the end. select_rollout_chunks stays 1.

Because both arms sample the same 16 candidates from the same policy and
differ only in the scorer, any gap is attributable to the world model.

Metric decided in advance: steps until eval/mean_return first crosses a
fixed threshold both arms clearly pass. Log the 2/3-cube crossing and the
3/3 crossing separately -- the previous single-seed run reached 2/3
earlier than the critic arm but never got the third cube.

Commands in COMMANDS.txt. Always pass run_name.
