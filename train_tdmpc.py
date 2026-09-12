""" TD-MPC2 with action chunks -- the reference model-based method as a
    baseline row. The agent is TD-MPC2 itself (model + MPPI planning in
    latent space); the only adaptation is chunk execution: plan
    plan_chunks x chunk_len steps, commit the first chunk, replan at the
    boundary. No QC machinery anywhere. See arms/tdmpc_arm.py.

      python train_tdmpc.py --tdmpc.train_every=1 \
          --general.run_name=t4_tdmpc_s0 --seed=0 """

import os
os.environ.setdefault('MUJOCO_GL', 'egl')

from arms.tdmpc_arm import TDMPC2Arm
from sac_chunked.experiment import main

if __name__ == '__main__':
    main(TDMPC2Arm)
