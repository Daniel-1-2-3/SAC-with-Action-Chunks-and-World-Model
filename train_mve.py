""" MVE: the control, with the critic's TD target bootstrapping on the world
    model's chunk value at the real next state instead of on the target
    critic. Selection, actor and replay are the control's, unchanged.

    tdmpc.chunk_boot_agg defaults to 'mean' -- the model's own bootstrap now
    uses the critic's aggregation rule, so both value functions carry equal
    optimism (see arms/mve.py).

      python train_mve.py --general.run_name=NAME --seed=0
      python train_mve.py --mve.critic_target_source=critic ...   # matched control """

import os
os.environ.setdefault('MUJOCO_GL', 'egl')

from arms.mve import MVEArm
from sac_chunked.experiment import main

if __name__ == '__main__':
    main(MVEArm)
