""" MVE: the control, with the critic's TD target bootstrapping on the world
    model's chunk value at the real next state instead of on the target
    critic. Selection, actor and replay are the control's, unchanged.

    See arms/mve.py for the method and configs.yaml (`mve` block) for its
    knobs.

      python train_mve.py --general.run_name=NAME --seed=0
      python train_mve.py --mve.critic_target_source=critic ...   # matched control """

import os
os.environ.setdefault('MUJOCO_GL', 'egl')

from arms.mve import MVEArm
from sac_chunked.experiment import main

if __name__ == '__main__':
    main(MVEArm)
