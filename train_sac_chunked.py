""" CONTROL: QC-FQL with critic best-of-N chunk selection (QC's own). No
    learned model. --chunk.select_n=1 for plain QC-FQL.

    See arms/critic.py for the method and configs.yaml for its knobs.

      python train_sac_chunked.py --general.run_name=NAME --seed=0
      python train_sac_chunked.py --configs toy --seed=0      # fast CPU sanity run """

import os
os.environ.setdefault('MUJOCO_GL', 'egl')  # headless rendering

from arms.critic import CriticArm
from sac_chunked.experiment import main

if __name__ == '__main__':
    main(CriticArm)
