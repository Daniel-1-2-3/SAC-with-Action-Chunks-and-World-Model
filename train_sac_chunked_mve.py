""" Model-based value expansion for the QC-FQL critic target. Acting is the
    control's critic best-of-N, so only the critic target differs.

    See arms/mve.py for the method and configs.yaml for its knobs.

      python train_sac_chunked_mve.py --general.run_name=NAME --seed=0
      python train_sac_chunked_mve.py --configs toy --seed=0      # fast CPU sanity run """

import os
os.environ.setdefault('MUJOCO_GL', 'egl')  # headless rendering

from arms.mve import MVEArm
from sac_chunked.experiment import main

if __name__ == '__main__':
    main(MVEArm)
