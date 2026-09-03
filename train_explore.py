""" World-model explore: critic best-of-N plus a novelty bonus from a
    TD-MPC2 dynamics ensemble, scaled by the critic's own doubt. Differs from
    train_qcfql_bon.py only in beta.

    See arms/explore.py for the method and configs.yaml for its knobs.

      python train_explore.py --general.run_name=NAME --seed=0 """

import os
os.environ.setdefault('MUJOCO_GL', 'egl')  # headless rendering

from arms.explore import ExploreArm
from sac_chunked.experiment import main

if __name__ == '__main__':
    main(ExploreArm)
