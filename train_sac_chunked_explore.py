""" Ranking plus a dynamics-ensemble disagreement bonus on the score.
    Differs from the ranking arm only in beta.

    See arms/explore.py for the method and configs.yaml for its knobs.

      python train_sac_chunked_explore.py --general.run_name=NAME --seed=0
      python train_sac_chunked_explore.py --configs toy --seed=0      # fast CPU sanity run """

import os
os.environ.setdefault('MUJOCO_GL', 'egl')  # headless rendering

from arms.explore import ExploreArm
from sac_chunked.experiment import main

if __name__ == '__main__':
    main(ExploreArm)
