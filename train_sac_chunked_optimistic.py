""" Ranking with an optimistic (RBMLE) dynamics loss on the latent model.
    Differs from the ranking arm only in that loss.

    See arms/optimistic.py for the method and configs.yaml for its knobs.

      python train_sac_chunked_optimistic.py --general.run_name=NAME --seed=0
      python train_sac_chunked_optimistic.py --configs toy --seed=0      # fast CPU sanity run """

import os
os.environ.setdefault('MUJOCO_GL', 'egl')  # headless rendering

from arms.optimistic import OptimisticArm
from sac_chunked.experiment import main

if __name__ == '__main__':
    main(OptimisticArm)
