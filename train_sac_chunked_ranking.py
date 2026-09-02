""" TD-MPC2 latent model ranks the candidate chunks. QC-FQL training
    untouched. Differs from the control only in what scores the candidates.

    See arms/ranking.py for the method and configs.yaml for its knobs.

      python train_sac_chunked_ranking.py --general.run_name=NAME --seed=0
      python train_sac_chunked_ranking.py --configs toy --seed=0      # fast CPU sanity run """

import os
os.environ.setdefault('MUJOCO_GL', 'egl')  # headless rendering

from arms.ranking import RankingArm
from sac_chunked.experiment import main

if __name__ == '__main__':
    main(RankingArm)
