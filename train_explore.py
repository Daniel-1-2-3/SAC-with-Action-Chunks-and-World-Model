""" EXPLORE: the combined method (control) plus a bandit-gated
    dynamics-disagreement bonus on act-time selection during online
    collection. Training, targets and eval are the control's; only WHICH
    candidate chunk gets executed while collecting differs, and a
    per-episode UCB bandit turns the bonus off whenever exploit episodes
    out-earn explore episodes. See arms/explore.py.

      python train_explore.py --general.run_name=NAME --seed=1
      python train_explore.py --explore.beta=0 ...   # exactly the control

    Aimed at late-discovery seeds (the documented task4 s1/s2 disease):
    run it against the same seed's plain control. """

import os
os.environ.setdefault('MUJOCO_GL', 'egl')

from arms.explore import ExploreArm
from sac_chunked.experiment import main

if __name__ == '__main__':
    main(ExploreArm)
