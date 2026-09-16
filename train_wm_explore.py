""" WM-EXPLORE: the combined method plus a bandit-gated
    dynamics-disagreement bonus on act-time selection during online
    collection. Training, targets and eval are the combined arm's; only WHICH
    candidate chunk gets executed while collecting differs, and a
    per-episode UCB bandit turns the bonus off whenever exploit episodes
    out-earn explore episodes. See arms/wm_explore.py.

      python train_wm_explore.py --general.run_name=NAME --seed=1
      python train_wm_explore.py --wm_explore.beta=0 ...   # exactly the combined arm

    Aimed at late-discovery seeds (the documented task4 s1/s2 disease):
    run it against the same seed's plain combined arm. """

import os
os.environ.setdefault('MUJOCO_GL', 'egl')

from arms.wm_explore import WMExploreArm
from sac_chunked.experiment import main

if __name__ == '__main__':
    main(WMExploreArm)
