""" WM-ONLY: no critic. The TD-MPC2 latent model's chunk Q is the sole
    value function -- it trains on real replay chunk transitions (actor
    bootstrap, tdmpc.chunk_boot_agg aggregation), scores best-of-N at act
    and eval time, and supplies the actor's Q-term gradient. See
    arms/wm_only.py.

    The value function trains in the model update, so run at full cadence:

      python train_wm_only.py --tdmpc.train_every=1 \
          --general.run_name=NAME --seed=0 """

import os
os.environ.setdefault('MUJOCO_GL', 'egl')

from arms.wm_only import WMOnlyArm
from sac_chunked.experiment import main

if __name__ == '__main__':
    main(WMOnlyArm)