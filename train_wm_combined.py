""" WM-COMBINED: the combined arm, with the critic's TD target bootstrapping
    on the world model's chunk value at the real next state instead of on
    the target critic. Selection, actor and replay are the combined arm's,
    unchanged.

    tdmpc.chunk_boot_agg defaults to 'mean' -- the model's own bootstrap now
    uses the critic's aggregation rule, so both value functions carry equal
    optimism (see arms/wm_combined.py).

      python train_wm_combined.py --general.run_name=NAME --seed=0
      python train_wm_combined.py --wm_combined.critic_target_source=critic ...   # matched control """

import os
os.environ.setdefault('MUJOCO_GL', 'egl')

from arms.wm_combined import WMCombinedArm
from sac_chunked.experiment import main

if __name__ == '__main__':
    main(WMCombinedArm)
