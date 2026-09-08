""" CONTROL (QC-FQL + critic best-of-N) with wm potential-based reward
    shaping in the critic's target. See arms/shaping.py and configs.yaml
    (`shaping` block).

      python train_shaped_control.py --general.run_name=NAME --seed=0
      python train_shaped_control.py --shaping.potential=none ...   # matched control """

import os
os.environ.setdefault('MUJOCO_GL', 'egl')

from arms.shaping import ShapedControlArm
from sac_chunked.experiment import main

if __name__ == '__main__':
    main(ShapedControlArm)
