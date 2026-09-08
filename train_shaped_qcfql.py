""" QC-FQL (Alg. 2, single actor sample -- select_n forced to 1) with wm
    potential-based reward shaping in the critic's target. See
    arms/shaping.py and configs.yaml (`shaping` block). Paper alpha for
    cube-triple QC-FQL is 100: pass --chunk.alpha=100 to match Table 3.

      python train_shaped_qcfql.py --general.run_name=NAME --seed=0
      python train_shaped_qcfql.py --shaping.potential=none ...   # matched control """

import os
os.environ.setdefault('MUJOCO_GL', 'egl')

from arms.shaping import ShapedQCFQLArm
from sac_chunked.experiment import main

if __name__ == '__main__':
    main(ShapedQCFQLArm)
