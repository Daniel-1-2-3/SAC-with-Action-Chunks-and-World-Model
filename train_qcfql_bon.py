""" QC-FQL (Li et al. 2025, Alg. 2) with critic best-of-N over the distilled
    one-step actor at act and eval time. `chunk.select_n=1` is plain QC-FQL.

    This is the paper's control: no learned model anywhere.

    See arms/qcfql_bon.py for the method and configs.yaml for its knobs.

      python train_qcfql_bon.py --general.run_name=NAME --seed=0 """

import os
os.environ.setdefault('MUJOCO_GL', 'egl')  # headless rendering

from arms.qcfql_bon import QCFQLBestOfNArm
from sac_chunked.experiment import main

if __name__ == '__main__':
    main(QCFQLBestOfNArm)
