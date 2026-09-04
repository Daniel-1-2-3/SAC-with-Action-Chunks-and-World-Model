""" QC-FQL (Li et al. 2025, Alg. 2), exactly: the one-step actor's single
    output is executed. The control with select_n=1, as its own script.

      python train_qc_fql.py --chunk.alpha=100 --general.run_name=NAME --seed=0 """

import os
os.environ.setdefault('MUJOCO_GL', 'egl')

from arms.qc_fql import QCFQLArm
from sac_chunked.experiment import main

if __name__ == '__main__':
    main(QCFQLArm)
