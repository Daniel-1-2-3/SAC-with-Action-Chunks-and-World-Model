""" QC (Li et al. 2025, Alg. 1): flow BC policy, critic best-of-N at act,
    eval and TD-target time. PyTorch port of ColinQiyangLi/qc
    agents/acfql.py with actor_type='best-of-n'; see sac_chunked/qc_agent.py
    for the ported lines and every deviation. No one-step actor, no alpha.

      python train_qc.py --general.run_name=NAME --seed=0 """

import os
os.environ.setdefault('MUJOCO_GL', 'egl')

from arms.qc_arm import QCArm
from sac_chunked.experiment import main

if __name__ == '__main__':
    main(QCArm)
