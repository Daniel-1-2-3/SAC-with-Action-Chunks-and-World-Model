""" CONTROL + TD-MPC2 latent model trained alongside.

    tdmpc.q_mode=chunk trains the model's Q on whole action chunks with the
    QC critic's own target family (see configs.yaml and arms/control_wm.py);
    wm_control.score_source=model makes that Q pick the act-time chunk
    instead of the critic. score_source=critic (default) is the control's
    behaviour plus model training and shadow scoring.

      python train_control_wm.py --tdmpc.q_mode=chunk --general.run_name=NAME --seed=0 """

import os
os.environ.setdefault('MUJOCO_GL', 'egl')

from arms.control_wm import WMControlArm
from sac_chunked.experiment import main

if __name__ == '__main__':
    main(WMControlArm)
