""" CONTROL: QC-FQL + critic best-of-N over the distilled one-step actor
    at act and eval time. No learned model anywhere.

    See arms/control.py for the method and configs.yaml for its knobs.

      python train_control.py --general.run_name=NAME --seed=0 """
      
import os
os.environ.setdefault('MUJOCO_GL', 'egl')

from arms.control import ControlArm
from sac_chunked.experiment import main

if __name__ == '__main__':
    main(ControlArm)
