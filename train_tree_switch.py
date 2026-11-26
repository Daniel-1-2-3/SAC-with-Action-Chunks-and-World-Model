""" TREE-SWITCH: the tree_explore arm with the explore amount set per
    decision by the critic's own stuck score instead of per episode
    (arms/tree_switch.py). Training, targets and eval are the combined
    method's.

      python train_tree_switch.py --general.run_name=NAME --seed=0
      python train_tree_switch.py --switch.signal=progress --general.run_name=NAME --seed=0
      python train_tree_switch.py --switch.signal=constant --switch.constant_stuck=0.5 --general.run_name=NAME --seed=0  # timing control
"""

import os
os.environ.setdefault('MUJOCO_GL', 'egl')

from arms.tree_switch import TreeSwitchArm
from sac_chunked.experiment import main

if __name__ == '__main__':
    main(TreeSwitchArm)
