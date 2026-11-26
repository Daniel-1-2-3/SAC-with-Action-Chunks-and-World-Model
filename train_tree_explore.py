""" TREE-EXPLORE: the combined method + a world-model tree of imagined
    futures at act-time selection, online collection only. Training,
    targets and eval are the combined method's (arms/tree_explore.py).

      python train_tree_explore.py --general.run_name=NAME --seed=0
      python train_tree_explore.py --tree.depth=1 --general.run_name=NAME --seed=0   # depth control: no tree, only each candidate's own end point
      python train_tree_explore.py --tree.beta=0 --general.run_name=NAME --seed=0    # every decision is the combined method's
      python train_tree_explore.py --tree.bandit=False --general.run_name=NAME --seed=0  # every online episode explores
"""

import os
os.environ.setdefault('MUJOCO_GL', 'egl')

from arms.tree_explore import TreeExploreArm
from sac_chunked.experiment import main

if __name__ == '__main__':
    main(TreeExploreArm)
