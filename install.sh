#!/bin/bash
set -e
pip install -r requirements_gpu.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_capability())"
apt-get update
apt-get install -y libgl1-mesa-glx libglu1-mesa libosmesa6 libosmesa6-dev libegl1 libglx-mesa0
# bash install.sh