#!/bin/bash
# bash install.sh   -- once per pod, from the repo root
set -e
pip install -r requirements_gpu.txt
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; assert torch.cuda.is_available(), 'torch cannot see the GPU: check nvidia-smi / the CUDA index above'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
apt-get update && apt-get install -y libgl1-mesa-glx libglu1-mesa libosmesa6 libegl1 libglx-mesa0
