#!/bin/bash
set -e
pip install -r requirements_gpu.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_capability())"
# bash install.sh