#!/bin/bash
set -e
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements_gpu.txt
python -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_capability())"
# bash install.sh