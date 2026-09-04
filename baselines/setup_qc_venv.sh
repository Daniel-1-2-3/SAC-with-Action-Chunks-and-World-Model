#!/bin/bash
# One-off: a separate interpreter for the official QC code (JAX 0.6 + CUDA
# 12 plugin, numpy 2.x), whose pins clash with this repo's PyTorch stack.
# Run from the repo root. Creates .venv-qc (gitignored).
set -e
cd "$(dirname "$0")/.."
git submodule update --init baselines/qc
python -m venv .venv-qc
.venv-qc/bin/pip install --upgrade pip
.venv-qc/bin/pip install -r baselines/qc/requirements.txt
.venv-qc/bin/python -c "import jax; print('jax', jax.__version__, jax.devices())"
