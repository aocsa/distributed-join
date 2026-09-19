#!/usr/bin/env bash
# Reproduce the build on a fresh Linux x86_64 machine with an NVIDIA driver installed.
#
# Requirements: NVIDIA driver >= 525 (CUDA 12.x runtime), git, curl, internet access.
# Nothing else: pixi downloads the CUDA 12.2 toolkit, GCC 12, cuDF, nvcomp, UCX, NCCL and
# Open MPI into ./.pixi (about 6 GB).
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/aocsa/distributed-join/pixi-cuda12-rapids2308/porting/bootstrap_new_box.sh | bash
# or, from a clone:
#   porting/bootstrap_new_box.sh
set -euo pipefail

REPO="${REPO:-https://github.com/aocsa/distributed-join.git}"
BRANCH="${BRANCH:-pixi-cuda12-rapids2308}"
DIR="${DIR:-distributed-join}"

if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi not found: install the NVIDIA driver first." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

if ! command -v pixi >/dev/null; then
  curl -fsSL https://pixi.sh/install.sh | bash
  export PATH="$HOME/.pixi/bin:$PATH"
fi
pixi --version

if [ ! -d "$DIR/.git" ]; then
  git clone --branch "$BRANCH" "$REPO" "$DIR"
fi
cd "$DIR"

pixi install          # solves from pixi.lock, no re-resolution
pixi run build        # cmake configure + ninja
porting/build_and_test.sh 2
