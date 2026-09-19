#!/usr/bin/env bash
# Reproduce the RAPIDS 25.12 / CUDA 12.9 build on a fresh Linux box
# (x86_64 or aarch64) that already has an NVIDIA driver.
#
# Empty machine:
#   curl -fsSL https://raw.githubusercontent.com/aocsa/distributed-join/pixi-cuda12-rapids2512/repro/bootstrap.sh | bash
#
# Existing clone of this branch:
#   repro/bootstrap.sh
#
# Optional env:
#   REPO, BRANCH, DIR, SKIP_TESTS=1, RANKS=2
set -euo pipefail

REPO="${REPO:-https://github.com/aocsa/distributed-join.git}"
BRANCH="${BRANCH:-pixi-cuda12-rapids2512}"
DIR="${DIR:-distributed-join}"
RANKS="${RANKS:-2}"
SKIP_TESTS="${SKIP_TESTS:-0}"

need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

need git
need curl
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found: install the NVIDIA driver (>= 525) first." >&2
  exit 1
fi
echo "=== GPU ==="
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
echo "arch=$(uname -m)"

avail_kb="$(df -Pk . | awk 'NR==2 { print $4 }')"
if [[ -n "$avail_kb" ]] && (( avail_kb < 8 * 1024 * 1024 )); then
  echo "need about 8 GiB free disk for .pixi + build (have $((avail_kb / 1024)) MiB)" >&2
  exit 1
fi

if ! command -v pixi >/dev/null 2>&1; then
  echo "=== installing pixi ==="
  curl -fsSL https://pixi.sh/install.sh | bash
  export PATH="${HOME}/.pixi/bin:${PATH}"
fi
echo "pixi $(pixi --version)"

# Already sitting in a checkout of this repo: do not clone into ./distributed-join.
if [[ -f pixi.toml && -d .git ]]; then
  DIR="."
elif [[ ! -d "${DIR}/.git" ]]; then
  echo "=== clone ${REPO} (${BRANCH}) -> ${DIR} ==="
  git clone --branch "${BRANCH}" "${REPO}" "${DIR}"
fi
cd "${DIR}"
echo "=== repo $(pwd) @ $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD) ==="

if [[ -x repro/check_box.sh ]]; then
  repro/check_box.sh
fi

echo "=== pixi install (from pixi.lock) ==="
pixi install

echo "=== patch cuda-nvcc activate for bash nounset ==="
porting/patch_nvcc_activate.sh

if [[ "${SKIP_TESTS}" == "1" ]]; then
  echo "=== build (SKIP_TESTS=1) ==="
  pixi run build
  echo "ALL DONE (build only)"
  exit 0
fi

echo "=== clean build + tests (${RANKS} ranks where applicable) ==="
porting/build_and_test.sh "${RANKS}"
