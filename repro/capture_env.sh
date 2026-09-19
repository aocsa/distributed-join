#!/usr/bin/env bash
# Dump host + pixi environment for comparing a new box against repro/env/.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "${root}"

echo "=== host ==="
echo "hostname $(hostname)"
echo "uname $(uname -a)"
echo "date_utc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "cwd ${root}"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "git $(git rev-parse --abbrev-ref HEAD) $(git rev-parse HEAD)"
fi

echo
echo "=== gpu ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found"
fi

echo
echo "=== pixi ==="
if command -v pixi >/dev/null 2>&1; then
  pixi --version
  if [[ -f pixi.toml ]]; then
    pixi list || true
  fi
else
  echo "pixi not found"
fi

echo
echo "=== nvcc (pixi env, if present) ==="
nvcc_bin="${root}/.pixi/envs/default/bin/nvcc"
if [[ -x "${nvcc_bin}" ]]; then
  "${nvcc_bin}" --version || true
else
  echo "no .pixi env yet"
fi
