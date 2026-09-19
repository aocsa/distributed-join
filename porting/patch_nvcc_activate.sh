#!/usr/bin/env bash
# cuda-nvcc's conda activate script does:
#   NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS} -ccbin=${CXX}"
# which exits under bash `set -u` (pixi run, this repo's scripts) when the
# variable is unset. Make the expansion nounset-safe. Idempotent; re-run after
# `pixi install`.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
shopt -s nullglob
patched=0
for f in "$root"/.pixi/envs/*/etc/conda/activate.d/*cuda-nvcc_activate.sh; do
  if grep -q 'NVCC_PREPEND_FLAGS="\${NVCC_PREPEND_FLAGS-}' "$f"; then
    continue
  fi
  python3 - "$f" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
replacements = [
    (
        'NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS} -ccbin=${CXX}"',
        'NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS-} -ccbin=${CXX}"',
    ),
    (
        'NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS} -I${PREFIX}/${targetsDir}/include"',
        'NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS-} -I${PREFIX}/${targetsDir}/include"',
    ),
    (
        'NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS} -L${PREFIX}/${targetsDir}/lib"',
        'NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS-} -L${PREFIX}/${targetsDir}/lib"',
    ),
    (
        'NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS} -L${PREFIX}/${targetsDir}/lib/stubs"',
        'NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS-} -L${PREFIX}/${targetsDir}/lib/stubs"',
    ),
    ('export CFLAGS="${CFLAGS} ${CUDA_CFLAGS}"', 'export CFLAGS="${CFLAGS-} ${CUDA_CFLAGS}"'),
    (
        'export CPPFLAGS="${CPPFLAGS} ${CUDA_CFLAGS}"',
        'export CPPFLAGS="${CPPFLAGS-} ${CUDA_CFLAGS}"',
    ),
    (
        'export CXXFLAGS="${CXXFLAGS} ${CUDA_CFLAGS}"',
        'export CXXFLAGS="${CXXFLAGS-} ${CUDA_CFLAGS}"',
    ),
    ('export LDFLAGS="${LDFLAGS} ${CUDA_LDFLAGS}"', 'export LDFLAGS="${LDFLAGS-} ${CUDA_LDFLAGS}"'),
]
for old, new in replacements:
    text = text.replace(old, new)
path.write_text(text)
PY
  patched=1
done
if [[ "$patched" -eq 1 ]]; then
  echo "patched cuda-nvcc activate scripts for bash nounset"
fi
