#!/usr/bin/env bash
# Clean-build the project inside the pixi env and run the test suite on a single GPU.
# Usage: porting/build_and_test.sh [ranks]   (default 2)
set -euo pipefail
cd "$(dirname "$0")/.."
RANKS="${1:-2}"

# install.sh puts pixi in ~/.pixi/bin and edits .bashrc; an already-open shell
# will not see it. Do not fall through to /usr/local/cuda (GB10 images often
# ship CUDA 13.x).
if ! command -v pixi >/dev/null 2>&1 && [[ -x "${HOME}/.pixi/bin/pixi" ]]; then
  export PATH="${HOME}/.pixi/bin:${PATH}"
fi
if ! command -v pixi >/dev/null 2>&1; then
  echo "pixi not found. Install it, then source ~/.bashrc or:" >&2
  echo "  export PATH=\"\$HOME/.pixi/bin:\$PATH\"" >&2
  exit 1
fi

# cuda-nvcc's activate script dies under `set -u` if NVCC_PREPEND_FLAGS is unset.
porting/patch_nvcc_activate.sh
set +u
eval "$(pixi shell-hook)"
set -u
nvcc_path="$(command -v nvcc || true)"
if [[ "${nvcc_path}" != *".pixi"* ]]; then
  echo "refusing to build with nvcc=${nvcc_path:-missing}; expected pixi CUDA 12.9" >&2
  echo "pixi shell-hook did not activate ./.pixi (do not use /usr/local/cuda)." >&2
  exit 1
fi
export UCX_WARN_UNUSED_ENV_VARS=n

rm -rf build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -- -k 0

# These two grab nearly all free GPU memory per rank; force one rank on one GPU.
for t in buffer_communicator; do
  echo "=== test/$t (1 rank)"
  mpirun -n 1 "build/bin/test/$t"
done
echo "=== benchmark/distributed_join (1 rank)"
mpirun -n 1 build/bin/benchmark/distributed_join --build-table-nrows 1000000 --probe-table-nrows 1000000

for t in test_shuffle_on compare_against_single_gpu compare_against_analytical string_payload; do
  echo "=== test/$t ($RANKS ranks)"
  mpirun --oversubscribe -n "$RANKS" "build/bin/test/$t"
done
rm -f rmm_log.txt
echo "ALL DONE"
