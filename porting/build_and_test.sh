#!/usr/bin/env bash
# Clean-build the project inside the pixi env and run the test suite on a single GPU.
# Usage: porting/build_and_test.sh [ranks]   (default 2)
set -euo pipefail
cd "$(dirname "$0")/.."
RANKS="${1:-2}"

eval "$(pixi shell-hook)"
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
