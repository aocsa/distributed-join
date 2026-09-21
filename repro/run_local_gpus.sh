#!/usr/bin/env bash
# Single-node launcher: one MPI rank per local GPU (default: every GPU nvidia-smi lists).
# Same knobs as run_two_nodes.sh: COMMUNICATOR, REPEAT, BUILD_NROWS, PROBE_NROWS,
# UCX_TLS, UCX_NET_DEVICES, UCX_RNDV_THRESH, NIXL_ROOT, NIXL_HOST_POOL, NIXL_HOST_POOL_GIB.
# NP overrides the rank count; EXTRA_ARGS is appended to the program's argv.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
export PATH="$HOME/.pixi/bin:$PATH"
set +u
eval "$(pixi shell-hook 2>/dev/null)"
set -u
# conda-forge's librdmacm aborts when the env prefix is longer than 90 characters, and
# Open MPI's libfabric (ofi) component calls into it. Intra-node needs only sm + tcp.
export OMPI_MCA_pml="${OMPI_MCA_pml:-ob1}"
export OMPI_MCA_btl="${OMPI_MCA_btl:-self,sm,tcp}"
export OMPI_MCA_mtl="${OMPI_MCA_mtl:-^ofi}"
export UCX_WARN_UNUSED_ENV_VARS=n
# Intra-node: never touch the NICs (pixi libibverbs + host mlx5 provider mismatch).
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export UCX_TLS="${UCX_TLS:-tcp,cuda_copy,cuda_ipc}"
export UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE:-n}"
export NIXL_ROOT="${NIXL_ROOT:-/home/aocsa/git/nixl/install}"
if [[ -d "${NIXL_ROOT}/lib" ]]; then
  export LD_LIBRARY_PATH="${NIXL_ROOT}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export NIXL_PLUGIN_DIR="${NIXL_PLUGIN_DIR:-${NIXL_ROOT}/lib/plugins}"
fi
NP="${NP:-$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)}"
app=(mpirun -np "$NP" --bind-to none
  -x PATH -x LD_LIBRARY_PATH -x UCX_TLS -x UCX_MEMTYPE_CACHE -x UCX_WARN_UNUSED_ENV_VARS
  -x NIXL_ROOT -x NIXL_PLUGIN_DIR -x OMPI_MCA_pml -x OMPI_MCA_btl -x OMPI_MCA_mtl)
for v in UCX_RNDV_THRESH UCX_NET_DEVICES NIXL_HOST_POOL NIXL_HOST_POOL_GIB NCCL_DEBUG NCCL_DEBUG_SUBSYS NCCL_IB_DISABLE NCCL_P2P_DISABLE NCCL_CUMEM_ENABLE RMM_POOL_GIB; do
  if [[ -n "${!v:-}" ]]; then app+=(-x "$v"); fi
done

comm_args=()
if [[ -n "${COMMUNICATOR:-}" ]]; then comm_args+=(--communicator "${COMMUNICATOR}"); fi
extra=(${EXTRA_ARGS:-})

case "${1:-all}" in
  hostname) "${app[@]}" hostname ;;
  shuffle)
    "${app[@]}" "$root/build/bin/test/test_shuffle_on" "${comm_args[@]}" "${extra[@]}" ;;
  join)
    "${app[@]}" "$root/build/bin/benchmark/distributed_join" \
      --build-table-nrows "${BUILD_NROWS:-1000000}" --probe-table-nrows "${PROBE_NROWS:-1000000}" \
      "${comm_args[@]}" "${extra[@]}" ;;
  all_to_all)
    a2a=("${comm_args[@]}")
    if [[ -n "${REPEAT:-}" ]]; then a2a+=(--repeat "${REPEAT}"); fi
    "${app[@]}" "$root/build/bin/benchmark/all_to_all" "${a2a[@]}" "${extra[@]}" ;;
  *) echo "usage: $0 [hostname|shuffle|join|all_to_all]" >&2; exit 2 ;;
esac
