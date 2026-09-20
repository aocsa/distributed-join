#!/usr/bin/env bash
# 1 rank on zeno-01 (10.87.131.64) + 1 rank on zeno-02 (10.87.131.68).
# Requires passwordless SSH from this box using ~/.ssh/id_ed25519.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
set +u
eval "$(pixi shell-hook)"
set -u
export UCX_WARN_UNUSED_ENV_VARS=n
export UCX_TLS="${UCX_TLS:-tcp,cuda_copy}"
export UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE:-n}"
export NIXL_ROOT="${NIXL_ROOT:-/home/aocsa/git/nixl/install}"
if [[ -d "${NIXL_ROOT}/lib" ]]; then
  export LD_LIBRARY_PATH="${NIXL_ROOT}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export NIXL_PLUGIN_DIR="${NIXL_PLUGIN_DIR:-${NIXL_ROOT}/lib/plugins}"
fi
KEY="${HOME}/.ssh/id_ed25519"
export PRTE_MCA_plm_ssh_args="-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i ${KEY}"
export OMPI_MCA_plm_rsh_args="${PRTE_MCA_plm_ssh_args}"
HOSTS="${HOSTS:-10.87.131.64:1,10.87.131.68:1}"
prefix="${CONDA_PREFIX}"
app=(mpirun --prefix "$prefix" -np 2 --host "$HOSTS" --map-by ppr:1:node
  -x PATH -x LD_LIBRARY_PATH -x UCX_TLS -x UCX_MEMTYPE_CACHE -x UCX_WARN_UNUSED_ENV_VARS
  -x NIXL_ROOT -x NIXL_PLUGIN_DIR)
if [[ -n "${UCX_RNDV_THRESH:-}" ]]; then
  app+=(-x UCX_RNDV_THRESH)
fi
if [[ -n "${UCX_NET_DEVICES:-}" ]]; then
  app+=(-x UCX_NET_DEVICES)
fi
# NIXL_HOST_POOL=1 backs the NIXL RMM pool with pinned host memory (GB10 has no GPUDirect RDMA).
for v in NIXL_HOST_POOL NIXL_HOST_POOL_GIB; do
  if [[ -n "${!v:-}" ]]; then
    app+=(-x "$v")
  fi
done

join_args=(--build-table-nrows "${BUILD_NROWS:-1000000}" --probe-table-nrows "${PROBE_NROWS:-1000000}")
if [[ -n "${COMMUNICATOR:-}" ]]; then
  join_args+=(--communicator "${COMMUNICATOR}")
fi

case "${1:-all}" in
  hostname) "${app[@]}" hostname ;;
  shuffle)
    shuffle_args=()
    if [[ -n "${COMMUNICATOR:-}" ]]; then
      shuffle_args+=(--communicator "${COMMUNICATOR}")
    fi
    "${app[@]}" "$root/build/bin/test/test_shuffle_on" "${shuffle_args[@]}"
    ;;
  join)
    "${app[@]}" "$root/build/bin/benchmark/distributed_join" "${join_args[@]}"
    ;;
  all_to_all)
    a2a_args=()
    if [[ -n "${COMMUNICATOR:-}" ]]; then
      a2a_args+=(--communicator "${COMMUNICATOR}")
    fi
    if [[ -n "${REPEAT:-}" ]]; then
      a2a_args+=(--repeat "${REPEAT}")
    fi
    "${app[@]}" "$root/build/bin/benchmark/all_to_all" "${a2a_args[@]}"
    ;;
  all)
    "${app[@]}" hostname
    "${app[@]}" "$root/build/bin/test/test_shuffle_on"
    "${app[@]}" "$root/build/bin/benchmark/distributed_join" "${join_args[@]}"
    ;;
  *) echo "usage: $0 [hostname|shuffle|join|all_to_all|all]" >&2; exit 2 ;;
esac
