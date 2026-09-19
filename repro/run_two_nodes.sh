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
KEY="${HOME}/.ssh/id_ed25519"
export PRTE_MCA_plm_ssh_args="-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i ${KEY}"
export OMPI_MCA_plm_rsh_args="${PRTE_MCA_plm_ssh_args}"
HOSTS="${HOSTS:-10.87.131.64:1,10.87.131.68:1}"
prefix="${CONDA_PREFIX}"
app=(mpirun --prefix "$prefix" -np 2 --host "$HOSTS" --map-by ppr:1:node
  -x PATH -x LD_LIBRARY_PATH -x UCX_TLS -x UCX_MEMTYPE_CACHE -x UCX_WARN_UNUSED_ENV_VARS)

case "${1:-all}" in
  hostname) "${app[@]}" hostname ;;
  shuffle)  "${app[@]}" "$root/build/bin/test/test_shuffle_on" ;;
  join)
    "${app[@]}" "$root/build/bin/benchmark/distributed_join" \
      --build-table-nrows "${BUILD_NROWS:-1000000}" \
      --probe-table-nrows "${PROBE_NROWS:-1000000}"
    ;;
  all)
    "${app[@]}" hostname
    "${app[@]}" "$root/build/bin/test/test_shuffle_on"
    "${app[@]}" "$root/build/bin/benchmark/distributed_join" \
      --build-table-nrows "${BUILD_NROWS:-1000000}" \
      --probe-table-nrows "${PROBE_NROWS:-1000000}"
    ;;
  *) echo "usage: $0 [hostname|shuffle|join|all]" >&2; exit 2 ;;
esac
