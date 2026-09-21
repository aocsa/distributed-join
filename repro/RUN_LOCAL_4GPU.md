# Single-node run on 4 GPUs (presto-gb200-gcn-18, 4x GB200)

Measured 2026-09-21 on branch `pixi-cuda12-rapids2512` + the uncommitted fixes listed
below. Launcher: `repro/run_local_gpus.sh` (1 MPI rank per local GPU, Open MPI `sm,tcp`
for MPI itself, `UCX_TLS=tcp,cuda_copy,cuda_ipc`, `NCCL_IB_DISABLE=1`).

```bash
export PATH="$HOME/.pixi/bin:$PATH"
porting/patch_nvcc_activate.sh
set +u; eval "$(pixi shell-hook)"; set -u
export NIXL_ROOT=<prefix with include/nixl.h, lib/libnixl.so, lib/plugins/libplugin_UCX.so>
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGPU_ARCHS=100   # GB200 = sm_100
cmake --build build -j --target distributed all_to_all distributed_join test_shuffle_on

export RMM_POOL_GIB=12                       # GPUs are shared, see below
COMMUNICATOR=NCCL ./repro/run_local_gpus.sh shuffle
COMMUNICATOR=NCCL ./repro/run_local_gpus.sh join
RMM_POOL_GIB=15 COMMUNICATOR=NCCL REPEAT=4 ./repro/run_local_gpus.sh all_to_all
COMMUNICATOR=UCX  ./repro/run_local_gpus.sh shuffle|join|all_to_all
```

## Box-specific traps

| Symptom | Cause | Fix |
|---|---|---|
| `path for conda-forge's environment with rdma-core cannot be longer than 90 characters`, abort in `librdmacm` during `MPI_Bcast` | conda-forge `librdmacm` hard-codes a 90-char prefix limit; Open MPI reaches it through its libfabric (`ofi`) component. The pixi prefix here is 116 chars. | `OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,sm,tcp OMPI_MCA_mtl=^ofi` (launcher default). |
| NCCL `ncclGroupEnd` "unhandled cuda error", log shows `ncclLaunchKernel ... Cuda failure 2 'out of memory'` | Another job (`sirius-doris-be`) holds ~170 GB of each 189 GB GPU. `recommended_rmm_pool_size()` takes 90% of what is free and NCCL cannot allocate. | New `RMM_POOL_GIB` env cap in `src/setup.cpp`. 12 GiB is enough for join/shuffle; the 4 GB all_to_all step needs 15 GiB with NCCL because the NCCL communicator stages every send/recv through a second pool buffer. |
| `ibv_query_port_speed failed ... Protocol not supported` spam from NCCL | pixi `libibverbs` vs host mlx5 provider. Irrelevant intra-node. | `NCCL_IB_DISABLE=1` (launcher default). |
| `cudaErrorMisalignedAddress` in the `compression=true` shuffle case (nvCOMP `PinnedPtrPool`/`ManagerBase` destructors) | Compressed partitions were packed back-to-back at raw cumulative byte offsets; nvCOMP reads the compressed header with aligned loads. | `src/all_to_all_comm.cpp` pads each compressed partition to 256 bytes. |
| `all_to_all` runs out of pool at 4 GB even with a big pool | `run_all_to_all` deallocated `irank < mpi_rank` buffers instead of all peers, leaking on every size. | Fixed loop in `benchmark/all_to_all.cpp`. |
| NIXL ranks spin at 100% CPU forever after `NIXL pool memory: vram` | The pip `nixl-cu12` wheel bundles its own (renamed) UCX. Its `libplugin_UCX.so` resolves `uct_*`/`ucp_*` against the pixi UCX already loaded by the UCX communicator; mismatched structs make `uct_md_query_tl_resources` realloc 85 GB. | Do not use the pip wheel. Build NIXL from source against the pixi UCX (`meson setup -Ducx_path=$CONDA_PREFIX -Dcudapath_inc=... -Denable_plugins=UCX -Dbuild_tests=false -Dbuild_examples=false -Dnixl_cuda_arch_list=100`). |

## Numbers measured 2026-09-21 (4 ranks, 1 per GB200, single run each, GPUs shared with another job)

Join is 1M x 1M rows per rank (the log reports the 4M totals). Shuffle is 1M INT32 rows
per rank; the first shuffle case is the in-process warmup. all_to_all is `--repeat 4`.

| Communicator | Join | Shuffle, no compression | Shuffle, cascaded | a2a 1 GB | a2a 2 GB | a2a 4 GB |
|---|---|---|---|---|---|---|
| NCCL | 0.0087 s | 0.00042 s | 0.030 s | 399 GB/s | 470 GB/s | 479 GB/s (needs `RMM_POOL_GIB=15`) |
| UCX `tcp,cuda_copy,cuda_ipc`, preregistered | 0.028 s | 0.0048 s | 0.034 s | 138 GB/s | 228 GB/s | 315 GB/s |
| NIXL VRAM pool (source-built against pixi UCX) | 0.0092 s | 0.00065 s | 0.031 s | 431 GB/s | 440 GB/s | 444 GB/s |

UCX all_to_all has a ~17 ms floor at every size below 256 MB (per-call cost of the
preregistered path), which is why its small-message bandwidth is far below the others.
