# Port notes: cuDF 0.19 / CUDA 11 → RAPIDS 25.12 / CUDA 12.9

These are the decisions and dead ends from bringing this repo up on an NVIDIA
GB10 (`linux-aarch64`, sm_121). Use them when a new box fails in a familiar way.

## Why RAPIDS 25.12

cuDF 0.19 only exists for CUDA 11.0/11.2. Any CUDA 12 build forces a newer RAPIDS.
GB10 is `linux-aarch64` + compute 12.1. Versions were probed with
`repro/tools/list_rapids_versions.py`.

| Stack | Result |
|---|---|
| libcudf 23.08, CUDA 12.2 | CUDA 12 builds exist only for linux-64. pixi: `unsupported-platform`. |
| libcudf 23.12, CUDA 12.9 | First aarch64 CUDA 12 stack, but libcudf fatbins stop at sm_90 with **no PTX**, so they cannot run on GB10. |
| libcudf 26.x | rmm 26 removes `rmm::mr::device_memory_resource`. Abandoned. |
| libcudf 25.12, CUDA 12.9 | aarch64 CUDA 12.9, sm_120 cubin + PTX (runs on GB10), nvcc can emit sm_121, rmm still has `device_memory_resource`. **Chosen.** |

Constraints that follow from 25.12: C++20, GCC 14, nvCOMP 5 (no
`nvcompCascadedFormatOpts`), rmm logs through `rapids_logger` instead of `fmt`,
`allocate`/`deallocate` take `(stream, bytes)`, string chars live in the parent
data buffer rather than `child(1)`, and unified-memory SoCs need an RMM pool cap
so the CPU keeps RAM.

## CCCL 3.1 vs CUDA 12.9 toolkit headers

RAPIDS 25.12 needs CCCL 3.1 (`cuda::mr::synchronous_resource`). The CUDA 12.9
`cuda-cccl` metapackage pins CCCL **2.8** and the toolkit still ships 2.8 under
`targets/sbsa-linux/include`. That `-I` beat conda CCCL until:

- `cuda-cccl` was **not** added as a dependency (standalone `cccl = "3.1.*"`).
- `$CONDA_PREFIX/include/cccl` is added as a **non-system** `-I` first
  (`cmake/BuildHelpers.cmake`).
- CUDA Toolkit imported include dirs are marked `SYSTEM` so they lose priority.

rmm 25.12 is a shared library (`librmm.so`) linked via `find_package(rmm CONFIG)`
and `rmm::rmm`.

## Source changes (short)

1. **nvCOMP 5** (`src/compression.hpp`): `nvcompCascadedFormatOpts` is gone. A
   local `CascadedFormatOpts` POD is MPI-broadcast. Compression uses
   `nvcomp::CascadedManager`; decompression uses `nvcomp::create_manager`.
   Auto-selected options are fixed at RLE=1, delta=1, bitpack=1.
2. **cuDF join**: `inner_join_all_columns` + gather maps. Header is
   `cudf/join/join.hpp`. `hash_id` lives in `cudf/partitioning.hpp`.
3. **rmm**: `allocate`/`deallocate` are `(stream, ptr/bytes)`. Pool is
   constructed with `*registered_mr` (the pointer lvalue does not bind to
   `Upstream2&`). Includes moved to `rmm/mr/*.hpp`.
4. **Strings**: chars in the parent data buffer; offsets in `child(0)`.
   `make_strings_column(num, offsets, chars_buffer)`. Do not call `child(1)`.
5. **GB10 unified memory** (`src/setup.cpp`): `recommended_rmm_pool_size()`
   caps the pool (GB10 reports the full CPU+GPU RAM via `cudaMemGetInfo`).
6. **GPU_ARCHS**: `80;90;120;121` when nvcc >= 12.9.

## pixi / bash traps

- `pixi.toml` must declare both `linux-64` and `linux-aarch64` or aarch64 boxes
  fail with `unsupported-platform`.
- After `pixi install`, run `porting/patch_nvcc_activate.sh`. cuda-nvcc does
  `NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS} -ccbin=${CXX}"`, which dies under
  bash `set -u` (`pixi run`, this repo's scripts) when the var is unset. pixi
  deactivates after a task (backup `UNSET`) and re-hooks, which is why
  `pixi run build` can link every target and still exit non-zero.
- Do **not** put empty `NVCC_PREPEND_FLAGS` in `[activation.env]`: pixi applies
  that **after** `activate.d` and would wipe `-ccbin=$CXX`.
- `eval "$(pixi shell-hook)"` must run under `set +u`. hwloc bash-completion
  references `ZSH_VERSION`.
- Re-run the nvcc patch after every `pixi install` (it rewrites `.pixi/`, which
  is gitignored).
- Do not export `UCX_ROOT`; UCX treats every `UCX_*` variable as its own.
  `CMAKE_PREFIX_PATH=$CONDA_PREFIX` is enough.
- `cmake --build build -k 0` is invalid; pass `-- -k 0` to Ninja.
- If you already sit in a git checkout, bootstrap must not clone into
  `./distributed-join` again.

## Tests on a single GPU

`buffer_communicator` and the `distributed_join` benchmark size their RMM pool
to almost all free GPU memory per rank: run them with `-n 1`. The other four
tests pass with `mpirun --oversubscribe -n 2`.

Keep-going compiler error ranking (used during the port):

```bash
repro/tools/summarize_build_errors.sh
```
