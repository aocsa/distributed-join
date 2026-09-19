# Porting notes: cuDF 0.19 / CUDA 11 -> RAPIDS 25.12 / CUDA 12.9

**New box:** use [`../repro/`](../repro/README.md). That folder is the
reproduction kit (bootstrap, env snapshot, notes, tools). This directory keeps
the helper scripts the bootstrap calls.

## Replicating on a new machine

Only an NVIDIA driver (>= 525), git and curl are needed. Everything else comes from
conda channels via pixi and lands in `./.pixi`, about 6 GB.

```bash
curl -fsSL https://raw.githubusercontent.com/aocsa/distributed-join/pixi-cuda12-rapids2512/repro/bootstrap.sh | bash
```

That installs pixi, clones branch `pixi-cuda12-rapids2512`, runs `pixi install` from the committed
`pixi.lock` (exact same package builds), builds, and runs the tests. From an existing
clone of this branch:

```bash
repro/bootstrap.sh
```

or the individual steps:

```bash
pixi install
porting/patch_nvcc_activate.sh
pixi run build
porting/build_and_test.sh 2
```

If `pixi install` fails with `unsupported-platform` / `linux-aarch64`, the lockfile
on the checked-out branch is still `linux-64` only. This branch declares both
platforms.

Scripts in this folder:

| Script | Purpose |
|---|---|
| `bootstrap_new_box.sh` | Fresh-machine setup: pixi, clone, install, build, test. |
| `build_and_test.sh [ranks]` | Clean build inside the pixi env and run all tests on one GPU. |
| `summarize_build_errors.sh` | Keep-going build; distinct compiler errors ranked by count. Used to drive the port. |
| `list_rapids_versions.py` | Lists libcudf CUDA 12 builds and their rmm/nvcomp pins from the rapidsai channel. |
| `patch_nvcc_activate.sh` | Makes `cuda-nvcc`'s conda activate script safe under bash `set -u`. Re-run after `pixi install`. |

Tested on: NVIDIA GB10 (linux-aarch64, driver 580.126.09), pixi 0.79.

## Why RAPIDS 25.12

cuDF 0.19 only exists for CUDA 11.0/11.2. Any CUDA 12 build forces a newer RAPIDS.
GB10 is `linux-aarch64` + compute 12.1 (`sm_121`). Candidates tried:

| Stack | Result |
|---|---|
| libcudf 23.08, CUDA 12.2 | CUDA 12 builds exist only for linux-64. pixi: `unsupported-platform`. |
| libcudf 23.12, CUDA 12.9 | First aarch64 CUDA 12 stack, but libcudf fatbins stop at sm_90 with no PTX, so they cannot run on GB10. |
| libcudf 26.x | rmm 26 removes `rmm::mr::device_memory_resource`. Abandoned for this port. |
| libcudf 25.12, CUDA 12.9 | aarch64 CUDA 12.9 builds, sm_120 cubin + PTX (binary-compatible with GB10), nvcc can emit sm_121, rmm still has `device_memory_resource`. Chosen. |

Constraints that follow from 25.12: C++20, GCC 14, nvCOMP 5 (no `nvcompCascadedFormatOpts`),
rmm logs through `rapids_logger` instead of `fmt`, `allocate`/`deallocate` take
`(stream, bytes)`, string chars live in the parent data buffer rather than `child(1)`,
and unified-memory SoCs need an RMM pool cap so the CPU keeps RAM.

CCCL must be 3.1 (RAPIDS 25.12). The CUDA 12.9 `cuda-cccl` metapackage pins CCCL 2.8;
this env uses the standalone `cccl` 3.1 package and searches `$CONDA_PREFIX/include/cccl`
before the toolkit headers. rmm 25.12 is a shared library (`librmm.so`) linked via
`find_package(rmm CONFIG)`.

Versions were discovered with `list_rapids_versions.py` plus `cuobjdump` on
`libcudf.so` fatbins.

## What changed in the source

1. **nvcomp** (`src/compression.hpp`): `CascadedCompressor`, `CascadedDecompressor`
   and `CascadedSelector` were removed in nvcomp 2.3. Compression now uses the HLIF
   `nvcomp::CascadedManager`; decompression uses `nvcomp::create_manager`, which reads
   the format from the compressed header. nvCOMP 5 also dropped
   `nvcompCascadedFormatOpts`; a local `CascadedFormatOpts` POD is MPI-broadcast instead.
   Behavior change: the selector is gone, so auto-selected options are fixed at
   RLE=1, delta=1, bitpack=1 instead of being sampled per column.
2. **cuDF join** (`src/distributed_join.cpp`): `inner_join(left, right, left_on, right_on)`
   returning a table no longer exists. `inner_join_all_columns` gathers both sides with
   the returned gather maps and keeps the 0.19 layout (all left columns, then all right).
   The header moved to `cudf/join/join.hpp`.
3. **rmm streams and allocate order**: `cuda_stream_view(int)` is deleted, so every
   `cudaStreamDefault`/`0` passed to `allocate`, `deallocate`, `device_buffer` and
   `resize` became `rmm::cuda_stream_default`. Public `allocate`/`deallocate` take
   `(stream, ptr/bytes)`.
4. **String columns**: chars are stored in the parent data buffer; offsets remain
   `child(0)`. `make_strings_column(num_strings, offsets_column, chars_buffer, ...)`.
   Communication of the char payload uses `column.head()`, not `child(1)`.
5. **Includes**: `cudf/partitioning.hpp` for `hash_id`, `cudf/join/join.hpp`,
   `rmm/mr/*.hpp` (the `rmm/mr/device/` path is deprecated), thrust `sort.h`,
   `set_operations.h`, `execution_policy.h` in `generate_dataset.cuh`.
6. **Unified memory** (`src/setup.cpp`): `recommended_rmm_pool_size()` caps the pool
   on integrated GPUs (GB10 reports the full CPU+GPU RAM via `cudaMemGetInfo`).
7. **GCC -Werror**: `strncpy` bound in `gpubdb_shuffle_on.cpp`, unused variable in
   `test_shuffle_on.cpp`.

## Build system

- C++/CUDA standard 14 -> 20 (cuDF 25.12 requires it).
- Link `rapids_logger` instead of `fmt::fmt`.
- `FindCUDF.cmake` probes `cudf/types.hpp` instead of `cudf/join.hpp`.
- `GPU_ARCHS` is `80;90;120;121` when nvcc >= 12.9.
- `cuda-profiler-api` and `libcurand-dev` added to the env for `cuda_profiler_api.h`
  and `curand.h`.

## pixi gotchas hit along the way

- `pixi run` uses its own restricted shell. Pipes and `$VAR` work, but `| head -1`
  and multi-command scripts are unreliable. `build_and_test.sh` uses
  `eval "$(pixi shell-hook)"` in bash instead.
- `cmake --build build -k 0` is not valid; pass `-- -k 0` to Ninja.
- Do not export `UCX_ROOT` from the pixi activation; UCX treats every `UCX_*` variable
  as its own and warns. `CMAKE_PREFIX_PATH=$CONDA_PREFIX` is enough for the finders.
- Named platforms with a `cuda = "12"` virtual package made `pixi run -e <env>` fail
  with "no platform supported by it matches the current system" on a CUDA 13 driver.
  A plain `linux-64`/`linux-aarch64` platform with an explicit `cuda-version` pin
  avoids it.
- If you already sit in a git checkout, `bootstrap_new_box.sh` must not clone into
  `./distributed-join` again.
- `cuda-nvcc`'s activate script does `NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS} ..."`,
  which aborts under bash `set -u` (`pixi run`, `build_and_test.sh`). After
  `pixi install`, run `porting/patch_nvcc_activate.sh`. The test/bootstrap scripts
  do that automatically, and they also `set +u` around `pixi shell-hook`.

## Test expectations on a single GPU

`buffer_communicator` and the `distributed_join` benchmark size their RMM pool to
almost all free GPU memory per rank, so run them with `-n 1` on a one-GPU box.
The other four tests pass with `mpirun --oversubscribe -n 2`.
