# Porting notes: cuDF 0.19 / CUDA 11 -> RAPIDS 23.08 / CUDA 12.2

This folder holds the notes and helper scripts used to bring the project back to a
building state with a self-contained [pixi](https://pixi.sh) environment. See the
top-level `pixi.toml` for the resulting environment.

## Replicating on a new machine

Only an NVIDIA driver (>= 525), git and curl are needed. Everything else comes from
conda channels via pixi and lands in `./.pixi`, about 6 GB.

```bash
curl -fsSL https://raw.githubusercontent.com/aocsa/distributed-join/pixi-cuda12-rapids2308/porting/bootstrap_new_box.sh | bash
```

That installs pixi, clones this branch, runs `pixi install` from the committed
`pixi.lock` (exact same package builds), builds, and runs the tests. From an existing
clone, run `porting/bootstrap_new_box.sh` or the individual steps:

```bash
pixi install
pixi run build
porting/build_and_test.sh 2
```

Scripts in this folder:

| Script | Purpose |
|---|---|
| `bootstrap_new_box.sh` | Fresh-machine setup: pixi, clone, install, build, test. |
| `build_and_test.sh [ranks]` | Clean build inside the pixi env and run all tests on one GPU. |
| `summarize_build_errors.sh` | Keep-going build; distinct compiler errors ranked by count. Used to drive the port. |
| `list_rapids_versions.py` | Lists libcudf CUDA 12 builds and their rmm/nvcomp pins from the rapidsai channel. |

Tested on: Ubuntu 24.04, NVIDIA L4, driver 595.71 (CUDA 13.2), pixi 0.77.

## Why RAPIDS 23.08 and not the newest cuDF

cuDF 0.19 only exists for CUDA 11.0/11.2. Any CUDA 12 build forces a newer RAPIDS.
Two candidates were tried:

| Stack | Result |
|---|---|
| libcudf 26.08, CUDA 13.4 | rmm 26 removed `rmm::mr::device_memory_resource`, cudf moved `join.hpp`, headers need C++20, ~50 call sites to port plus the custom memory resource. Abandoned. |
| libcudf 23.08, CUDA 12.2 | Oldest release with CUDA 12 builds. Classic rmm API and `cudf/join.hpp` layout survive. Chosen. |

Constraints that follow from 23.08: nvcc 12.2 accepts at most GCC 12, and the
package pulls nvcomp 2.6.1 and needs `fmt` at link time because rmm logs via spdlog.

Versions were discovered with `list_rapids_versions.py`, which reads the rapidsai
channel repodata and prints every libcudf CUDA 12 build with its rmm and nvcomp pins.

## What changed in the source

1. **nvcomp** (`src/compression.hpp`): `CascadedCompressor`, `CascadedDecompressor`
   and `CascadedSelector` were removed in nvcomp 2.3. Compression now uses the HLIF
   `nvcomp::CascadedManager`; decompression uses `nvcomp::create_manager`, which reads
   the format from the compressed header. `nvcompCascadedFormatOpts` still exists, so
   option plumbing and the MPI broadcast are unchanged.
   Behavior change: the selector is gone, so auto-selected options are fixed at
   RLE=1, delta=1, bitpack=1 instead of being sampled per column.
2. **cuDF join** (`src/distributed_join.cpp`): `inner_join(left, right, left_on, right_on)`
   returning a table no longer exists. `inner_join_all_columns` gathers both sides with
   the returned gather maps and keeps the 0.19 layout (all left columns, then all right).
3. **rmm streams**: `cuda_stream_view(int)` is deleted, so every `cudaStreamDefault`/`0`
   passed to `allocate`, `deallocate`, `device_buffer` and `resize` became
   `rmm::cuda_stream_default`.
4. **cuDF signatures**: `make_strings_column(chars, offsets, null_mask, null_count)` and
   `create_chars_child_column(bytes, stream, mr)`.
5. **Includes**: `cudf/hashing.hpp` for `hash_id`, thrust `sort.h`, `set_operations.h`,
   `execution_policy.h` in `generate_dataset.cuh` (CCCL no longer pulls them transitively).
6. **GCC 12 -Werror**: `strncpy` bound in `gpubdb_shuffle_on.cpp`, unused variable in
   `test_shuffle_on.cpp`.

## Build system

- C++/CUDA standard 14 -> 17 (cuDF requires it).
- `find_package(fmt)` and link `fmt::fmt`.
- `FindCUDF.cmake` probes `cudf/types.hpp` instead of `cudf/join.hpp`.
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
  A plain `linux-64` platform with an explicit `cuda-version` pin avoids it.

## Test expectations on a single GPU

`buffer_communicator` and the `distributed_join` benchmark size their RMM pool to
almost all free GPU memory per rank, so run them with `-n 1` on a one-GPU box.
The other four tests pass with `mpirun --oversubscribe -n 2`.
