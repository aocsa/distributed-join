# Distributed Join Project

## Overview

This proof-of-concept repo implements the distributed repartitioned join algorithm. The algorithm consists of three steps:
1. Hash partition: reorder input tables into partitions based on the hash values of the key columns.
2. All-to-all communication: send each partition to the corresponding MPI rank so that rows with the same hash values end up in the same rank.
3. Local join: each MPI rank performs local join independently.

For more information about the algorithm used and optimizations, please refer to [the ADMS'21 paper](http://www.adms-conf.org/2021-camera-ready/gao_adms21.pdf) and [the presentatiton](http://www.adms-conf.org/2021-camera-ready/gao_presentation.pdf).

For production-quality distributed join implementation, checkout [cuDF's Dask integration](https://rapids.ai/dask.html).

The following plot shows the weak-scaling performance when joining the `l_orderkey` column from lineitem table with the `o_orderkey` and the `o_orderpriority` columns from the orders table on TPC-H dataset with SF100k.

![weak scaling performance](/doc/tpch_perf.svg)

## Build and e2e tests

The committed `pixi.lock` is RAPIDS 25.12 + CUDA 12.9 + GCC 14 + CCCL 3.1 + nvCOMP 5
for `linux-64` and `linux-aarch64` (GB10 / DGX Spark). Pixi puts that stack in
`./.pixi` (~6 GiB). Do **not** install a system CUDA toolkit and do not run bare
`cmake` against `/usr/local/cuda` (GB10 images often ship CUDA 13).

### 1. Install on the box first

These are the only host packages. Everything else comes from pixi.

| You provide | Why |
|---|---|
| NVIDIA driver (`nvidia-smi`) | Kernel driver; GB10 on this port used 580.126.09 |
| git, curl | Clone + pixi installer |
| ~8 GiB free disk | pixi env + build tree |
| `x86_64` or `aarch64` | Declared pixi platforms |

Then install [pixi](https://pixi.sh) **once**:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
export PATH="$HOME/.pixi/bin:$PATH"
```

`install.sh` edits `~/.bashrc`. An already-open shell still needs that `export`
(or `source ~/.bashrc`) or `pixi` is not found and CMake will pick `/usr/local/cuda`.

### 2. Run the e2e tests

From a checkout of branch `pixi-cuda12-rapids2512`:

```bash
export PATH="$HOME/.pixi/bin:$PATH"
pixi install                      # from pixi.lock; do not pixi update
porting/build_and_test.sh 2       # nvcc patch, clean build, MPI tests
```

`porting/build_and_test.sh` is the e2e entry. It refuses a non-pixi `nvcc`, rebuilds
in the env, then runs:

- `buffer_communicator` and `benchmark/distributed_join` at **1 rank** (each sizes
  the RMM pool to most of free GPU memory)
- `test_shuffle_on`, `compare_against_single_gpu`, `compare_against_analytical`,
  `string_payload` at **2 ranks** (`mpirun --oversubscribe`)

Success is the line `ALL DONE`. Compile only: `SKIP_TESTS=1 repro/bootstrap.sh`.

Empty machine (installs pixi, clones this branch, then the same e2e):

```bash
curl -fsSL https://raw.githubusercontent.com/aocsa/distributed-join/pixi-cuda12-rapids2512/repro/bootstrap.sh | bash
```

Pins, traps, and env snapshots: [`repro/README.md`](repro/README.md).
