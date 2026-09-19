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

## Compilation

This project depends on CUDA 12.9, UCX, NCCL, MPI, cuDF 25.12 and nvCOMP 5.

### Reproduce on a new NVIDIA box

On a machine that already has an NVIDIA driver, git, and curl:

```bash
curl -fsSL https://raw.githubusercontent.com/aocsa/distributed-join/pixi-cuda12-rapids2512/repro/bootstrap.sh | bash
```

That clones branch `pixi-cuda12-rapids2512`, installs the pixi environment from
`pixi.lock`, builds, and runs the tests. Details, env pins, and port notes are in
[`repro/`](repro/README.md).

### Using pixi (existing clone)

`pixi.toml` describes a self-contained build environment (CUDA 12.9 toolkit, GCC 14, libcudf and librmm 25.12, nvCOMP 5, NCCL, UCX, Open MPI, CMake) from conda-forge and rapidsai, for both `linux-64` and `linux-aarch64` (GB10 / DGX Spark). Install [pixi](https://pixi.sh), then:
```bash
pixi install
porting/patch_nvcc_activate.sh
pixi run build
```
Binaries end up in `build/bin/benchmark` and `build/bin/test`. To run them, activate the environment first with `pixi shell`.
