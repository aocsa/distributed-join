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

This project depends on CUDA 12, UCX, NCCL, MPI, cuDF 23.08 and nvcomp 2.6.

To compile, make sure the variables `CUDA_ROOT`, `CUDF_ROOT`, `MPI_ROOT`, `UCX_ROOT`, `NCCL_ROOT` and `NVCOMP_ROOT` are pointing to the installation path of CUDA, cuDF, MPI, UCX, NCCL and nvcomp repectively.

[The wiki page](https://github.com/rapidsai/distributed-join/wiki/How-to-compile-and-run-the-code) contains step-by-step instructions for setting up the environment.

To compile, run
```bash
mkdir build && cd build
cmake ..
make -j
```

### Using pixi

`pixi.toml` describes a self-contained build environment (CUDA 12.2 toolkit, GCC 12, libcudf and librmm 23.08, nvCOMP 2.6, NCCL, UCX, Open MPI, CMake) from conda-forge and rapidsai. Install [pixi](https://pixi.sh), then:
```bash
pixi run build
```
Binaries end up in `build/bin/benchmark` and `build/bin/test`. To run them, activate the environment first with `pixi shell`.
