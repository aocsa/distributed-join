# Environment that built on pdx02-zeno-01

Captured 2026-09-19. Use this to sanity-check a new box after `pixi install`.
The lockfile (`pixi.lock`) is the real pin; this page is the readable subset.

## Host

- Machine: `pdx02-zeno-01`
- OS: Ubuntu, kernel `6.14.0-1015-nvidia`
- Arch: `aarch64` (`linux-aarch64`)
- GPU: NVIDIA GB10, compute 12.1 (`sm_121`)
- Driver: `580.126.09`
- pixi: `0.79.0`

Full dump: `repro/env/pdx02-zeno-01-host.txt`.

## Stack (from pixi.toml)

| Package | Pin |
|---|---|
| libcudf / librmm | 25.12.* (rapidsai, CUDA 12 builds) |
| cuda-version / cuda-nvcc / cuda-cudart-dev | 12.9.* |
| cccl | 3.1.* (do **not** use the `cuda-cccl` 12.9 metapackage; it is CCCL 2.8) |
| gxx | 14.* |
| libnvcomp-dev | 5.0.* |
| cmake | >= 3.26 |
| ninja, nccl, ucx, openmpi | unpinned, locked in `pixi.lock` |

## Exact builds on this GB10 box

From `pixi list` (full table: `repro/env/pdx02-zeno-01-pixi-list.txt`):

| Package | Version | Build |
|---|---|---|
| libcudf | 25.12.00 | `cuda12_251210_580975be` |
| librmm | 25.12.00 | `cuda12_251210_86731e05` |
| cccl | 3.1.4 | `h9248bf7_0` |
| cuda-nvcc | 12.9.86 | `ha346c71_106` |
| gxx | 14.4.0 | `hfdd745d_5` |
| libnvcomp-dev | 5.0.0.6 | `h6defbd5_3` |
| nccl | 2.30.7.1 | `h86acffb_0` |
| openmpi | 5.0.10 | `hba2b3a9_2` |
| ucx | 1.22.0 | `h8905ae9_1` |

CMake GPU archs for this project's kernels: `80;90;120;121` (nvcc 12.9).
libcudf 25.12 aarch64 ships sm_120 cubin + PTX, which runs on GB10.

## What a new box should see after `pixi install`

```bash
pixi list | grep -E 'libcudf|librmm|cccl |cuda-nvcc |gxx |libnvcomp-dev'
```

`libcudf` and `librmm` must be 25.12, `cccl` must be 3.1.x, `cuda-nvcc` 12.9.x.
If `cccl` is 2.8, you pulled `cuda-cccl`; the compile will fail on
`cuda::mr::synchronous_resource`.
