# Reproduce distributed-join on a new NVIDIA box

This folder is the new-machine kit for the RAPIDS 25.12 / CUDA 12.9 port.
It is meant to run on a **fresh Linux box** with only an NVIDIA driver, git, and curl.

Proven on: NVIDIA GB10 (`linux-aarch64`, driver 580.126.09, compute 12.1) as
`pdx02-zeno-01`. The same lockfile also covers `linux-64`.

Branch: `pixi-cuda12-rapids2512` on <https://github.com/aocsa/distributed-join/>.

## One command (empty machine)

```bash
curl -fsSL https://raw.githubusercontent.com/aocsa/distributed-join/pixi-cuda12-rapids2512/repro/bootstrap.sh | bash
```

That script:

1. Checks for `nvidia-smi`, git, curl, and ~8 GiB free disk.
2. Installs [pixi](https://pixi.sh) if it is missing.
3. Clones this branch into `./distributed-join` (skipped if you already sit in the repo).
4. Runs `pixi install` from the committed `pixi.lock` (no re-solve).
5. Patches `cuda-nvcc`'s conda activate script so bash `set -u` does not abort.
6. Configures and builds with CMake/Ninja.
7. Runs the test suite (1 rank for the memory-hungry tests, 2 ranks for the rest).

Expect about 6 GiB in `./.pixi` plus a few minutes to compile. Override defaults:

```bash
REPO=https://github.com/aocsa/distributed-join.git \
BRANCH=pixi-cuda12-rapids2512 \
DIR=distributed-join \
SKIP_TESTS=0 \
bash repro/bootstrap.sh
```

Set `SKIP_TESTS=1` to stop after a successful build.

## Clone or existing checkout

Install on the box first (driver, git, curl, then pixi). Do not install a system
CUDA toolkit.

```bash
curl -fsSL https://pixi.sh/install.sh | bash
export PATH="$HOME/.pixi/bin:$PATH"   # this shell; install.sh only updates ~/.bashrc
```

New clone:

```bash
git clone --branch pixi-cuda12-rapids2512 https://github.com/aocsa/distributed-join.git
cd distributed-join
repro/check_box.sh
repro/bootstrap.sh          # pixi install from lockfile, patch, e2e tests
```

Already sitting in this repo: `repro/bootstrap.sh` (it will not nest another clone).

Or the same e2e by hand:

```bash
export PATH="$HOME/.pixi/bin:$PATH"
pixi install                      # from pixi.lock; do not pixi update
porting/build_and_test.sh 2       # nvcc nounset patch, clean build, MPI tests
```

`porting/build_and_test.sh` runs `porting/patch_nvcc_activate.sh` for you.
`pixi run build` **will fail** with `NVCC_PREPEND_FLAGS: unbound variable` unless
that patch has been applied. Success is the line `ALL DONE`.

## What this box must provide

| Need | Why |
|---|---|
| NVIDIA driver >= 525 | CUDA 12 user-mode driver; GB10 on this port used 580.126.09 |
| `linux-x86_64` or `linux-aarch64` | Declared pixi platforms |
| git, curl | Clone + pixi installer |
| ~8 GiB free disk | pixi env (~6 GiB) + build tree |
| Internet | conda-forge / rapidsai / nvidia channels |

Do **not** install a system CUDA toolkit. pixi brings CUDA 12.9, GCC 14, libcudf,
librmm, nvCOMP, NCCL, UCX, and Open MPI into `./.pixi`.

## Layout

| Path | Purpose |
|---|---|
| `repro/bootstrap.sh` | Fresh-box entry: preflight, clone, install, patch, build, test |
| `repro/check_box.sh` | Preflight only (driver, arch, disk) |
| `repro/capture_env.sh` | Dump host + `pixi list` for debugging a failed box |
| `repro/NOTES.md` | Why 25.12, source changes, pitfalls |
| `repro/ENV.md` | Pinned versions that built on GB10 |
| `repro/env/` | Frozen `pixi list` and host snapshot from `pdx02-zeno-01` |
| `repro/tools/` | Channel probe and keep-going error summarizer used during the port |
| `porting/patch_nvcc_activate.sh` | Nounset fix for cuda-nvcc activate |
| `porting/build_and_test.sh` | Clean build + MPI tests |
| `pixi.toml` / `pixi.lock` | Exact environment; do not re-solve on a new box |

## Tests the bootstrap runs

On a **one-GPU** box:

- `buffer_communicator` and `benchmark/distributed_join` (1M×1M rows) at **1 rank**
  (each sizes the RMM pool to a large fraction of free GPU memory).
- `test_shuffle_on`, `compare_against_single_gpu`, `compare_against_analytical`,
  `string_payload` at **2 ranks** with `mpirun --oversubscribe`.

If a test fails, run `repro/capture_env.sh` and compare against `repro/env/`.
See `NOTES.md` for the RAPIDS/CUDA/CCCL traps that already burned one GB10 box.
