# Build, run, and reproduce the two-node shuffle / join on zeno-01 + zeno-02

Last verified: 2026-09-20 on branch `pixi-cuda12-rapids2512` (uncommitted NIXL
work present, see `NIXL_NCCL_STATUS.md`).

Two SSH terminals: one on **zeno-01** (does everything) and one on **zeno-02**
(only checks). All `mpirun` launches happen on zeno-01. Homes are **not**
shared, so every rebuild is followed by an rsync to zeno-02.

| | zeno-01 | zeno-02 |
|---|---|---|
| Hostname | `pdx02-zeno-01` | `pdx02-zeno-02` |
| Interconnect IP (`enp1s0f0np0`) | `10.87.131.64` | `10.87.131.68` |
| RDMA device | `rocep1s0f0` | `rocep1s0f0` |
| GPU | 1x GB10 (sm_121, aarch64) | 1x GB10 |
| Role | build, sync, launch | idle target |

`ssh zeno-02` from zeno-01 does **not** resolve. Use the IP `10.87.131.68`
(or `pdx02-zeno-02` if your DNS has it).

---

## 1. Terminal A (zeno-01): one-time setup

```bash
ssh pdx02-zeno-01
cd /home/aocsa/git/distributed-join
git status --short          # expect the NIXL files listed in NIXL_NCCL_STATUS.md section 3
export PATH="$HOME/.pixi/bin:$PATH"
```

Only if `.pixi/` is missing (fresh clone):

```bash
pixi install                      # from pixi.lock, do not `pixi update`
```

Every shell that builds or runs must activate the pixi env this way
(`set +u` is required, the hook is not nounset-safe):

```bash
porting/patch_nvcc_activate.sh
set +u; eval "$(pixi shell-hook)"; set -u
which nvcc                        # must be under .pixi/, never /usr/local/cuda
```

Passwordless SSH to zeno-02 must work with `~/.ssh/id_ed25519`:

```bash
/usr/bin/ssh -o BatchMode=yes -i ~/.ssh/id_ed25519 10.87.131.68 hostname
```

If that fails, follow `repro/ZENO02_STANDBY_PROMPT.md` on zeno-02.

## 2. Terminal B (zeno-02): make sure the box is idle

```bash
ssh pdx02-zeno-02
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv
pgrep -af 'all_to_all|distributed_join|test_shuffle_on' || echo idle
ip -br addr show enp1s0f0np0        # expect 10.87.131.68
```

Do **not** run `mpirun`, builds, or GPU jobs here. Leave this terminal open
for the checks in step 5.

## 3. Terminal A: build

```bash
cd /home/aocsa/git/distributed-join
# first configure, or after CMake/FindNIXL changes:
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target distributed all_to_all distributed_join test_shuffle_on
```

CMake needs NIXL. `cmake/FindNIXL.cmake` looks in `$NIXL_ROOT`, then
`/home/aocsa/git/nixl/install`. If configure fails:

```bash
export NIXL_ROOT=/home/aocsa/git/nixl/install
```

`cmake --build` sometimes prints a failure after ~30 s while Ninja is still
linking. Check that the binaries exist and are fresh before rebuilding:

```bash
ls -l build/bin/benchmark/all_to_all build/bin/benchmark/distributed_join build/bin/test/test_shuffle_on
```

## 4. Terminal A: sync to zeno-02

```bash
python3 repro/sync_nixl_zeno02.py
```

This rsyncs `build/` and the NIXL prefix `/home/aocsa/git/nixl/install/` to
the same absolute paths on zeno-02. Run it after **every** rebuild.

## 5. Terminal B: confirm both nodes run the same binary

Terminal A (zeno-01):

```bash
md5sum build/bin/benchmark/all_to_all build/bin/benchmark/distributed_join build/bin/test/test_shuffle_on
```

Terminal B (zeno-02):

```bash
md5sum /home/aocsa/git/distributed-join/build/bin/benchmark/all_to_all \
       /home/aocsa/git/distributed-join/build/bin/benchmark/distributed_join \
       /home/aocsa/git/distributed-join/build/bin/test/test_shuffle_on
ls /home/aocsa/git/nixl/install/lib/plugins/libplugin_UCX.so
```

The hashes must match. If they differ, rerun step 4.

## 6. Terminal A: run

`repro/run_two_nodes.sh` launches 1 rank per node with
`mpirun --host 10.87.131.64:1,10.87.131.68:1`. Defaults:
`UCX_TLS=tcp,cuda_copy`, `UCX_MEMTYPE_CACHE=n`, join 1M x 1M rows.

Hostnames first (proves MPI + SSH):

```bash
./repro/run_two_nodes.sh hostname
```

Shuffle correctness test (`test_shuffle_on`, 1M rows per rank, checks every
received key lands on the right rank, then reports the time of a warm run).
`--communicator` defaults to UCX; the launcher forwards `COMMUNICATOR`:

```bash
COMMUNICATOR=NCCL ./repro/run_two_nodes.sh shuffle
COMMUNICATOR=UCX  ./repro/run_two_nodes.sh shuffle
COMMUNICATOR=NIXL ./repro/run_two_nodes.sh shuffle
```

Expected output from rank 0 (first case is an in-process warmup):

```
Communicator: NCCL
Test case (1000000,false) passes successfully.
Shuffle time (s) 0.51 compression=false
Test case (1000000,false) passes successfully.
Shuffle time (s) 0.0005 compression=false
Test case (1000000,true) passes successfully.
Shuffle time (s) 0.025 compression=true
```

A wrong key prints `FAIL: rank N received key ...` and exits 1.

Join benchmark, one communicator at a time (the log line is spelled
`Elasped time (s)`):

```bash
COMMUNICATOR=NCCL ./repro/run_two_nodes.sh join
COMMUNICATOR=UCX  ./repro/run_two_nodes.sh join
COMMUNICATOR=NIXL ./repro/run_two_nodes.sh join
```

All-to-all bandwidth sweep, 1 MB to 4 GB, `--repeat 4` by default:

```bash
COMMUNICATOR=NCCL ./repro/run_two_nodes.sh all_to_all
REPEAT=1 COMMUNICATOR=NIXL ./repro/run_two_nodes.sh all_to_all
```

Other knobs: `BUILD_NROWS`, `PROBE_NROWS`, `HOSTS`. `UCX_RNDV_THRESH`,
`UCX_NET_DEVICES`, `NIXL_HOST_POOL`, `NIXL_HOST_POOL_GIB` are forwarded to
both ranks when set.

### NIXL fast path on GB10 (pinned host pool)

GB10 has no GPUDirect RDMA and no dmabuf export
(`CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED=0`), so the NIC cannot register
`cudaMalloc` memory and UCX RMA from VRAM degrades to 8 KiB bounce copies
(~0.12 GB/s) on both `tcp` and `rc`. `NIXL_HOST_POOL=1` backs the RMM pool
with pinned host memory registered as `DRAM_SEG`; the integrated GPU reads it
directly.

```bash
NIXL_HOST_POOL=1 UCX_TLS=rc,cuda_copy,tcp UCX_NET_DEVICES=rocep1s0f0:1 \
  COMMUNICATOR=NIXL ./repro/run_two_nodes.sh join
NIXL_HOST_POOL=1 UCX_TLS=rc,cuda_copy,tcp UCX_NET_DEVICES=rocep1s0f0:1 \
  REPEAT=1 COMMUNICATOR=NIXL ./repro/run_two_nodes.sh all_to_all
```

Rank 0 prints `NIXL pool memory: host`. The pool is capped at 6 GiB
(`NIXL_HOST_POOL_GIB`) because pinned pages **and** the NIC registration both
count against `ulimit -l` (15688968 kB, not raisable). The 4 GB all-to-all
size needs ~8 GB of pool and ends with an RMM `Maximum pool size exceeded`;
sizes up to 2 GB run.

Always pin `UCX_NET_DEVICES=rocep1s0f0:1` with `rc`. Unpinned RC across all
four RoCE ports hangs UCX on `ibv_reg_mr` memlock.

## 7. Full comparisons

Join + all_to_all, NCCL/NIXL/UCX, 1 warmup + 3 timed (~15 min, dominated by
the VRAM-pool NIXL all_to_all):

```bash
repro/compare_communicators.sh /tmp/nixl_exp_<label>
cat /tmp/nixl_exp_<label>/parsed.txt
```

Join + shuffle test, NCCL/UCX/NIXL/NIXL-host, 1 warmup + 3 timed (~7 min):

```bash
repro/compare_join_shuffle.sh /tmp/nixl_exp_<label>
cat /tmp/nixl_exp_<label>/parsed.txt
```

Never run two compares at once; the RC memlock hang gets worse.

## 8. Numbers measured 2026-09-20 (same fabric, 1 rank/GPU/node)

Join + shuffle test, means of 3 timed runs, all 9/9 test cases pass
(`/tmp/nixl_exp_joinshuffle_fixed_0920/parsed.txt`). NIXL numbers include the
sender-notification wait added the same day (see `NIXL_NCCL_STATUS.md` E7);
earlier NIXL join numbers were measured without it and are not comparable.

| Communicator | Join 1M x 1M | Shuffle 1M/rank, no compression | Shuffle, cascaded compression |
|---|---|---|---|
| NCCL | 0.054 s | 0.0005 s | 0.025 s |
| UCX `tcp,cuda_copy` | 0.060 s | 0.0020 s | 0.032 s |
| NIXL VRAM pool `tcp,cuda_copy` | 0.285 s | 0.036 s | 0.101 s |
| NIXL host pool `rc` pinned | 0.225 s | 0.017 s | 0.081 s |

All-to-all bandwidth (earlier the same day, before E7):

| Communicator | Join 1M x 1M | a2a 1 GB | a2a 2 GB | a2a 4 GB |
|---|---|---|---|---|
| NCCL | 0.051 s | 5.8 GB/s | | 6.0 GB/s |
| UCX `tcp,cuda_copy` | 0.059 s | 2.0 GB/s | | 2.0 GB/s |
| NIXL VRAM pool `tcp,cuda_copy` | 0.244 s | 0.13 GB/s | | 0.12 GB/s |
| NIXL host pool `rc` pinned | **0.14-0.23 s** (5 runs, mean 0.18) | **9.9-10.7 GB/s** | **9.3-12.1 GB/s** | does not fit 6 GiB pool |
| NIXL host pool `tcp,cuda_copy` | 0.200 s | | | |

Standalone `ucx_perftest -t ucp_put_bw` over `rc` at 64 MB, zeno-01 to zeno-02:
CUDA memory 0.70 GB/s, managed 0.62 GB/s, host 12.4 GB/s.

## 9. When something hangs or fails

Run on both boxes:

```bash
pgrep -af 'all_to_all|distributed_join|test_shuffle_on|ucx_perftest'
pkill -x all_to_all; pkill -x distributed_join
```

- `connect() failed` / rank never starts: wrong IP in `HOSTS`, or SSH key not
  authorized on zeno-02 (step 1).
- Binaries differ between nodes: step 4, then step 5.
- `ibv_reg_mr ... Cannot allocate memory`: memlock. Lower `NIXL_HOST_POOL_GIB`
  or pin `UCX_NET_DEVICES`.
- `NIXL UCX plugin does not advertise VRAM_SEG` or plugin not found: check
  `NIXL_PLUGIN_DIR=/home/aocsa/git/nixl/install/lib/plugins` on **both** nodes.
- CMake picks `/usr/local/cuda` (13.x): the pixi hook was not evaluated; redo
  the activation lines in step 1.
