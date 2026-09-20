# NCCL vs NIXL status (handoff for a new session)

Last updated: 2026-09-20 (E6, E7 added).

This is the continuation kit for **NIXL as GPU payload transport instead of NCCL, keep MPI**. Read it before changing communicator code. Do not assume GitHub has any of this: **NIXL work is uncommitted**.

| Item | Value |
|---|---|
| Repo | `/home/aocsa/git/distributed-join` |
| Branch | `pixi-cuda12-rapids2512` |
| Last **committed** SHA | `fdcdae6e5e5ed431df28b3257ace740ebdb9358f` (“Add a two-node MPI launcher for zeno-01 and zeno-02”) |
| Launch node | `pdx02-zeno-01` (`10.87.131.64`) |
| Peer node | `pdx02-zeno-02` (`10.87.131.68`) |
| GPU | NVIDIA GB10, linux-aarch64, driver 580.126.09, sm_121, 1 GPU/box |
| RAPIDS / CUDA | 25.12 / 12.9 via pixi |
| NIXL source / prefix | `/home/aocsa/git/nixl` / `/home/aocsa/git/nixl/install` |
| Prior chat | [NIXL vs NCCL experiments](8a61d744-fde4-4802-8c37-562cc22af8c9) |

Plans (do **not** edit when implementing):

- `/home/aocsa/.cursor/plans/nixl_five_path_experiments.plan.md`
- `/home/aocsa/.cursor/plans/nixl_pool-register_path_7aede9a9.plan.md`

---

## 0. Goal, keep/revert rule, current verdict

**Goal:** GPU payload over NIXL; MPI stays the launcher and dest-VA side channel; NCCL stays in the tree; default `--communicator` stays **UCX**. Compare NCCL / NIXL / UCX on the same two GB10 boxes.

**Keep/revert rule (used for every experiment):** keep a NIXL change only if join mean **or** 4 GB bandwidth improves by **> ~10%** **and** NCCL/UCX stay within ~10% of the then-current baseline. Otherwise revert; do not stack a failed change into the next path.

**Verdict (updated 2026-09-20, E6):** the 0.12 GB/s was never a NIXL plugin problem. GB10 has **no GPUDirect RDMA and no dmabuf** (`CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED=0`, `GPU_DIRECT_RDMA_SUPPORTED=0`), so the NIC cannot register `cudaMalloc` memory and UCX `ucp_put` from VRAM degrades to 8 KiB bounce copies on **both** `tcp` and `rc`. Standalone `ucx_perftest ucp_put_bw` over `rc`: CUDA 0.70 GB/s, managed 0.62 GB/s, host **12.4 GB/s**. **E6 (kept, opt-in `NIXL_HOST_POOL=1`)** backs the RMM pool with pinned host memory registered as `DRAM_SEG`: join 0.244 → **~0.18 s** (0.14–0.23 over 5 runs), all_to_all 1–2 GB 0.12 → **10–12 GB/s** over pinned `rc`. See §7a. Earlier E1–E5 conclusions still hold (all reverted); the "packed AM" nsys reading was the RMA emulation fallback, not a plugin bug.

---

## 1. Current numbers (kept tree)

Same fabric unless noted: `UCX_TLS=tcp,cuda_copy`, `UCX_MEMTYPE_CACHE=n`, 1 rank/GPU/node. Join is 1M×1M rows (`--build-table-nrows 1000000 --probe-table-nrows 1000000`). all_to_all `--repeat 4`. Means of 3 timed runs unless noted. Join log line is misspelled `Elasped time (s)`.

| Stage | NIXL join | NIXL 4 GB | NCCL join | NCCL 4 GB | UCX join | UCX 4 GB |
|---|---|---|---|---|---|---|
| First NIXL cut (`registerMem` every `stop()`) | 0.289 s | ~0.12 GB/s | 0.051 s | ~5.3 GB/s | 0.059 s | ~2.0 GB/s |
| **Pool-register (kept, experiment baseline)** | **0.235 s** | **~0.12 GB/s** | **0.051 s** | **~6 GB/s** | **0.058 s** | **~2.0 GB/s** |
| After E1–E5 (all discarded) | ~0.23–0.26 s | ~0.12 GB/s | ~0.049–0.053 s | ~5.5–6.2 GB/s | ~0.059 s | ~2.0 GB/s |

**Vs first NIXL cut:** join ~19% faster (pool-register). 4 GB bandwidth **unchanged**.  
**Vs pool-register baseline:** E1–E5 added **0%**.  
**Vs NCCL now:** join **~4.6× slower**; 4 GB **~50× slower**.

Opt-in **UCX + RC** (not default, not for NIXL): `UCX_TLS=rc,cuda_copy,tcp` + `UCX_NET_DEVICES=rocep1s0f0:1` → UCX 4 GB **~7.3 GB/s**, join ~0.047 s. NIXL on the same TLS stayed ~0.10 GB/s.

Older logs (not in git): `/tmp/nixl_exp_e1_rndv8192/`, `/tmp/nixl_exp_e2_tls/`, `/tmp/nixl_exp_e2_rc/`, `/tmp/nixl_exp_e2_rc_clean/`, `/tmp/nixl_exp_e3_chunk/`, `/tmp/nixl_exp_e3_64m/`, `/tmp/nixl_exp_e3_128m/` (if present), `/tmp/nixl_exp_e3_256m/`, `/tmp/nixl_exp_e4_join/`, `/tmp/nixl_exp_e5/`. Re-run protocol: `repro/compare_communicators.sh /tmp/nixl_exp_label`.

---

## 2. Hardware / fabric (do not rediscover)

| | zeno-01 | zeno-02 |
|---|---|---|
| Hostname | `pdx02-zeno-01` | `pdx02-zeno-02` |
| Interconnect IPv4 | `10.87.131.64` on `enp1s0f0np0` | `10.87.131.68` on `enp1s0f0np0` |
| RDMA device for that NIC | `rocep1s0f0` | `rocep1s0f0` |

- ConnectX-7 (MT4129 / `mlx5_core`), four RoCE ports, 200 Gb/s, link layer **Ethernet** (RoCE, not native IB LID).
- UCX 1.22 (pixi) transports: `rc_mlx5`, `rc_verbs`, `dc_mlx5`, `tcp`, `cuda_copy`, `cuda_ipc`. **No `gdr_copy`** (not in this UCX build; no `/dev/gdrdrv`; no `nvidia-peermem`).
- `ulimit -l` is **15688968 kB** (~15 GiB) and cannot be raised without privilege. Unpinning RC across all four HCAs hung UCX at 4 GB (`ibv_reg_mr` memlock). Pin `UCX_NET_DEVICES=rocep1s0f0:1` if you retry RC.
- Management NIC `enP7s7` is **not** the 64/68 path (zeno-01 `.181`, zeno-02 `.182`). Do not put that in the hostfile.
- Home directories are **not** shared. After every rebuild, rsync `build/` and NIXL `install/` to zeno-02. `mpirun` executes the **same absolute paths** on both ranks.
- SSH: `~/.ssh/id_ed25519`. `PRTE_MCA_plm_ssh_args` / `OMPI_MCA_plm_rsh_args` already in `repro/run_two_nodes.sh`. zeno-02 standby: `repro/ZENO02_STANDBY_PROMPT.md`. **Do not run `mpirun` on zeno-02.** Launch only from zeno-01.

Hostfile: `repro/hosts.zeno01-zeno02`. Launcher default `--host 10.87.131.64:1,10.87.131.68:1`.

---

## 3. Uncommitted files (the NIXL work)

`git status` as of this note (relative to `fdcdae6`):

```
M  CMakeLists.txt
M  benchmark/all_to_all.cpp
M  benchmark/distributed_join.cu
M  cmake/BuildHelpers.cmake
M  repro/run_two_nodes.sh
M  repro/README.md
M  src/setup.cpp
M  src/setup.hpp
?? cmake/FindNIXL.cmake
?? repro/NIXL_NCCL_STATUS.md
?? repro/sync_nixl_zeno02.py
?? repro/compare_communicators.sh
?? src/nixl_communicator.cpp
?? src/nixl_communicator.hpp
?? src/nixl_registered_memory_resource.hpp
```

`src/CMakeLists.txt` GLOBs `*.cpp` / `*.cu`; `nixl_communicator.cpp` is picked up automatically. Do not add a duplicate source list.

**Do not commit unless asked.** Do not put NIXL in `pixi.lock` unless CMake cannot find `/home/aocsa/git/nixl/install`.

---

## 4. NCCL path (unchanged, leave it)

Files: `src/communicator.hpp` (`NCCLCommunicator`), `src/communicator.cpp` (~799–875), `src/setup.cpp` NCCL branch.

1. MPI launches ranks. `ncclGetUniqueId` on rank 0, `MPI_Bcast`, `ncclCommInitRank`.
2. RMM pool is the **default** device resource (no NCCL preregistration).
3. `start()` → `ncclGroupStart()`.
4. `send` / `recv`: **256-byte-aligned RMM bounce**, `cudaMemcpyAsync` D2D into bounce, then `ncclSend` / `ncclRecv` of the aligned size (`ncclChar`). Comment in code says NCCL is slow on unaligned buffers.
5. `stop()` → `ncclGroupEnd()`, copy recvs out of bounce, free bounce, `cudaStreamSynchronize(comm_stream)`.
6. NCCL uses its **own** IB/RoCE/socket stack. `UCX_TLS` does **not** drive NCCL payload (only MPI bootstrap). On this fabric NCCL 4 GB is ~5.5–6.2 GB/s.

`group_by_batch()` is `false` (same as NIXL). Do not change NCCL as part of NIXL work.

---

## 5. NIXL path (what is in the tree now)

### Roles

| Path | Role |
|---|---|
| `src/nixl_communicator.hpp` / `.cpp` | `NIXLCommunicator`: NIXL UCX backend, one-sided `NIXL_WRITE` |
| `src/nixl_registered_memory_resource.hpp` | `cudaMalloc`/`cudaFree` + `registerMem(VRAM_SEG)` / `deregisterMem` |
| `src/setup.cpp` / `setup.hpp` | NIXL branch: init agent, pool on `memory_resource()`, then `publish_registered_memory()` |
| `cmake/FindNIXL.cmake` | finds `nixl.h` + `libnixl`, `nixl_build`, `nixl_common`, `serdes` |
| `CMakeLists.txt` | `find_package(NIXL REQUIRED)` — **configure fails without NIXL** |
| `cmake/BuildHelpers.cmake` | SYSTEM includes, link `NIXL_LIBRARIES`, `BUILD_RPATH` / `INSTALL_RPATH` |
| `benchmark/all_to_all.cpp` | `--communicator UCX\|NCCL\|NIXL`; SIZES 1 MB–4 GB; `REPEAT` default 4 |
| `benchmark/distributed_join.cu` | same communicator flag |
| `repro/run_two_nodes.sh` | two-node launcher; `COMMUNICATOR=`; optional `-x UCX_RNDV_THRESH` / `-x UCX_NET_DEVICES` |

Ctor/dtor of `NIXLCommunicator` are **out-of-line** in the `.cpp` (`unique_ptr` to incomplete `nixl_registered_memory_resource`). If you move them into the header, `setup.cpp` fails to compile.

`registered_mr` in `setup_memory_pool_and_communicator` is **nullptr** for NIXL (NIXL registers internally). Destroy path must tolerate that.

### Hot path (kept)

1. `initialize()`: `nixlAgent` (`use_prog_thread=true`) + `createBackend("UCX")` + require `VRAM_SEG` + `nixl_registered_memory_resource`.
2. Setup builds the RMM **pool on that MR**, then `publish_registered_memory()`: one `getLocalMD` / `MPI_Allgatherv` / `loadRemoteMD` + `makeConnection` per peer. **No bounce buffer.**
3. `send`/`recv` only record `{peer, ptr, nbytes}`.
4. `stop()`:
   - `cudaStreamSynchronize(cudaStreamDefault)`
   - `MPI_Alltoall` of dest counts + `MPI_Alltoallv` of `{addr, len, dev_id}`
   - group sends by dest
   - same-rank: `cudaMemcpyAsync` D2D on `comm_stream`
   - remote: one `nixlBasicDesc` per send into `src`/`dst` dlists, `createXferReq(NIXL_WRITE)`, then `wait_posted` (`postXferReq` all, poll `getXferStatus`, `releaseXferReq`)
   - `cudaStreamSynchronize(comm_stream)`
5. `finalize()`: drop MR (deregisters pool), `invalidateRemoteMD`, destroy stream.

### First cut vs kept (do not re-introduce the first cut)

First cut registered the payload (or a bounce) on **every** `stop()`, allgathered MD, then WRITE. Same ~0.12 GB/s; join worse (~0.289 s) because of setup tax. Pool-register removed that tax. **Do not spend more work on pool `registerMem` or bounce copies.**

### NIXL UCX plugin (why 4 GB stays packed)

`/home/aocsa/git/nixl/src/plugins/ucx/ucx_utils.cpp` around the UCX worker:

```text
config.modify("RNDV_THRESH", "inf");
```

`nixl::ucx::config::modify` **skips if the env var is already set** (`config.h`: “Modify the config if it is not already set via environment variable”). `modifyAlways` exists for `NET_DEVICES` when the plugin itself lists devices.

E1 exported `UCX_RNDV_THRESH=8192` through `mpirun -x`. NIXL WRITE **still** packed 8 KiB AM. So either WRITE does not use rndv on `tcp,cuda_copy`, or the plugin’s PUT path ignores this knob.

`createBackend("UCX")` uses `getPluginParams` defaults; it does **not** pass `UCX_TLS` as a NIXL backend param. Env `UCX_TLS` is supposed to apply to the plugin’s UCX context; E2 showed NIXL did not pick up RC the way in-tree UCX tag send did.

---

## 6. UCX communicator (control, not the NIXL backend)

In-tree UCX tag send, **preregistered** pool (`registered_memory_resource` + `ucp_mem_map`). Default TLS `tcp,cuda_copy` → ~2 GB/s at 4 GB (rendezvous; nsys: 8 `cuMemcpyAsync` + `cuMemHostRegister` at 1 GB).

Same TLS with NIXL WRITE → ~0.12 GB/s (eager/packed).

These are **two different UCX contexts**. Do not assume launcher `UCX_TLS` / `UCX_RNDV_THRESH` that help `UCXCommunicator` also help NIXL WRITE.

Opt-in RC for **in-tree UCX only** (do **not** default this; do **not** use it as a NIXL “win”):

```bash
export UCX_TLS=rc,cuda_copy,tcp
export UCX_NET_DEVICES=rocep1s0f0:1
COMMUNICATOR=UCX ./repro/run_two_nodes.sh all_to_all
```

`gdr_copy` is not available (`UCX WARN transport 'gdr_copy' is not available`; run continues without it).

---

## 7. Experiments E1–E5 (all discarded except launcher `-x` forwarding)

Protocol (repeat this, do not invent a new one):

```bash
# after rebuild + python3 repro/sync_nixl_zeno02.py
repro/compare_communicators.sh /tmp/nixl_exp_label
# or by hand:
COMMUNICATOR=NCCL ./repro/run_two_nodes.sh all_to_all
COMMUNICATOR=NCCL ./repro/run_two_nodes.sh join
COMMUNICATOR=NIXL ./repro/run_two_nodes.sh all_to_all
COMMUNICATOR=NIXL ./repro/run_two_nodes.sh join
COMMUNICATOR=UCX  ./repro/run_two_nodes.sh all_to_all
COMMUNICATOR=UCX  ./repro/run_two_nodes.sh join
```

Keep/revert bar: >~10% NIXL join **or** 4 GB, without regressing NCCL/UCX ~10%.

| ID | Change | Result | Keep? |
|---|---|---|---|
| E1 | `UCX_RNDV_THRESH=8192` (`-x` if set) | NIXL 4 GB still ~0.13 GB/s; join 0.279 s (worse) | **No default thresh.** Keep optional `-x` in the launcher |
| E2 | Probe RC/GDR; try `rc,cuda_copy,gdr_copy,tcp` then `rc,cuda_copy,tcp` | RC exists (RoCE). `gdr_copy` missing. NIXL 4 GB ~0.10 GB/s. UCX RC ~7.3 GB/s if NIC pinned; unpinned 4 GB hung on memlock. Two concurrent compares made the hang worse | **Do not default `UCX_TLS=rc`.** Optional `-x UCX_NET_DEVICES` kept |
| E3 | Chunk `NIXL_WRITE` 64 / 128 / 256 MiB (one dlist, multiple `ucp_put`) | 4 GB still ~0.12–0.126 GB/s; join flat/worse | **Reverted** (one desc per send) |
| E4 | Drop `cudaStreamSynchronize(cudaStreamDefault)` at `stop()`; reuse `createXferReq` when dest/size match | Join 0.260 s (worse); 4 GB 0.123 GB/s | **Reverted** |
| E5 | `nsys` NIXL vs UCX **1 GB** all_to_all, `REPEAT=1`, `cudaProfilerApi` ranges | See §8 | No code keep; **SIZES restored** to 1 MB–4 GB |

Worth keeping in git (when asked to commit): pool-register NIXL, `FindNIXL.cmake`, launcher `NIXL_ROOT`/`NIXL_PLUGIN_DIR` + optional RNDV/NET_DEVICES forwarding, default TLS `tcp,cuda_copy`.

Discarded: E1 8192 default, E2 RC-as-default, E3 chunks, E4 reuse/nosync, first-cut per-stop `registerMem`.

---

## 7a. E6: pinned host pool (kept, opt-in)

Baseline rerun 2026-09-20 (`/tmp/nixl_exp_rerun_0920/parsed.txt`) matched §1: NCCL 0.051 s / 6.0 GB/s, NIXL 0.244 s / 0.12 GB/s, UCX 0.059 s / 2.0 GB/s.

Root cause probe (`/tmp/dmabuf_probe.cu`): GB10 is `INTEGRATED=1`, `PAGEABLE_MEMORY_ACCESS=1`, `DMA_BUF_SUPPORTED=0`, `GPU_DIRECT_RDMA_SUPPORTED=0`; `cuMemGetHandleForAddressRange(DMA_BUF_FD)` returns invalid argument. `ucx_perftest -t ucp_put_bw -m {cuda,cuda-managed,host}` over `rc`, `rocep1s0f0:1`, 64 MB: 0.70 / 0.62 / 12.4 GB/s.

Change (`src/nixl_communicator.{hpp,cpp}`, `src/nixl_registered_memory_resource.hpp`, `src/setup.cpp`, `repro/run_two_nodes.sh`):

- `NIXL_HOST_POOL=1` → pool upstream uses `cudaMallocHost`/`cudaFreeHost`, registers and builds xfer dlists as `DRAM_SEG` (dev id 0), same-rank copy uses `cudaMemcpyDefault`. Default stays VRAM.
- Pool capped at `NIXL_HOST_POOL_GIB` (default 6): pinned pages and `ibv_reg_mr` of the same range both count against `ulimit -l` (15688968 kB). 12 GiB failed with `ibv_reg_mr ... Cannot allocate memory`.
- Launcher forwards `NIXL_HOST_POOL`, `NIXL_HOST_POOL_GIB`.

| Config | Join (3 runs) | a2a 64 MB | a2a 1 GB | a2a 2 GB | a2a 4 GB |
|---|---|---|---|---|---|
| NIXL host pool, `rc,cuda_copy,tcp` + `rocep1s0f0:1` | 0.208 / 0.167 / 0.175 / 0.140 / 0.232 → **mean 0.184 s**, noisy | 2.2–8.7 GB/s | 9.9–10.7 GB/s | 9.3–12.1 GB/s | RMM `Maximum pool size exceeded` (needs ~8 GB) |
| NIXL host pool, `tcp,cuda_copy` | 0.200 s | | | | |
| NCCL / UCX with the E6 binary | 0.050 s / 0.060 s | | | | unchanged |

Keep/revert bar: join −25% (mean), a2a ×80–100, NCCL/UCX unchanged → **kept**.

Still open: join is ~3.5× NCCL (0.18 vs 0.051) and run-to-run noisy. The all-to-all is now faster than NCCL, so the gap is cuDF compute on pinned host memory and/or NIXL `stop()` overhead (MPI dest exchange + per-dest xfer req), not transport. Next layer: `report_timing` breakdown of the join, then try `cudaHostAlloc` flags / a hybrid (VRAM compute, pinned staging for the shuffle only).

---

## 7b. E7: NIXL `stop()` never waited for incoming writes (correctness fix, kept)

Found by running `test_shuffle_on` with `--communicator NIXL` (the test now accepts the flag and does a real key check; the old `assert` was compiled out in Release). Uncompressed shuffle passed, the **compressed** case failed in 5 of 8 runs: `FAIL: rank 0 received key -9154409 with hash -1` or nvCOMP `Cuda Error: 9: 'invalid configuration argument'`, on both VRAM and host pools.

Cause: `stop()` posted its own WRITEs, waited for **local** completion, then returned. Nothing told the receiver that the peer's WRITEs into its recv buffers had landed. The join hid it because cuDF work follows the shuffle; decompression reads the recv buffer immediately.

Fix (`src/nixl_communicator.cpp`): every `createXferReq` sets `hasNotif` + `notifMsg`; after `wait_posted`, `stop()` polls `agent->getNotifs` until one notification has arrived from every rank it posted recvs from (`notif_credits` carries surplus). 9/9 test cases pass on NIXL and NIXL-host over 3 runs each.

Cost: all earlier NIXL join numbers (§1, §7a) were measured **without** waiting for data arrival and are optimistic. With E7:

| Config | Join (3 runs) | Shuffle no-comp | Shuffle cascaded |
|---|---|---|---|
| NCCL | 0.054 s | 0.0005 s | 0.025 s |
| UCX | 0.060 s | 0.0020 s | 0.032 s |
| NIXL VRAM, tcp | 0.285 s (0.322 / 0.267 / 0.266) | 0.036 s | 0.101 s |
| NIXL host pool, rc pinned | 0.225 s (0.194 / 0.261 / 0.220) | 0.017 s | 0.081 s |

Reproduce: `repro/compare_join_shuffle.sh /tmp/nixl_exp_<label>`.

---

## 8. nsys (why 4 GB cannot be fixed in the adapter)

Command shape (also `/tmp/nixl_exp_e5/run_nsys.sh` if it still exists). Reports: `/tmp/nixl_exp_e5/NIXL_rank{0,1}.{1,2}.nsys-rep` (`.2` is the 1 GB range). `all_to_all` already wraps the timed loop in `cudaProfilerStart/Stop`.

```bash
# SIZES must be the full 1MB–4GB vector in all_to_all.cpp when you are done.
# E5 temporarily compiled 1GB-only; that is reverted.
NSYS=/usr/local/bin/nsys
mpirun --prefix "$CONDA_PREFIX" -np 2 --host 10.87.131.64:1,10.87.131.68:1 --map-by ppr:1:node \
  -x PATH -x LD_LIBRARY_PATH -x UCX_TLS -x UCX_MEMTYPE_CACHE -x UCX_WARN_UNUSED_ENV_VARS \
  -x NIXL_ROOT -x NIXL_PLUGIN_DIR \
  "$NSYS" profile --trace=cuda,nvtx,osrt,mpi --mpi-impl=openmpi \
    --capture-range=cudaProfilerApi --capture-range-end=repeat \
    --stats=true --force-overwrite=true \
    -o /tmp/nixl_exp_e5/${comm}_rank%q{PMIX_RANK} \
    ./build/bin/benchmark/all_to_all --communicator NIXL --repeat 1
```

**NIXL 1 GB:** wall ~7.1 s under nsys (~0.07 GB/s; ~0.12 GB/s unprofiled). **62,653 × 8 KiB D2H and 62,653 × 8 KiB H2D** (512 MB each way), **125,306 `cuMemcpyAsync`**. OS time in `send` / `recv` / `epoll` / `poll`. Packed **active message + host staging**, not `ucp_put` RMA.

**UCX 1 GB:** wall ~0.35 s / ~1.5 GB/s under nsys. **8 `cuMemcpyAsync`** + `cuMemHostRegister` (rendezvous).

NIXL NVTX plugin was missing (`libtrace_backend_nvtx.so`); ignore missing NVTX rows.

Plan table for “where time is”:

| Where time is | Next layer |
|---|---|
| `cudaMemcpy` / packed AM | **This is where we are.** E1/E2 were the right layer; NIXL WRITE still not using rndv/RMA on `tcp,cuda_copy` |
| `createXferReq` / MPI | more E4 (already tried, did not help) |
| `ucp_put` / RMA | GDR/RC for **NIXL’s** UCX context — not more C++ in `nixl_communicator.cpp` |

---

## 9. NIXL install (not in pixi.lock)

**Do not add NIXL to `pixi.lock` unless CMake cannot find it.** CMake is `find_package(NIXL REQUIRED)` via `cmake/FindNIXL.cmake`. Search order:

1. `-DNIXL_ROOT=...` or `$NIXL_ROOT`
2. `/home/aocsa/git/nixl/install`
3. `/opt/nvidia/nvda_nixl`

Local build: `/home/aocsa/git/nixl`, meson **1.5.0**, prefix **`/home/aocsa/git/nixl/install`**. Python bindings were **skipped** (`# subdir('bindings')` in NIXL `src/meson.build`) because `python3-dev` is missing. That is a **local NIXL tree edit**, not in this repo.

Plugins: `$NIXL_ROOT/lib/plugins/libplugin_UCX.so`. Launcher sets `NIXL_ROOT`, `NIXL_PLUGIN_DIR`, prepends `$NIXL_ROOT/lib` to `LD_LIBRARY_PATH`, all forwarded with `mpirun -x`.

`porting/build_and_test.sh` does **not** export `NIXL_ROOT`; FindNIXL’s hard-coded hint still finds the prefix. After a clean CMake cache, if configure fails, `export NIXL_ROOT=/home/aocsa/git/nixl/install`.

Need `libnixl`, `libnixl_build`, `libnixl_common`, `libserdes`.

---

## 10. Build, sync, run (from zeno-01 only)

pixi env: RAPIDS **25.12**, CUDA **12.9**, C++20, nvCOMP 5, CCCL 3.1 (see `repro/NOTES.md`). `eval "$(pixi shell-hook)"` under `set +u`. After `pixi install`, `porting/patch_nvcc_activate.sh`. Do not use `/usr/local/cuda` (often 13.x on these images). `which nvcc` must be under `.pixi`.

Do **not** export `UCX_ROOT`; UCX treats every `UCX_*` variable as its own. `CMAKE_PREFIX_PATH=$CONDA_PREFIX` is enough.

```bash
cd /home/aocsa/git/distributed-join
porting/patch_nvcc_activate.sh
set +u; eval "$(pixi shell-hook)"; set -u
# first configure (or if FindNIXL / flags changed):
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target distributed all_to_all distributed_join

# zeno-02 has no shared home
python3 repro/sync_nixl_zeno02.py

COMMUNICATOR=NCCL ./repro/run_two_nodes.sh join
COMMUNICATOR=NIXL ./repro/run_two_nodes.sh all_to_all
```

Binaries:

- `build/bin/benchmark/distributed_join`
- `build/bin/benchmark/all_to_all`
- `build/bin/test/test_shuffle_on`

Join defaults: 1M×1M rows (printed banner talks about 2M rows **total** across 2 ranks). Override `BUILD_NROWS` / `PROBE_NROWS`. all_to_all `REPEAT` override: `REPEAT=1 ./repro/run_two_nodes.sh all_to_all`. SIZES: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096 MB. Warmup buffer 4 MB.

Verify both ranks see the same binary after every sync. Cursor’s shell denylists `ssh`/`ldd`/`ip`; use Python:

```python
# md5 of all_to_all on both nodes; they must match.
# Last kept rebuild (historical): 6488442ad131676cdeea64433aa14cd3
# Re-check after every rebuild; do not trust that hash forever.
```

Remote `all_to_all` must `NEEDED libnixl.so` and contain `NIXLCommunicator::stop`. Rank 1 must load `libplugin_UCX.so`. Join/all_to_all logs print `Communicator: NIXL`.

Do **not** use `python -c '...'` over ssh with nested quotes (that failed). Prefer `ssh … md5sum /path` via `/usr/bin/ssh` in a Python argv list.

---

## 11. Cursor / box pitfalls (already burned)

- Cursor agent shell **denylists** `ssh`, `ldd`, `ip`. Use `python3` + `/usr/bin/ssh` / `/usr/bin/rsync` (`repro/sync_nixl_zeno02.py`).
- `cmake --build` can look “failed” after ~30s even when Ninja later links all 28 targets. Check the Ninja log / binaries before rebuilding from scratch.
- Incomplete type: `NIXLCommunicator` ctor/dtor **must** stay in the `.cpp`.
- Two concurrent RC compares hung UCX / leaked MPI ranks. If a 4 GB RC run hangs, abort leftover ranks before starting another.
- Unpinned `UCX_TLS=rc` at 4 GB hung on `ibv_reg_mr` memlock (`ulimit -l` ~15 GB, four RoCE HCAs). Pin `UCX_NET_DEVICES=rocep1s0f0:1`.
- Do not default `UCX_RNDV_THRESH=inf` in the launcher (would regress in-tree UCX tag send). Optional `-x` when the var is set is enough.
- Do not default `UCX_TLS=rc` (hurts NIXL, can hang UCX).
- GB10 unified memory: `recommended_rmm_pool_size()` caps the pool so the CPU keeps RAM. Do not size the pool to 90% of `cudaMemGetInfo`.
- CCCL 3.1 vs toolkit 2.8: `cmake/BuildHelpers.cmake` puts `$CONDA_PREFIX/include/cccl` as a **non-system** `-I` first. Do not add `cuda-cccl` to pixi.
- `pixi run build` can still exit non-zero after a successful link because pixi deactivates and hits `NVCC_PREPEND_FLAGS` nounset. Use the patch + `cmake --build`.
- `eval "$(pixi shell-hook)"` must run under `set +u`.
- Open MPI 5 / PRTE: `PRTE_MCA_plm_ssh_args` (not only `OMPI_MCA_plm_rsh_args`).
- all_to_all E5 temporarily compiled 1 GB-only SIZES; **restored**. Only leftover `all_to_all.cpp` intent vs HEAD is the `--communicator` / MPI-launcher comment.

---

## 12. Constraints / do not undo

- Keep **MPI** as launcher and dest-VA side channel.
- Leave **NCCL** in place; default communicator **UCX**.
- Do **not** put NIXL in `pixi.lock` unless configure cannot find the meson prefix.
- Do **not** default `UCX_TLS` to `rc`.
- Do **not** default `UCX_RNDV_THRESH`.
- Do **not** commit unless asked.
- `group_by_batch()` stays `false` for NIXL/NCCL.
- Do not edit the plan files when implementing.
- Out of scope previously: changing NCCL, etcd, swapping MPI, changing default UCX tests, Python NIXL bindings.

---

## 13. What a new session should try next (if the goal is still NIXL bandwidth)

**Bandwidth is solved by E6 (§7a) for GB10; start from the join gap there.** The list below predates E6 and is kept for history; items 1–3 are moot on GB10 (no GDR hardware path at all).

Promising remaining layers, in order (historical):

1. **Make NIXL WRITE use rendezvous/RMA** on this UCX. Confirm whether `UCX_RNDV_THRESH` is visible inside the plugin worker (E1 set it and WRITE still packed). Likely a NIXL UCX plugin / `ucp_put_nbx` change in `/home/aocsa/git/nixl`, not more C++ here.
2. **NIXL + RC** only if NIXL’s own UCX context actually selects `rc_mlx5` and `ucp_put` (E2 showed env `UCX_TLS=rc` did **not** move NIXL). Pin `UCX_NET_DEVICES=rocep1s0f0:1`. Need `ulimit -l unlimited` or stay on one HCA.
3. **gdrcopy / nvidia-peermem** — not present; installing it is a box-admin change, not a git change.
4. Optional: in-tree UCX can already use RC for non-NIXL runs (~7 GB/s). That does not fix NIXL.

---

## 14. RAPIDS / GB10 port (needed to build at all)

See `repro/NOTES.md` and `repro/README.md`. Short: pixi platforms include `linux-aarch64`; RAPIDS 25.12 / CUDA 12.9; CCCL 3.1 before toolkit 2.8; `NVCC_PREPEND_FLAGS` nounset patch; GPU_ARCHS `80;90;120;121`; nvCOMP 5; rmm `allocate(stream, bytes)`; string chars in parent buffer; `recommended_rmm_pool_size()` cap.

---

## 15. Fresh-session checklist

1. `hostname` is `pdx02-zeno-01` to launch. zeno-02 agent stays idle (SSH authorized).
2. `cd /home/aocsa/git/distributed-join && git status` — uncommitted NIXL files still exist; branch `pixi-cuda12-rapids2512`.
3. `export PATH="$HOME/.pixi/bin:$PATH"`; `porting/patch_nvcc_activate.sh`; `set +u; eval "$(pixi shell-hook)"; set -u`. `which nvcc` under `.pixi`.
4. `ls /home/aocsa/git/nixl/install/lib/plugins/libplugin_UCX.so` and `ls /home/aocsa/git/nixl/src/meson.build` (`bindings` still commented).
5. Rebuild + `python3 repro/sync_nixl_zeno02.py`.
6. `COMMUNICATOR=NIXL ./repro/run_two_nodes.sh join` — expect ~0.23–0.26 s, `Communicator: NIXL`.
7. `COMMUNICATOR=NCCL ./repro/run_two_nodes.sh join` — expect ~0.05 s (control: NCCL still works).
8. Do not start from bounce/`registerMem`-every-stop. Start from packed-AM vs RMA in NIXL’s UCX plugin.
