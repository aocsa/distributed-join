#!/usr/bin/env bash
# 1 warmup + 3 timed runs of all_to_all and join for NCCL, NIXL, UCX.
# Usage: repro/compare_communicators.sh /tmp/nixl_exp_label [label]
set -euo pipefail
LOGDIR="${1:?logdir}"
LABEL="${2:-compare}"
mkdir -p "$LOGDIR"
unset UCX_RNDV_THRESH || true
unset UCX_NET_DEVICES || true
export UCX_TLS="${UCX_TLS:-tcp,cuda_copy}"
export UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE:-n}"
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
echo "start $(date -Iseconds) UCX_TLS=$UCX_TLS $LABEL" | tee "$LOGDIR/summary.txt"
for comm in NCCL NIXL UCX; do
  for kind in all_to_all join; do
    for tag in warmup run1 run2 run3; do
      echo "===== $kind $comm $tag $LABEL $(date -Iseconds) =====" | tee -a "$LOGDIR/summary.txt"
      COMMUNICATOR="$comm" ./repro/run_two_nodes.sh "$kind" \
        >"$LOGDIR/${kind}_${comm}_${tag}.log" 2>&1
    done
  done
done
echo "done $(date -Iseconds)" | tee -a "$LOGDIR/summary.txt"
python3 - "$LOGDIR" <<'PY'
import re, sys, pathlib, statistics
logdir = pathlib.Path(sys.argv[1])
def join_time(p):
    m = re.search(r'Elasped time \(s\)\s+([0-9.]+)', p.read_text(errors='replace'))
    return float(m.group(1)) if m else None
def a2a(p):
    rows = {}
    for line in p.read_text(errors='replace').splitlines():
        m = re.search(r'Size \(MB\): (\d+), Elasped time \(s\): ([0-9.eE+-]+), Bandwidth per GPU \(GB/s\): ([0-9.eE+-]+)', line)
        if m:
            rows[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    return rows
out=['=== parsed means (timed run1-3) ===']
for comm in ('NCCL','NIXL','UCX'):
    j=[]
    for tag in ('run1','run2','run3'):
        p=logdir/f'join_{comm}_{tag}.log'
        j.append(join_time(p) if p.exists() else None)
    valid=[x for x in j if x is not None]
    out.append(f'join {comm}: {j} mean={statistics.mean(valid) if valid else None}')
    for size in (64,1024,4096):
        bws=[]; ts=[]
        for tag in ('run1','run2','run3'):
            p=logdir/f'all_to_all_{comm}_{tag}.log'
            rows=a2a(p) if p.exists() else {}
            if size in rows:
                ts.append(rows[size][0]); bws.append(rows[size][1])
        if bws:
            out.append(f'  a2a {comm} {size}MB t={ts} mean_t={statistics.mean(ts):.4f} bw={bws} mean_bw={statistics.mean(bws):.4f}')
        else:
            out.append(f'  a2a {comm} {size}MB missing')
text='\n'.join(out)+'\n'
(logdir/'parsed.txt').write_text(text)
print(text)
PY
