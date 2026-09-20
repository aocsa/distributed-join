#!/usr/bin/env bash
# 1 warmup + 3 timed runs of join and shuffle for NCCL, UCX, NIXL (VRAM pool, default tcp),
# NIXL-host (pinned host pool, rc pinned to rocep1s0f0).
set -euo pipefail
LOGDIR="${1:?logdir}"; mkdir -p "$LOGDIR"
cd /home/aocsa/git/distributed-join
export PATH="$HOME/.pixi/bin:$PATH"
run_cfg() { # name kind tag
  local name=$1 kind=$2 tag=$3
  case $name in
    NCCL|UCX|NIXL) env -u NIXL_HOST_POOL -u UCX_NET_DEVICES UCX_TLS=tcp,cuda_copy COMMUNICATOR=$name ./repro/run_two_nodes.sh $kind ;;
    NIXLhost) NIXL_HOST_POOL=1 UCX_TLS=rc,cuda_copy,tcp UCX_NET_DEVICES=rocep1s0f0:1 COMMUNICATOR=NIXL ./repro/run_two_nodes.sh $kind ;;
  esac > "$LOGDIR/${kind}_${name}_${tag}.log" 2>&1 || echo "FAILED $kind $name $tag rc=$?" | tee -a "$LOGDIR/summary.txt"
}
echo "start $(date -Iseconds)" | tee "$LOGDIR/summary.txt"
for name in NCCL UCX NIXL NIXLhost; do
  for kind in join shuffle; do
    for tag in warmup run1 run2 run3; do
      echo "===== $kind $name $tag $(date -Iseconds)" | tee -a "$LOGDIR/summary.txt"
      run_cfg $name $kind $tag
    done
  done
done
echo "done $(date -Iseconds)" | tee -a "$LOGDIR/summary.txt"
python3 - "$LOGDIR" <<'PY'
import re, sys, pathlib, statistics
d = pathlib.Path(sys.argv[1]); out = []
def grab(p, rx):
    m = re.findall(rx, p.read_text(errors='replace')) if p.exists() else []
    return [float(x) for x in m]
for name in ('NCCL','UCX','NIXL','NIXLhost'):
    j = [grab(d/f'join_{name}_{t}.log', r'Elasped time \(s\)\s+([0-9.]+)') for t in ('run1','run2','run3')]
    j = [x[0] for x in j if x]
    sf, st, ok = [], [], 0
    for t in ('run1','run2','run3'):
        p = d/f'shuffle_{name}_{t}.log'
        f = grab(p, r'Shuffle time \(s\) ([0-9.]+) compression=false'); tt = grab(p, r'Shuffle time \(s\) ([0-9.]+) compression=true')
        if len(f) >= 2: sf.append(f[-1])   # f[0] is the in-process warmup
        if tt: st.append(tt[-1])
        ok += len(re.findall(r'passes successfully', p.read_text(errors='replace'))) if p.exists() else 0
    fmt = lambda v: f"{statistics.mean(v):.4f} ({' '.join(f'{x:.4f}' for x in v)})" if v else 'missing'
    out.append(f'{name:9s} join={fmt(j)}  shuffle_nocomp={fmt(sf)}  shuffle_comp={fmt(st)}  passes={ok}/9')
text = '\n'.join(out) + '\n'; (d/'parsed.txt').write_text(text); print(text)
PY
