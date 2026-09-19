#!/usr/bin/env bash
# Build with Ninja keep-going and print distinct compiler errors ranked by frequency.
# Usage: porting/summarize_build_errors.sh
set -uo pipefail
cd "$(dirname "$0")/.."
eval "$(pixi shell-hook)"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -- -k 0 > /tmp/build.log 2>&1
echo "failed targets: $(grep -cE '^FAILED' /tmp/build.log)"
grep -E "error|warning" /tmp/build.log | grep -v " -c " \
  | sed "s|$PWD/||g; s|\.pixi/envs/default/include/||g" \
  | sort | uniq -c | sort -rn | head -60
