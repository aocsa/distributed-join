#!/usr/bin/env bash
# Compatibility wrapper. The canonical new-box entry is repro/bootstrap.sh.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || true)"
if [[ -n "${here}" && -x "${here}/../repro/bootstrap.sh" ]]; then
  exec "${here}/../repro/bootstrap.sh" "$@"
fi
if [[ -x ./repro/bootstrap.sh ]]; then
  exec ./repro/bootstrap.sh "$@"
fi
exec bash -c "$(curl -fsSL https://raw.githubusercontent.com/aocsa/distributed-join/pixi-cuda12-rapids2512/repro/bootstrap.sh)"
