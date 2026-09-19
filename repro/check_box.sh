#!/usr/bin/env bash
# Preflight for the RAPIDS 25.12 pixi build. Safe to run before or after clone.
# Exit 0 if this machine looks capable; non-zero with a reason otherwise.
set -euo pipefail

fail=0
arch="$(uname -m)"
echo "host $(hostname)"
echo "uname $(uname -a)"
echo "arch ${arch}"

if [[ "${arch}" != "x86_64" && "${arch}" != "aarch64" ]]; then
  echo "unsupported arch ${arch} (need x86_64 or aarch64)" >&2
  fail=1
fi

for cmd in git curl; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "missing: ${cmd}" >&2
    fail=1
  else
    echo "${cmd}: $(command -v "${cmd}")"
  fi
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "missing: nvidia-smi (install NVIDIA driver >= 525)" >&2
  fail=1
else
  nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv || fail=1
fi

if ! command -v pixi >/dev/null 2>&1 && [[ -x "${HOME}/.pixi/bin/pixi" ]]; then
  export PATH="${HOME}/.pixi/bin:${PATH}"
fi
if command -v pixi >/dev/null 2>&1; then
  echo "pixi: $(pixi --version) ($(command -v pixi))"
else
  echo "pixi: not installed (bootstrap will install it)"
fi

avail_kb="$(df -Pk . | awk 'NR==2 { print $4 }')"
echo "free_disk_miB $((avail_kb / 1024))"
if (( avail_kb < 8 * 1024 * 1024 )); then
  echo "need about 8 GiB free for .pixi + build" >&2
  fail=1
fi

if [[ -f pixi.toml ]]; then
  if grep -q 'linux-aarch64' pixi.toml && grep -q 'libcudf = "25.12' pixi.toml; then
    echo "pixi.toml: linux-aarch64 + libcudf 25.12 present"
  else
    echo "pixi.toml does not look like the RAPIDS 25.12 port" >&2
    fail=1
  fi
fi

exit "${fail}"
