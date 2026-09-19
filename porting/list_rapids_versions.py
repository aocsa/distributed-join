#!/usr/bin/env python3
"""Print every libcudf CUDA 12 build on the rapidsai channel with its rmm and nvcomp pins.

Used to pick the oldest RAPIDS release that supports CUDA 12 (23.08).
"""
import json
import urllib.request

URL = "https://conda.anaconda.org/rapidsai/linux-64/repodata.json"

with urllib.request.urlopen(URL) as resp:
    data = json.load(resp)

records = list(data.get("packages", {}).values()) + list(data.get("packages.conda", {}).values())
pins = {}
for rec in records:
    if rec["name"] != "libcudf" or "cuda12" not in rec["build"]:
        continue
    deps = [d for d in rec["depends"] if d.split()[0] in ("nvcomp", "libnvcomp", "libnvcomp-dev", "librmm", "cuda-version")]
    pins.setdefault(rec["version"], set()).add(" | ".join(deps))

for ver in sorted(pins, key=lambda s: [int(x) for x in s.split(".")]):
    print(ver, sorted(pins[ver])[0])
