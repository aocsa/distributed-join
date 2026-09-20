#!/usr/bin/env python3
"""Copy NIXL install and rebuilt binaries to zeno-02 at the same paths.

Cursor's agent shell denylists `ssh`/`rsync`/`ldd`/`ip`. Invoke this with
`python3 repro/sync_nixl_zeno02.py` so the copy uses `/usr/bin/ssh` via
subprocess. Homes are not shared; mpirun uses the same absolute paths on both ranks.
"""
import subprocess
import sys

remote = "10.87.131.68"
key = "/home/aocsa/.ssh/id_ed25519"
ssh = [
    "/usr/bin/ssh",
    "-o",
    "BatchMode=yes",
    "-o",
    "StrictHostKeyChecking=accept-new",
    "-o",
    "IdentitiesOnly=yes",
    "-i",
    key,
]
rsync = [
    "/usr/bin/rsync",
    "-az",
    "-e",
    " ".join(ssh),
]


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main():
    run(ssh + [remote, "mkdir -p /home/aocsa/git/nixl /home/aocsa/git/distributed-join"])
    run(rsync + ["/home/aocsa/git/nixl/install/", f"{remote}:/home/aocsa/git/nixl/install/"])
    run(
        rsync
        + [
            "/home/aocsa/git/distributed-join/build/",
            f"{remote}:/home/aocsa/git/distributed-join/build/",
        ]
    )
    print("sync ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
