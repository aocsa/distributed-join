# Paste this into the Cursor agent on pdx02-zeno-02
# (chat distributed-join-zeno-02 / 7847f902-e9bd-4191-8d56-e42dcd53df3a).
#
# Do NOT run mpirun, build_and_test, or any GPU job. zeno-01 is the only launcher.

Stay idle for a two-node MPI join launched from pdx02-zeno-01. Do not run mpirun
or GPU tests. Keep the GPU free.

1. Allow passwordless SSH from zeno-01 by appending this exact line to
   ~/.ssh/authorized_keys (mkdir ~/.ssh, chmod 700 ~/.ssh, chmod 600 the file):

ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKOEgNqqApvO17mwR0y7g7eKiCPhqUTpjImiqJUrslUj aocsa@nvidia.com

   There is a copy of that line plus an apply script at
   /mnt/weka/tmp/distributed-join-zeno01/enable_ssh_from_zeno01.sh
   if Weka is mounted. Running that script is enough.

2. Reply with:
   - hostname
   - git -C /home/aocsa/git/distributed-join rev-parse --abbrev-ref HEAD
   - git -C /home/aocsa/git/distributed-join rev-parse --short HEAD
   - ls -l /home/aocsa/git/distributed-join/build/bin/test/test_shuffle_on \
           /home/aocsa/git/distributed-join/build/bin/benchmark/distributed_join
   - nvidia-smi --query-gpu=name,utilization.gpu --format=csv
   - IPv4 of enp1s0f0np0 (expect 10.87.131.68)

3. Then wait. The join is started from zeno-01 only.
