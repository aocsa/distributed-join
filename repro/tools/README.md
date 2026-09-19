# Tools used while porting

These were used on `pdx02-zeno-01` to choose a RAPIDS version and grind compiler
errors. They are not required on a new box if `repro/bootstrap.sh` succeeds.

| Script | Purpose |
|---|---|
| `list_rapids_versions.py [subdir]` | libcudf CUDA 12 builds on the rapidsai channel + rmm/nvcomp pins |
| `summarize_build_errors.sh` | Ninja `-k 0` build; distinct errors ranked by count |

Also used (live under `porting/`):

| Script | Purpose |
|---|---|
| `porting/patch_nvcc_activate.sh` | Make cuda-nvcc conda activate safe under bash `set -u` |
| `porting/build_and_test.sh [ranks]` | Wipe `build/`, compile, run the MPI tests |
