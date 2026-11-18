# Environment setup

## Clone and prepare the checkout

```bash
git clone --recursive https://github.com/moonlightplague/ExcavatorVLA_new.git
cd ExcavatorVLA_new
conda create -n ExcavatorVLA python=3.12
conda activate ExcavatorVLA
bash scripts/setup/prepare.sh
```

The preparation script uses shared installations under `/root/gpufree-data`. It downloads Isaac Sim and the SmolVLA base model when their respective directories are missing, then installs the policy dependencies. If the shared installations already exist, check that the checkout's `isaacsim` and `models` links point to them. Set up `data` on your dataset storage before collecting episodes.

## Set paths for the workflow guides

Run these commands from the repository root in each new shell.

```bash
export BASE="$PWD"
export ISAAC_REPO="$BASE/ExcavatorVLA-isaac"
export POLICY_REPO="$BASE/ExcavatorVLA-smolvla"
export ISAAC_PYTHON="$BASE/run_isaac_python.sh"
export POLICY_PYTHON="$(command -v python)"
export DATA_ROOT="$BASE/data"
```

Use `ISAAC_PYTHON` for simulator commands. The wrapper gives Isaac Sim's bundled `libpython` priority over system and Conda libraries. Starting Isaac with a system library first in `LD_LIBRARY_PATH` can segfault during Python initialization. Policy commands use the active environment through `POLICY_PYTHON`.

See [collection and export](collection.md) for graphics runtime dependencies and [SmolVLA training](training.md) for the policy environment.
