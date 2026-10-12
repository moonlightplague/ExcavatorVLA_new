# ExcavatorVLA_new

## Preperations

### Clone the repo
```bash
git clone https://github.com/moonlightplague/ExcavatorVLA_new.git --recursive
cd ExcavatorVLA_new
```
### Conda environment
Make sure you have conda installed and initialized, then run:
```bash
conda create -n ExcavatorVLA python=3.12
conda activate ExcavatorVLA
```
All our later expriments would run in this conda environment.

### Run preperations
The `preperations.sh` script will help you download and extract `isaacsim` and install pip packages expected to use.
```bash
bash preperations.sh
```

### State environment variables
```bash
export BASE=$(pwd)
export ISAAC_REPO="$BASE/ExcavatorVLA-isaac"
export POLICY_REPO="$BASE/ExcavatorVLA-smolvla"
export ISAAC_PYTHON="$BASE/run_isaac_python.sh"
export POLICY_PYTHON=$(which python)
export DATA_ROOT=$BASE/data
export RUN=$DATA_ROOT
```

The repository launcher gives Isaac Sim's bundled `libpython` priority over
system and Conda libraries. Calling `isaacsim/python.sh` directly with
`/usr/lib/x86_64-linux-gnu` at the front of `LD_LIBRARY_PATH` can load the wrong
Python shared library and segfault before the simulator starts (for example,
inside `nest_asyncio.apply()` / `_asyncio._get_running_loop()`). Isaac still uses
its bundled Python; `POLICY_PYTHON` uses the active `ExcavatorVLA` environment.

## Data collection
At least 8GB of VRAM is required for an atomic simulation process.

On a minimal Ubuntu container, install the graphics runtime dependencies first:

```bash
sudo apt-get update
sudo apt-get install -y libegl1 libgl1 libxt6t64 vulkan-tools
env -u DISPLAY vulkaninfo --summary
```

The Vulkan device list must include an NVIDIA GPU. CUDA working in `nvidia-smi`
does not establish that Vulkan rendering works. Missing `libEGL.so.1` can prevent
the NVIDIA Vulkan driver from loading, leaving camera warmup waiting for frames.
Collection prints a heartbeat every 15 seconds; set
`EXCAVATOR_AUTO_COLLECT_HEARTBEAT_SECONDS=0` to disable it.

The collection launcher allocates RTX pools of 360,000 descriptor sets and
900,000 resource descriptors before renderer startup. Override these with
`--rtx-descriptor-sets` and `--rtx-reserved-descriptors` when needed. Headless
camera windows remain visible inside Kit to avoid repeated show/hide changes;
the process still uses `--no-window`. Completed camera readbacks release their
helper references. These changes address `Out of resource descriptors!` during
capture; a process that has already logged this fatal error must be restarted.

To collect a small dataset (headless mode):
```bash
cd "$ISAAC_REPO"
"$ISAAC_PYTHON" run_vla_train_scene.py \
  --headless --auto-collect --no-bridge \
  --success-count 1 --max-attempts 5 --sand-amount 1.0 \
  --dataset-root "$DATA_ROOT/raw" --disable-export
```
You can change the `--success-count` and `--max-attempts` parameter to acquire for a larger dataset.

To collect and export all trainable episodes in one command:
```bash
cd "$BASE"
bash collect.sh 50
```

## Export
Set RUN to the actual `run_*` directory produced above and run the following command to export data: 
```bash
export RUN=/path/to/excavator-data/raw/run_ACTUAL_TIMESTAMP
cd "$ISAAC_REPO"
"$POLICY_PYTHON" excavator_dataset_tools.py "$RUN" \
  --export-lerobot-v3 --export-append \
  --export-dir "$DATA_ROOT/export" \
  --export-require-standard --export-require-vla
```
To verify the exported data: 
```bash
cd "$POLICY_REPO"
"$POLICY_PYTHON" scripts/dataset/inspect_lerobot_dataset.py \
  --dataset-root "$DATA_ROOT/export"
```

## Training (current-v4, 28D)

Fine-tune `models/smolvla_base` on `data/export`: 28D state, three cameras,
4D actions. No 27D conversion needed.

Training and evaluation require unpatched LeRobot 0.4.4. If your existing
`ExcavatorVLA` environment has the legacy stage-supervision patches, create a
fresh Conda environment (do not clone the patched environment):

```bash
conda create -n ExcavatorVLA-current python=3.12 -y
conda activate ExcavatorVLA-current
python -m pip install -r requirements.txt
export POLICY_PYTHON="$(command -v python)"
```

The launchers use `POLICY_PYTHON` when set, otherwise the active environment's
`python`. Refresh `POLICY_PYTHON` after switching environments if it was already
exported. No repository-local virtual environment is required.

The smoke training script runs 20 steps:

```bash
cd "$BASE"
bash smoke_train.sh
```

Checkpoint: `outputs/train/smolvla_smoke_*/checkpoints/last/pretrained_model`.

## Evaluate current 28D model

Evaluate the first N episodes in `data/export` (default: 1):

```bash
cd "$BASE"
bash eval_smolvla.sh outputs/train/<run>/checkpoints/last/pretrained_model 1
```

Results: `outputs/eval/smolvla_*/{metrics.json,actions.npz}`.
Reports action MAE/RMSE in rad/s. Training-data evaluation is a fitting check,
not a measure of generalization or simulator success.

## Evaluate current 28D success rate

Requires a working Isaac Sim GUI/display. Each episode starts a fresh simulator.

```bash
cd "$BASE"
# Checkpoint, episodes, maximum policy steps, minimum deposited particles
bash eval_smolvla_success.sh outputs/train/<run>/checkpoints/last/pretrained_model 10 350 100
```

Uses the existing `ExcavatorVLA-smolvla/run_simulation.py` lockstep simulator,
28D observations, checkpoint normalization, and dataset FPS. Replans every
3 steps without a handwritten action supervisor; scene seeds start at 2.

Success follows the old rule: deposited particles meet the threshold and do
not increase for 3 observations. An open bucket ends the successful episode.
The default threshold is the old value of 1 particle; the example uses 100.
Runtime errors are reported separately and stop the batch.

Results: `outputs/eval/success_*/success_rate_summary.json`, `episode_results.csv`,
and per-episode logs/action traces. This adapts the legacy evaluation simulator;
its scene/material settings are not guaranteed identical to the collection scene.
