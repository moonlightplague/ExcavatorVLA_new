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
conda create -n ExcavatorVLA python=3.13
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
export ISAAC_PYTHON="$BASE/isaacsim/python.sh"
export POLICY_PYTHON=$(which python)
export DATA_ROOT=$BASE/data
export RUN=$DATA_ROOT
```

## Data collection
At least 8GB of VRAM is required for an atomic simulation process.

To collect a small dataset (headless mode):
```bash
cd "$ISAAC_REPO"
"$ISAAC_PYTHON" run_vla_train_scene.py \
  --headless --auto-collect --no-bridge \
  --success-count 1 --max-attempts 5 --sand-amount 1.0 \
  --dataset-root "$DATA_ROOT/raw" --disable-export
```
You can change the `--success-count` and `--max-attempts` parameter to acquire for a larger dataset.

## Export
Set RUN to the actual `run_*` directory produced above and run the following command to export data: 
```bash
export RUN=/path/to/excavator-data/raw/run_ACTUAL_TIMESTAMP
cd "$ISAAC_REPO"
"$POLICY_PYTHON" excavator_dataset_tools.py "$RUN" \
  --export-lerobot-v3 --export-limit 1 \
  --export-dir "$DATA_ROOT/export" \
  --export-require-standard --export-require-vla
```
To verify the exported data: 
```bash
cd "$POLICY_REPO"
"$POLICY_PYTHON" scripts/dataset/inspect_lerobot_dataset.py \
  --dataset-root "$DATA_ROOT/export"
```

## Training preperations
For a current-v4 export, conversion needs the corresponding raw run to reconstruct historical features:

```bash
cd "$POLICY_REPO"
"$POLICY_PYTHON" convert_lerobot_stage_dataset.py \
  --src "$DATA_ROOT/export" \
  --dst "$DATA_ROOT/prepared27-stage10" \
  --raw-run "$RUN" --raw-split trainable \
  --runtime "$POLICY_REPO/scripts/excavator_app/excavator_runtime.py" \
  --horizon 30 --video-mode copy

"$POLICY_PYTHON" scripts/dataset/inspect_lerobot_dataset.py \
  --dataset-root "$DATA_ROOT/prepared27-stage10"
```

## Training
The implemented canonical recipe resumes an existing trained checkpoint. It is not a turnkey fresh-training recipe for arbitrary collected data.

In a compatible, unpatched environment, apply the six patchers in this order, using explicit installation paths. Skip this if the original environment already contains the patches; validate its markers instead.
```bash
cd "$POLICY_REPO"
LEROBOT_DIR=$("$POLICY_PYTHON" -c 'import lerobot; print(next(iter(lerobot.__path__)))')
SMOL_DIR="$LEROBOT_DIR/policies/smolvla"
"$POLICY_PYTHON" scripts/training/patch_smolvla_stage_training.py --smolvla-dir "$SMOL_DIR"
"$POLICY_PYTHON" scripts/training/patch_smolvla_stage_seq50.py --smolvla-dir "$SMOL_DIR"
"$POLICY_PYTHON" scripts/training/patch_smolvla_dynamic_stage_seq50.py \
  --smolvla-dir "$SMOL_DIR" --factory "$LEROBOT_DIR/datasets/factory.py"
"$POLICY_PYTHON" scripts/training/patch_stage_batch_and_logging.py \
  --modeling "$SMOL_DIR/modeling_smolvla.py" --train "$LEROBOT_DIR/scripts/lerobot_train.py"
"$POLICY_PYTHON" scripts/training/patch_smolvla_stage_action_direction_loss.py --smolvla-dir "$SMOL_DIR"
"$POLICY_PYTHON" scripts/training/patch_smolvla_stage_action_low_motion_loss.py --smolvla-dir "$SMOL_DIR"
```
