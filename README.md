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

## Patch checking
The implemented canonical recipe resumes an existing trained checkpoint. It is not a turnkey fresh-training recipe for arbitrary collected data.

In a compatible, unpatched environment, apply the six patchers in this order, using explicit installation paths. Skip this if the original environment already contains the patches; validate its markers instead.
```bash
cd "$POLICY_REPO"
LEROBOT_DIR=$("$POLICY_PYTHON" -c 'import lerobot; print(next(iter(lerobot.__path__)))')
SMOL_DIR="$LEROBOT_DIR/policies/smolvla"

"$POLICY_PYTHON" scripts/training/patch_smolvla_stage_training.py \
  --smolvla-dir "$SMOL_DIR"

"$POLICY_PYTHON" scripts/training/patch_smolvla_dynamic_stage_seq50.py \
  --smolvla-dir "$SMOL_DIR" --factory "$LEROBOT_DIR/datasets/factory.py"

"$POLICY_PYTHON" scripts/training/patch_stage_batch_and_logging.py \
  --modeling "$SMOL_DIR/modeling_smolvla.py" \
  --train "$LEROBOT_DIR/scripts/lerobot_train.py"

"$POLICY_PYTHON" scripts/training/patch_smolvla_stage_action_direction_loss.py \
  --smolvla-dir "$SMOL_DIR"

"$POLICY_PYTHON" scripts/training/patch_smolvla_stage_action_low_motion_loss.py \
  --smolvla-dir "$SMOL_DIR"
```
Stop on a patch mismatch and recover a compatible installation; do not force text replacements into a different LeRobot version.

## Training
Before running `resume_latest_checkpoint_seed2_fixed5ep_canonical_300epochs.sh`, edit its hardcoded paths for this machine: `PROJECT`, `PYTHON`, `TRAIN`, `DATASET`, `OUTPUT_ROOT`, `LOG_ROOT`, `STAGE_ACTION_PRIOR`, `SMOLVLA_DIR`, and `HF_HOME`. Also inspect absolute paths inside the source checkpoint configs, including VLM and prior paths: the resume script only updates selected config fields. Merely exporting `PROJECT` or `DATASET` does not override unconditional shell assignments.

Then, with the exact prepared dataset and a complete source run:

```bash
cd "$POLICY_REPO"
SOURCE_OUTPUT=/path/to/complete/source-training-run \
  bash resume_latest_checkpoint_seed2_fixed5ep_canonical_300epochs.sh
```

The source checkpoint must contain `pretrained_model/{model.safetensors,config.json,train_config.json}` and `training_state/{training_step.json,optimizer_param_groups.json,optimizer_state.safetensors,scheduler_state.json}`. Weights alone cannot perform this true resume.

The launcher validates inputs, runs a two-step CUDA smoke test, copies the source checkpoint to a new output, and starts training with `nohup`. At 810 frames and batch 144, it uses six steps per epoch; 300 added epochs mean 1,800 added steps (16650 → 18450 for the documented source). The current script uses **16 workers**, although its README says 8. Follow the generated log and verify the final complete checkpoint; launcher exit alone does not mean training finished.

For a new dataset or training from the base SmolVLA model, a separate training configuration is needed. The repo does not supply a validated general-purpose fresh-training recipe; do not remove the canonical dataset guards and call that reproduction.

## Evaluate offline

With a compatible checkpoint and prepared data, the Python entry point exposes paths directly:

```bash
export DATASET=/path/to/prepared-dataset
export CHECKPOINT=/path/to/checkpoints/018450/pretrained_model
cd "$POLICY_REPO"
"$POLICY_PYTHON" scripts/evaluation/evaluate_smolvla_episodes.py \
  --dataset "$DATASET" --checkpoint "$CHECKPOINT" \
  --vlm "$VLM" --prior "$PRIOR" \
  --num-episodes 5 --seed 2 --device cuda \
  --output-dir "$DATA_ROOT/offline-eval"
```

For the canonical experiment, adapt hardcoded paths in `run_latest_three_checkpoints_seed2_fixed5ep_eval.sh`, then run it with `TRAIN_RUN=/path/to/completed/run`. It produces checkpoint comparison CSV/JSON, episode arrays, metrics, and plots. `run_action_stage_chunk_sweep.sh` follows it and compares offline execution chunks. Its chunk length is a separate setting from live replanning or temporal ensembling. Offline action/stage accuracy does not establish successful physical digging/dumping.

## Run a live policy smoke test

Use **both simulator and client from `smolvla`**. Its current client uses protocol v2. Do not connect it to `isaac`'s `run_vla_train_scene.py --bridge` or the old Script Editor bridge.

Terminal A (GUI/viewport-capable Isaac session required):

```bash
cd "$POLICY_REPO"
export EXCAVATOR_PROJECT_ROOT="$POLICY_REPO"
"$ISAAC_PYTHON" run_simulation.py --scene-seed 2 --sand-amount 1
```

Terminal B (same path variables as above):

```bash
cd "$POLICY_REPO"
mkdir -p "$DATA_ROOT/live-smoke"
"$POLICY_PYTHON" scripts/bridge_test/smolvla_policy_client.py \
  --host 127.0.0.1 --port 5555 \
  --ckpt "$CHECKPOINT" --vlm "$VLM" \
  --dataset-meta "$DATASET/meta/info.json" --steps 10 \
  --trace-log "$DATA_ROOT/live-smoke/steps.jsonl" \
  --chunk-log "$DATA_ROOT/live-smoke/chunks.jsonl"
```

This is a short connectivity/inference check, not exact checkpoint reproduction. Use the actual training FPS from metadata. Check that all three camera images are fresh/nonblack, state/effort features match the checkpoint, and commanded joints move in the expected directions. Stop the simulator with Ctrl-C after the client exits.

For the recorded checkpoint-18450 experiment, adapt the paths in **`run_isaacsim_ckpt18450_seed2_one_episode.sh`** and run that launcher. It includes recorded pose/target settings, videos, validation, logs, and result packaging. At the inspected revision it uses 500 policy steps, replanning interval 3, ensemble width 1, and an **enabled hand-written excavation sequence supervisor**. Its results therefore include supervisor intervention. The bare client defaults to model-only execution. Record which mode you used when comparing outcomes.

## What must be supplied before the full run

The immediate next step is to obtain the original Isaac/LeRobot environment identity, local VLM, stage prior, and a complete checkpoint together with its exact prepared dataset. Alternatively, install a compatible Isaac environment and start at the bounded collection test, retaining raw runs. The handover ZIP alone does not establish that the later canonical continuation experiment can be reproduced.
