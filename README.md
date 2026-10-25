# ExcavatorVLA_new

See [the training/deployment diagnosis](docs/training_diagnosis.md) for the
verified causes of the failed SmolVLA evaluations, fixes, checkpoints and videos.
The [2026-10-20 evaluation regression fix](docs/smolvla_evaluation_regression_20261020.md)
documents the retained-payload stall, corrected sand/target initialization,
and fresh simulator lifecycle.

ACT training, fixed episode splits and benchmark evaluation are implemented in
the [ExcavatorVLA-vlm submodule](ExcavatorVLA-vlm/README.md).

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

For repeated digging, unloading across low areas of the truck bed, and recorded
physical arm returns, run `./collect_repeat.sh 50 2 0,1 3`. Each episode contains
three complete cycles in one scene. Output defaults to `data/repeat/{raw,export}`.
See [baseline/README.md](baseline/README.md) for options, validation, and SmolVLA training.

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

To collect 50 successful episodes total with two simulators, one per GPU:

```bash
bash collect.sh 50 2 0,1
```
The optional arguments are worker count and comma-separated GPU IDs. Without
GPU IDs, workers use GPUs `0` through `workers - 1`. Each worker has a unique run
directory; the parent exports successful episodes sequentially into `data/export`.
Incomplete targets are retried while collection makes progress. Worker failures
stop the command after exporting available successful results. Logs and result
files are saved under `data/raw/.collection_logs/collection_*/round_*/`.
Ctrl+C stops the worker process groups. `POLICY_PYTHON` selects the export Python
(default: the active environment), and `DATA_ROOT` overrides the data directory.

The terminal reports each running worker's latest log line, elapsed time, and
time since its log last changed every 15 seconds (`COLLECT_STATUS_SECONDS` overrides
this interval). Initial scene/RTX warmup can take several minutes while CPU shader
compilation runs and GPU utilization stays low. Keep `isaacsim/kit/cache` between
runs so compiled shaders can be reused. Each worker also has a separate
`worker_*.kit.log` for native Isaac diagnostics. To follow all output from a running
batch, use `tail -F data/raw/.collection_logs/collection_*/round_*/worker_*.log`.

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

Edit `training_config.yaml` to configure training. It contains dataset and
checkpoint paths, steps, per-GPU batch size, learning rate and schedule,
tokenizer length, trainable layers, checkpoint/log frequencies, and GPU/precision
settings. Paths are resolved relative to the YAML file.

The default is a full 20,000-step run with batch size 8 per GPU on GPUs 0 and 1. Review the
resolved settings before starting:

```bash
cd "$BASE"
bash train.sh --dry-run
bash train.sh
```

Use `bash train.sh --config /path/to/config.yaml` for another configuration.
`smoke_train.sh` is a compatibility alias for the same YAML-driven launcher;
it no longer hardcodes a 20-step smoke run. For a short check, copy the YAML and
set `steps: 20`, `batch_size: 1`, `num_workers: 0`, `save_freq: 20`,
`log_freq: 1`, `policy.scheduler_warmup_steps: 1`, and
`policy.scheduler_decay_steps: 20`.

Set `launcher.gpu_ids: [0, 1]` to use both GPUs through Accelerate. Batch size
is per GPU; `launcher.mixed_precision` selects `bf16`, `fp16`, or `"no"`.
The launcher derives exactly the three dataset cameras and 28D state and applies
the configured tokenizer length to the saved processor. Its flow-matching loss
uses the four real action dimensions and divides by valid supervised timesteps;
28 artificial action channels and episode padding contribute no loss. Recovery
weights are queried at each future action timestamp. The pretrained 32D heads
remain compatible. The default predicts 10 actions (1 second at 10 Hz) and
executes 3 before replanning.
It leaves the installed package files unchanged. Isaac success evaluation is
still run separately; `eval_freq` does not configure that simulator.

Each run saves `training_config.yaml`, `resolved_training_config.json`, and
`train.log`. Checkpoint:
`outputs/train/excavator_smolvla_*/checkpoints/last/pretrained_model`.

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

The final protocol is **`excavator_success_v2`**, specified in
[the evaluation standard](docs/smolvla_evaluation_standard.md). It runs headlessly
with one policy worker per visible GPU, capped by episode count. Policies remain
loaded, while each episode starts a fresh Isaac Sim process and scene. Every
physics tick renders. Historical v1 scores use a different counting/completion
rule and must be reported separately.

```bash
cd "$BASE"
# Checkpoint, episodes, maximum policy steps, minimum deposited particles
bash eval_smolvla_success.sh outputs/train/<run>/checkpoints/last/pretrained_model 10 350 100
# Reference benchmark: seeds 2–501, 350 actions, 100 source particles
bash eval_smolvla_success.sh outputs/train/<run>/checkpoints/last/pretrained_model 500 350 100 --gpus 0,1 --no-video
```

Uses the existing `ExcavatorVLA-smolvla/run_simulation.py` lockstep simulator,
28D observations, checkpoint normalization, and dataset FPS. Replans every
3 steps without a handwritten action supervisor; scene seeds start at 2.
The simulator opens the collection USD in `ExcavatorVLA-isaac`, preserving its
authored robot base pose, lighting and rectangular retaining walls. Evaluation
uses dataset episode 0's initial joint pose by default (`--initial-episode`
selects another), and integrates command velocities into a persistent position
target across requests, matching the training action semantics. Random scenes
reject truck footprints overlapping the excavator body; center distance alone
can admit a long truck that physically locks the swing joint.

Success requires threshold payload AND an open bucket for three consecutive
action observations; increasing payload is allowed. This completes and ends the
task. The default threshold is 100 source particles. Counting uses the current
truck's oriented bed bounds, excludes the bucket, and excludes identities already
deposited initially. Reset diagnostics rebuild the bounds after moving the truck.
Runtime errors are reported separately and stop the batch, including the other
workers. Episode indices and seeds are independent of worker count: episode N
always uses `--seed + N - 1`.

Physics advances at 60 Hz and actions run at dataset FPS; all physics ticks
render. All three policy cameras render at the
collection resolution of 1024x1024 before downsampling to 256x256. Videos remain
enabled by default. `--no-video` disables overview transport and MP4 encoding;
the overview viewport remains active so recording does not change render topology.
CPU thread budgets account for the container's CPU quota. Each GPU gets its own
persistent writable shader cache, seeded from the installation's existing cache
when available, to avoid simultaneous shader-cache writers. Override its root
with `EXCAVATOR_EVAL_CACHE` if needed.

Use `--workers 1 --gpus 0` to restrict resources or `CUDA_VISIBLE_DEVICES` to limit
the allowed GPUs. `--no-headless` shows one simulator and selects one worker;
its GPU must support presenting windows on the current display. Multiple workers
use headless rendering. `--no-fresh-simulator`,
`--no-render-every-physics-step`, reference scenes and GUI mode are explicitly
diagnostic profiles and cannot produce an official score.

Use `official_success_rate` only when `official_result_valid` is true. It remains
null for incomplete/error batches or diagnostic profiles; mixed evaluation
identities are rejected. `provenance.json` hashes checkpoint weights, source code,
root scene USD, metadata, protocol and software versions. Initial observations
and traces include camera hashes and current truck counting geometry.

Results: `outputs/eval/success_*/success_rate_summary.json`, `episode_results.csv`,
and per-episode logs/action traces. Each new evaluation also records these metrics
in `episode_status.json` and CSV, with numeric distributions and failure-mode
counts in the summary JSON:

| Metric | Definition |
| --- | --- |
| Payload | `payload_particles`: source particles in the truck, excluding particles already deposited at the initial observation. |
| Spillage | `spillage_particles` and `spillage_fraction`: unique source particles observed in the bucket that are now outside both bucket and truck, divided by `scooped_particles`. Material still retained is reported separately. The fraction is null if nothing was scooped. |
| Cycle time | `cycle_time_seconds`: simulated action time until success and bucket open; null for an incomplete cycle. `cycle_time_wall_seconds` reports wall time. Failed rollouts still report `simulated_rollout_time_seconds`. |
| Failure mode | v2 distinguishes `no_scoop`, `retained_payload`, `lost_payload`, `insufficient_payload`, `completion_not_sustained`, `bucket_not_open`, and infrastructure/validation errors. |
| Latency | `policy_latency_ms_mean/p50/p95/max` measures each actual replan, including image preparation, preprocessing, model execution, postprocessing and CPU action transfer. Bridge round-trip and complete control-step mean/p95 are also recorded. Cached actions and expert replay have no model latency samples. |
| FPS | `fps`: executed policy actions per rollout wall second, including trace/video writing. Startup/reset, the initial zero-action observation and the final video hold are excluded. This is measured throughput; dataset FPS remains the simulation cadence. |

Particle metrics use initial-pile particle identities sampled once per returned
observation, independently of the bucket-load fallback used by the policy.
Spillage includes particles returned to the pile or still in flight at termination;
recovered particles cease to count as spilled. Transfers entirely between
observations are not observed. Payload and spillage use particle counts. Cycle time
excludes the initial zero-action observation and sums actual simulator step
durations, so it also supports policy rates that do not divide 60 Hz evenly.
Failure classification uses the final payload and task state: `lost_payload`
requires spillage with no delivered or retained material; `retained_payload`
identifies material still carried with nothing delivered. Threshold payload
without the complete observation window is `completion_not_sustained`; a closed
bucket is `bucket_not_open` and cannot count as a v2 success.
Historical status files remain readable; measurements
they did not record stay null and their failure mode may be `unclassified`.
Summary numeric distributions cover completed episodes; failure-mode counts also
include infrastructure/validation errors. Latency summaries aggregate episode
statistics, rather than pooling step samples across episodes.

`runtime.json` records batch wall time,
GPU assignments and worker exit codes. Worker and Kit logs are under
`workers/worker_XX/`; each episode's `simulator.log` links to its worker log.
Episode status files include simulator process IDs, reuse status, and startup
or reset time. The legacy `*_from_simulation_launch_seconds` fields include
the current episode's startup or reset; `timing_origin` identifies this origin.
With videos enabled, every evaluated episode also saves
`episodes/episode_XXXX/evaluation.mp4`: a fixed overview plus the three policy
cameras, with step count, deposited particles, bucket state, and result text.
Videos play at dataset FPS (simulation time) and hold the final result for two
seconds. Failed evaluations keep a partial video or an error screen if startup
failed before images arrived. Video paths are also stored in `episode_status.json`.
For diagnosis, reproduce a raw successful episode's exact truck, sand, joint
pose, task and planning targets. Additional evaluator options follow the first
four launcher arguments:

```bash
export CHECKPOINT=outputs/train/<run>/checkpoints/last/pretrained_model
export REFERENCE=data/raw/<run>/episode_<id>
bash eval_smolvla_success.sh "$CHECKPOINT" 1 350 100 --reference-episode "$REFERENCE"
# Test exported expert actions independently of the policy:
bash eval_smolvla_success.sh "$CHECKPOINT" 1 350 100 --reference-episode "$REFERENCE" --replay-reference
# Audit causal alignment, cadence, limits and normalization:
"$POLICY_PYTHON" scripts/audit_smolvla_data.py --dataset data/export --chunk-size 10 --output outputs/analysis/data_audit.json
```

Reference evaluation fails when the initial target-relative geometry differs
from collection by more than 5 cm. `initial_observation.json` and `steps.jsonl`
record state and commanded positions for debugging. Expert replay is explicitly
marked `execution_source: expert_replay`; its success is not model success.
Particle regeneration and contact physics can still differ between launches.
