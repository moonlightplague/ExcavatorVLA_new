# Collection and export

For repeated digging, unloading across low areas of the truck bed, and recorded
physical arm returns, run `./collect_repeat.sh 10 2 0,1 10`. Each episode contains
ten complete cycles in one scene, using Newton MPM with the tuned reachable
side-loading workspace, sand amount 3 and explicit episode efficiency acceptance
(see [the policy definition](../baseline/TUNING.md#acceptance-policy)). Successful episodes export immediately to
`data/repeat/export_repeat`; validated episodes retain only metrics in `raw`.
Combined batch totals and averages are saved under `data/repeat/metrics`.
See [baseline/README.md](../baseline/README.md) for options, validation, and SmolVLA training.

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

## Single-cycle collection

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

## Export a single-cycle run
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
