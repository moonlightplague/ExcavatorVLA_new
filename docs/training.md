# Train SmolVLA

Fine-tune `models/smolvla_base` on `data/repeat/export_repeat`: 28D state, three cameras,
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

The default is a full 80,000-step run with batch size 16 per GPU on GPUs 0 and 1. Review the
resolved settings before starting:

```bash
cd "$BASE"
bash train.sh --dry-run
bash train.sh
```

Use `bash train.sh --config /path/to/config.yaml` for another configuration.
For a short check, copy the YAML and
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
`outputs/train/repeat/excavator_smolvla_*/checkpoints/last/pretrained_model`.
