# SmolVLA causal visual history

The 20k closed-loop test evaluation is prepared/resumed with:

```bash
.venv-smolvla-history/bin/python scripts/eval_smolvla_repeat40.py \
  --experiment outputs/experiments/smolvla_history_repeat20k_20261116_01 \
  --output outputs/experiments/smolvla_history_20k_closedloop_test40_20261117_04 \
  --gpus 0,1
```

`manifest.json` binds all 40 sealed parent IDs to validated original export
receipts and the frozen training split. Scene references contain initial
observations, scene geometry and task instructions, with **no expert actions**.
The optional `--limit 1` verifies the first scene; resuming the full command
keeps its completed policy outcome and runs the remaining scenes.

Action cadence comes from the source publication's `meta/info.json` (`fps: 10`),
which the evaluator sends to Isaac as `training_fps`. There is no independent
`planner_frequency` setting. The training YAML saves `n_action_steps: 3`, while
evaluation uses its explicit `--replan-interval 3`: execute three control steps,
then predict another ten-action chunk. This is 0.3 seconds between replans in
simulation time. The loop is synchronous, so this is not a guarantee of 10 Hz
wall-clock execution. Image history updates on every control step.

`--steps 5000` currently sets an upper bound: finishing all ten physical
attempts and returns can end a rollout earlier. A mandatory 5,000-step mode
has not been implemented; the paused evaluation must not be resumed expecting
that behavior without changing the rollout protocol.

Each scene uses the collection's coarse Newton MPM solver, 120 Hz soil,
12,000 point budget, rock display and recorded ten-cycle dig/unload coordinates.
Fresh soil is settled at episode startup. Every motion, including the return
after the tenth unload, comes from the model. History is updated every 10 Hz
control observation, uses `[-6,-3,0]`, persists across cycles and executes three
actions from each predicted chunk. Goal changes consume zero physics ticks
and trigger replanning without clearing history. Goals advance after each
physical dig/carry/open/return attempt, including attempts whose deposits fail
the mass gate. Expert timings and phase labels never determine transitions.

The source mass gate requires a new source deposit of at least 50 kg in each
cycle, at least 400 kg in the truck at completion and retention of at least 80%
of cumulative delivered mass. Live material ID counts and metric-point mass
must match each original source scene. Physical attempts require a dig visit,
a curled-bucket carry toward the truck, opening and a physical return. Return
means the tip is inside the collection's recorded Newton source search radius
(`source_search_initial.radius_m`, 3 m in this publication), 0.15–2.5 m above ground
for three consecutive observations. Remaining bucket contents are assessed
separately: a qualified cycle must also have no more than five source IDs left.
`repeat_cycles_completed` counts physical attempts and
`repeat_cycles_successful` counts qualified deposits/returns. The fixed episode
budget is 5,000 actions at 10 Hz, with no per-cycle early timeout; evaluation
can stop earlier after all ten physical attempts and returns. Incomplete tasks
fail, and unattempted later cycles remain in the ten-cycle denominator.

The earlier `_20261117_01` evaluation is invalidated. Its counter conflated
successful deposits with physical attempts and used an overly narrow return
point, merging distinct rounds and causing premature timeouts. Its original
videos and numeric traces remain for diagnosis. Replaying episode 15 now
detects three physical cycles and two qualified deposits (239.657 and 287.589
kg, cumulative 527.246 kg). Recorded observations form a regression fixture.
All 40 scenes are rerun under v3; corrected goal progression changes future
model inputs, so rescoring old trajectories cannot replace the new policy run.

The `_20261117_02` evaluation is also invalidated for source-identity mismatch.
Collection already uses all Newton material IDs except soil initially in the
bucket/truck (`baseline.tuning.install_source_identity`). The deployment
evaluator still used a spatial ellipse after settling: episode 11 retained
only 1,591 of 10,573 IDs, reporting a 47-point bucket as one eligible source
point. V4 matches collection provenance, includes avalanched material and
validates the live initial exclusions. The original source count is recorded
separately because freshly settled episodes can start with different initial
bucket/truck occupants. Traces now retain bucket/truck positions, and videos
include paired before/after unload PNGs, to distinguish missed accounting,
spillage and visual overlap without changing physics, rendering or weights.
The `_20261117_03` run is diagnostic only: it exposed that the authored 2.4 m
footprint also excluded returns to the actual avalanched pile. `_04` matches the
collection's recorded 3 m source search region for all 40 scenes. Material
diagnostics from `_03` remain valid: episode 11 transported the same 54 IDs
into the truck, but median nearest-neighbor distance fell from 7.8 cm to 1.5 mm.
Visible marker overlap is a separate simulation/contact/volume/rendering
question; its exact mechanism has not yet been isolated. Evaluation results
remain within the collection's existing simulator, with no physics changes.

`summary.json` and `REPORT.md` update during evaluation. A full-task rate uses
all ten cycles and all returns; per-cycle progress is reported separately.
Infrastructure errors are preserved separately, never counted as completed
policy failures. A completed model failure is never retried. Every valid scene
retains an MP4, action trace, zero-time goal transition log, source hashes and an
independent mass/identity/history/geometry audit. This evaluates held-out
recorded scene/task coordinates with fresh soil; it does not measure novel
scenes or the collection planner's live choice of subsequent goals.

The experiment status is authoritative at
[`outputs/experiments/smolvla_history_repeat20k_20261116_01/status.json`](../outputs/experiments/smolvla_history_repeat20k_20261116_01/status.json).
Starting training is not confirmation that step 20,000 has completed. The
controller writes `completion.json` and replaces the running `REPORT.md` after
training and final offline diagnostics finish.

## Existing PI05 implementation

The implementation inspected before this adaptation is:

- `ExcavatorVLA-vlm/excavator_vlm/policies/pi05/history.py`: versioned offsets,
  camera/state padding validation, time-major image expansion and the online buffer.
- `.../pi05/dataset.py`: matching negative observation timestamps for every
  camera and state, with future actions and recovery weights at offsets `0..H-1`.
- `.../pi05/adapter.py`: sends all historical images into the existing VLM prefix.
- `.../pi05/history_processor.py` and `processors.py`: quantile-normalized state
  history in the PI05 language prompt, with masked missing image blocks.
- `.../pi05/config.py`, `checkpoints.py`, `inference.py`: bind, persist and check
  the history contract during configuration, save, load and resume.
- `ExcavatorVLA-vlm/configs/pi05/history_c.yaml` and `history_c_repeat20k.yaml`:
  three observations at offsets `[-6,-3,0]`.
- `scripts/eval_smolvla_success.py`: append every control observation, including
  cached-action steps, and query the buffer when generating a chunk.

Offsets count **control observations**, not chunks. At 10 Hz they are
`[-0.6,-0.3,0]` seconds relative to the current decision. They can be used at any
decision step, including one triggered by a task-goal change.
With regular execute-3 replanning these timestamps also coincide with earlier
chunk starts; the implementation still selects control-step offsets rather
than looking up chunk boundaries.

For a decision at episode frame 20 (2.0 seconds), the exact inputs and targets are:

```text
frame 14 (1.4 s): camera 0, camera 1, camera 2
frame 17 (1.7 s): camera 0, camera 1, camera 2
frame 20 (2.0 s): camera 0, camera 1, camera 2
target: actions at frames 20..29 (2.0..2.9 s)
```

At decision frame 0, all three image slots contain frame 0 and the mask is
`[true,true,false]`. At frame 3, selected frames are `[0,0,3]`, with mask
`[true,false,false]`. The unavailable oldest slot is ignored by attention;
the real observation at frame 0 remains valid. Clamping and masks are local
to the episode. History continues across scoops and resets only at a new episode.

PI05 introduces no new learned temporal embeddings: ordered image blocks,
transformer positions, and its explicit state-history prompt identify the
context. Action targets are never shifted backward with observation history.

## SmolVLA adaptation

`scripts/smolvla_history.py` reuses PI05's actual buffer, image expansion and
padding checks. Native SmolVLA's `prepare_images` normally selects only the last
frame of a five-dimensional camera tensor. The local subclass instead calls
native preprocessing once per timestamp and passes all nine images to the
existing visual encoder and VLM prefix, in the fixed order above.

Camera identity remains the canonical key/order `0,1,2`; timestamps are ordered
oldest to newest. No camera keys are invented, no images are spatially stitched,
and no vision/expert projections are added or reinitialized. Full base weights
load strictly. The real model produces a 641-token prefix (576 visual tokens,
64 padded language tokens and one state token), below its 8192-position budget.
The current recorded instruction is 58 tokens, so the existing 64-token setting
retains it completely.

SmolVLA keeps its **current** robot state vector through its existing state
projection. This differs deliberately from PI05's quantized state-history
language prompt: the visual sampling is identical, while SmolVLA's native
current-state representation and mean/std normalization remain intact.

| Boundary | Tensor / ordering |
| --- | --- |
| Dataset, each camera key | `[B,3,3,256,256]`: batch, timestamp, RGB, height, width |
| Dataset state | `[B,3,28]`; native `prepare_state` uses only the current timestamp |
| History mask | Boolean `[B,3]`, common across cameras and state |
| Visual encoder input | Nine native-preprocessed image tensors, time then camera |
| Model state | Current `[B,28]`, internally zero-padded to `[B,32]` |
| Physical action targets/output | `[B,10,4]`, command-target velocities in rad/s |
| Internal pretrained action heads | `[B,10,32]`; existing loss/output logic selects four real joints |

Image scaling, resizing, state/action normalization, inverse action transforms,
four-joint velocity semantics, 10 Hz cadence, predict-10/execute-3 behavior and
the repository's masked, recovery-weighted flow objective are preserved.
The raw model input filter excludes phase/return labels, rewards and other
annotations. Inference additionally removes action targets and loss weights.

## Dataset and frozen split

The source default is `data/repeat`; its current LeRobot publication is
`data/repeat/export_repeat`. Original metadata, tables and videos are immutable.
The new `AGENTS.md` records this default for future training work.

The current publication contains 200 whole episodes and 493,456 frames, with
three 256x256 RGB cameras at 10 Hz, current-v4 28D states and four velocity
actions. Its original split is all 200 episodes in training. It uses the
standard `[6,2.5,2.5,6]` rad/s command-target slew limits, not the earlier PI05
bucket20 experiment. No action relabeling or limit change is performed.

Following the request for a larger test set, the experiment freezes:

| Split | Whole episodes | Frames |
| --- | ---: | ---: |
| Train | 140 | 345,181 |
| Validation | 20 | 49,515 |
| Test | 40 | 98,760 |

The split uses seed 1000 and preserves whole episodes and their cycles.
Statistics are fitted only on the 140 training episodes. All 600 camera/episode
decode checks passed, camera start/end metadata agree across all 200 episodes,
timestamps are contiguous at 10 Hz, and previous-action state matches preceding
action labels exactly. Equal recorded timestamps establish the publication's
synchronization contract, not independent proof of physical camera exposure timing.
SHA-256 inventories of all source metadata, tables and video files are checked
before preparation, before training and after the experiment.

Prepared artifacts are in
`outputs/experiments/smolvla_history_repeat20k_20261116_01/prepared/`:
`dataset_audit.json`, `source_inventory.json`, `split_manifest.json`,
`normalization_stats.json`, and `phase_subsets.json`.

## Training and verification

Configuration: [`training_config_smolvla_history.yaml`](../training_config_smolvla_history.yaml).
The established settings remain batch 16 per GPU (32 globally across GPUs 0/1),
expert/state-projection training with frozen VLM/vision, BF16 forward computation,
LR 1e-4, 1000-update warmup, cosine decay to 2.5e-6, and the original loss.
The requested total is 20,000 updates, saving every 2,000. All ten saves are retained.
The actual model has 450,046,176 parameters, with 99,880,992 trainable parameters.

The real GPU gate measured about 16.44 GiB peak allocated memory for batch 16,
so no batch, learning-rate, image-resolution or horizon adjustment was needed.
Optional microbatch support preserves the logical batch and loss denominator;
it is disabled in this run.

Before launch, 49 focused unit/regression tests passed. Real-model checks on
training observations verified frame/camera identities at 14/17/20, target
actions 20..29, both episode boundaries, finite forward/backward computation,
sampled trainable updates, unchanged sampled frozen weights, and strict checkpoint
save/reload with exactly matching physical action output. Optimizer, scheduler
and step restoration passed. Changing valid historical pixels affected actions;
changing attention-masked padding pixels had zero effect. These are integration
checks, not evidence of better task success. Temporary gate checkpoints were removed.
An additional regression confirms that other policies' observation-count fields
do not activate the SmolVLA-specific history path, bringing the aggregate to
50 passing tests. The actual trained `step_2000` checkpoint also passed strict
CPU reload and nine-image physical-action inference; its normalizer matches
the 140-episode training statistics exactly, and its optimizer/scheduler files
record step 2000. See `checkpoint_2000_reload.json` in the experiment directory.

The dedicated `.venv-smolvla-history` inherits the existing
`ExcavatorVLA-current` model stack and installs only the pinned plotting/analysis
requirements. Torch 2.10.0, Transformers 4.57.6, LeRobot 0.4.4 and NumPy 2.2.6
remain unchanged. Setup is:

```bash
/root/gpufree-data/conda/envs/ExcavatorVLA-current/bin/python \
  -m venv --system-site-packages .venv-smolvla-history
.venv-smolvla-history/bin/python -m pip install -r requirements-smolvla-history.txt
```

The initial attempt stopped at update 258 when its plotting import was missing.
Its logs and a rendered loss curve are preserved in
`attempts/dependency_failure_step258/`; it had not reached the first 2k save.
The replacement starts from the immutable base with the same seed, split and
hyperparameters. Plotting dependencies are checked before launch, and a transient
live-plot failure now preserves the numerical log and lets training continue.

The already-started experiment owns its paths. To reproduce in a **new** experiment,
prepare a fresh output and update `prepared_data` and `output_dir` in a new config:

```bash
POLICY_PYTHON="$PWD/.venv-smolvla-history/bin/python"
"$POLICY_PYTHON" -B scripts/prepare_smolvla_history.py --output <new-experiment>/prepared
"$POLICY_PYTHON" -B scripts/validate_smolvla_history.py \
  --config <new-config.yaml> --output <new-experiment>/preflight.json --gpu 0
"$POLICY_PYTHON" -B -u scripts/run_smolvla_history_experiment.py --config <new-config.yaml>
```

The controller also requires `unit_tests.json` recording passed regression tests.
The underlying training command for this configuration is:

```bash
.venv-smolvla-history/bin/python -B -u \
  scripts/train_smolvla.py --config training_config_smolvla_history.yaml
```

Checkpoints are `train/checkpoints/step_2000`, `step_4000`, ..., `step_20000`.
Every checkpoint includes full model weights, policy/training configuration,
processors and normalization state, `observation_history.json`, local tokenizer
and backbone configuration assets, and optimizer/scheduler/RNG/step state.
`train/checkpoints/last` points to the latest save.

Resume the existing run from a complete save with:

```bash
"$POLICY_PYTHON" -B scripts/train_smolvla.py \
  --config training_config_smolvla_history.yaml \
  --resume outputs/experiments/smolvla_history_repeat20k_20261116_01/train/checkpoints/step_2000
```

Resume checks history offsets, training episode selection, logical batch,
optimizer schedule and action horizon. Native LeRobot restores its optimizer,
scheduler, saved RNG and step state. Exact replay of prefetched distributed
data order or per-rank distributed RNG continuation is not guaranteed by that
native resume mechanism.

Logs are `logs/training.log`, `train/train.log` and per-update `train/losses.jsonl`.
`train/progress.json` reports actual completed updates. The controller refreshes
`train/loss_curve.png` every 250 updates and after training.

## Supported inference and evaluation

Use `load_history` or the shared evaluator, which recognizes the manifest.
Stock SmolVLA's unadapted image preparation does not implement nine-image history.

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'scripts'))  # Run from the repository root.
from smolvla_history import ObservationHistory, load_history

policy, preprocess, postprocess = load_history(checkpoint / 'pretrained_model', 'cuda')
history = ObservationHistory(policy.history)
# At every control step, including cached-action steps:
history.append(current_three_camera_and_28d_state_batch, step)
# When a new action chunk is needed:
chunk = policy.predict_action_chunk(preprocess(history.batch()))
actions = [postprocess(chunk[:, i]) for i in range(3)]
```

The checkpoint contains every native weight and tokenizer/backbone asset used by
this supported loader. No pretrained weight download or phase input is required.
Recreate/reset the buffer at a new episode, retain it across cycles, and append
observations after executed control steps rather than after chunk boundaries only.
The shared evaluator records history indices/masks in its step trace.

After final training, the controller evaluates the fixed final checkpoint on
the validation and sealed test subsets. Recorded stages and return flags select
excavation, loaded transit, unloading and physical-return subsets; they never enter
the policy input. Each phase samples up to eight decision observations per episode:
640 validation decisions and 1280 test decisions, not every held-out frame.
Metrics are physical first-action/chunk MAE and direction accuracy on joint
commands with absolute targets at least 0.05 rad/s.

If still available, the existing 80k single-timestamp checkpoint is evaluated
on the same test observations. Its training budget and source publication are
unmatched, so this comparison cannot isolate the effect of visual history.
No closed-loop excavation-success improvement is claimed without rollout evidence.
No real excavator hardware commands are part of this pipeline.

`scripts/analyze_smolvla_phase_pairs.py` additionally searches validation-only
unloading/return observations for nearby joint poses and opposite action
directions, preserving its criteria and selection in `phase_pairs/`. Any model
history-masking comparison is a qualitative validation diagnostic, not a
separately trained baseline or a test-set success result.
The strict search (joint-vector distance <=0.25 rad and target-direction cosine
<=-0.5) found no pairs. A documented exploratory search at <=0.5 rad and
cosine <=-0.25 found two pairs; their images are `phase_pairs/pair_1.png` and
`pair_2.png`. They have nearby, rather than identical, postures. The complete
28D state also contains velocities and previous commands, so visual similarity
alone does not establish that a single-timestamp policy is ambiguous.

## Files implementing this change

- `AGENTS.md`, `training_config_smolvla_history.yaml`: defaults and experiment settings.
- `requirements-smolvla-history.txt`: isolated plotting/analysis dependencies.
- `scripts/smolvla_history.py`: causal transport, native model adapter and strict loader.
- `scripts/prepare_smolvla_history.py`: source audit, split, statistics and evaluation subsets.
- `scripts/validate_smolvla_history.py`: actual-data/model memory, gradients and reload gate.
- `scripts/train_smolvla.py`: opt-in dataset/model/processor adapters, progress and resumable saves.
- `scripts/eval_smolvla_success.py`, `scripts/success_eval_runtime.py`: causal rollout history and provenance.
- `scripts/eval_smolvla_history_offline.py`: phase-stratified physical-action diagnostics.
- `scripts/analyze_smolvla_phase_pairs.py`: validation-pose inspection and optional history masking.
- `scripts/run_smolvla_history_experiment.py`: persistent controller, checkpoint checks and loss curves.
- `tests/test_smolvla_history.py`, `tests/test_repeat_eval.py`: history/gradient and shared-rollout regressions.
- `docs/smolvla_history.md`, `CHANGELOG.md`: implementation, verification and experiment notes.

No installed LeRobot/Transformers files, immutable base models, source dataset
files or prior checkpoint directories are modified.
