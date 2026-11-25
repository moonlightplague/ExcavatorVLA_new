# ACT-internal temporal encoding experiment

This experiment changes the model and its training inputs only. It does not
add task-phase labels, a return controller, a supervisor, goal-dependent action
overrides, evaluator-side observation history, or new success criteria.

## Model

The native ACT current-image spatial encoder, current-state token, CVAE,
action decoder and ten-action output remain. A learned temporal branch uses
three decision observations: state projections and globally pooled features
from the shared ResNet for the three cameras. Learned time/view embeddings
and a two-layer transformer produce a residual for the native state token.
The final residual projection starts at zero. Exact core-weight transfer and
the zero residual preserve the repaired 40k source model at initialization.

The model has 56,321,924 parameters. Its public inference input remains one
current 28D state and three RGB cameras. Its own bounded feature deque records
each `predict_action_chunk` invocation; `ACTPolicy.reset()` clears it per
episode. History persists across task-goal changes. There is no external phase
switch or action controller. A full explicit training/validation window does
not mutate online history.

With the unchanged execute-three protocol, three regular prediction calls
correspond to control-frame offsets `[-6,-3,0]`, nominally `[-0.6,-0.3,0]`
seconds at 10 Hz. The model sees prediction observations, not observations
from the two intervening actions. An immediate goal-triggered replan can make
the interval shorter; the online contract is the last three prediction calls,
not an asserted wall-clock window. This is a bounded history experiment, not
long-term recurrent task memory.

Missing initial observations repeat the first sample and are attention-masked.
Training requests the same negative offsets for state and all cameras and
rejects inconsistent padding. Future action targets remain at offsets 0..9.
No future observations or stage/return labels enter the model.

## Training

Configuration: `training_config_act_temporal.yaml`.

- Source weights: repaired native ACT 40k, strict transfer of every existing key.
- New updates: 10,000 with a fresh optimizer; predict 10 / execute 3.
- Native ACT masked L1 plus 10 times KL remains unchanged.
- Base-network LR 3e-6; backbone LR 1e-6; temporal branch LR 1e-4.
- Batch eight per GPU on two GPUs, global batch sixteen; BF16 computation.
- Same repaired publication, action contract, 140/20/40 parents and train-only
  normalization as the repaired 40k baseline. No new data labels are created.
- Validation uses independent explicit causal windows over the full 20-parent
  validation split; it never carries an inference cache between random batches.

Run: `outputs/experiments/act_temporal_repaired10k_20261118_01/`.
Read `status.json` for actual initialization, training and completion state.
The frozen training sources and resolved configuration are retained. Recovery
checkpoints follow the existing 5k schedule and final-only retention after
successful final validation/reload/source-inventory checks.

## Private model runtime

The installed LeRobot package is unchanged. A private copy extends only ACT's
configuration and model implementation; its opt-in defaults leave single-frame
ACT behavior intact. The temporal model implementation is bundled inside that
copy, with checksums verified by its Python launcher:

`outputs/experiments/act_temporal_runtime_20261118_02/python`

Use that launcher to load the new checkpoint through the ordinary ACT factory.
The generic `scripts/eval_smolvla_success.py` needs no edits or new history
options. Keep `--replan-interval 3` and use the same dataset, references, physics
and criteria as the repaired baseline. The model's configuration still exposes
the existing current-observation ABI; temporal settings are explicit in
`temporal_horizon`, `temporal_stride` and `temporal_layers`. Loading this new
architecture with the unextended LeRobot runtime is unsupported.

The legacy standalone offline ACT CLI creates single-frame datasets and is not
the validation entrypoint for this model. Full offline validation is supplied
by the temporal training helper, which preserves the required window masks.

The model runtime records the hashes of the evaluator, repeated-task scoring,
evaluation runner and simulator files in `unchanged_external_files.json`.
They remain byte-identical during this model-only experiment. Historical
changes from earlier experiments are preserved.

## Verification

Six temporal-model tests cover exact zero-adapter compatibility, influence of
past images/state and temporal order, initial-padding invariance, online versus
explicit-window equivalence and episode reset, gradients entering the temporal
branch, and strict checkpoint reconstruction. Original ACT tests pass in both
the original and private runtimes; the existing continuation tests also pass.
Real-data checks verify `[0,0,0]`, `[0,0,3]`, `[0,3,6]` windows and a fresh
second episode, synchronized camera padding and unchanged ten-action labels.

The real-data two-update preflight passed finite loss/gradient checks, exact
source-output initialization and strict checkpoint reload. Peak allocated CUDA
memory was 3,011.97 MiB for per-GPU batch eight. These are implementation checks,
not evidence of learned multi-cycle success. Closed-loop evaluation after
training must determine whether the model now starts its physical return.
