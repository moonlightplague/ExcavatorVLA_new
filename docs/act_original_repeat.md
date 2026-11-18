# Original ACT on the completed repeat dataset

Short-horizon follow-up: `training_config_act_shortchunk.yaml` starts a fresh
40k-update original-objective ACT run at
`outputs/experiments/act_shortchunk_repeat40k_20261117_01/`. It changes the
predicted action chunk from 100 to 10 and the saved execution horizon from 100
to 3, retaining the same source publication, parent-disjoint split, seed,
batch, optimizer and single-frame inputs. The completed long-chunk model is
preserved. The preceding replan-three evaluation was stopped at the user's
request; its existing results remain in its separate evaluation directory.
This run isolates short-window training before changing history or sampling.
Actual launch, preflight and training progress are recorded in `status.json`.

Configuration: [`training_config_act_original.yaml`](../training_config_act_original.yaml).
Experiment: `outputs/experiments/act_original_repeat40k_20261117_01/`.
Read `status.json` for actual progress; a launched process does not mean training has completed.

The source is the immutable `data/repeat/export_repeat` publication: 200 complete
episodes and 493,456 frames at 10 Hz. All action labels are expert actions and all
supervision weights equal one. The experiment copies the existing seed-1000
140/20/40 episode split from the SmolVLA history experiment, validates its dataset
fingerprint and fits state/action statistics only on the 140 training episodes.
Images use fixed ImageNet statistics. Phase and recovery annotations are not
policy inputs. Source metadata, tables and videos are hashed before launch and
again after completion.

`objective: original` instantiates the unmodified LeRobot 0.4.4 `ACTPolicy` and
calls its native `forward`: masked L1 averaged over the padded chunk plus
`10 * KL`. It does not call the repository's weighted-loss policy. Checkpoint
loading recognizes the original objective from saved provenance. Existing
training configurations retain their weighted-loss behavior.

The architecture uses one current observation, three 256x256 RGB cameras,
28 state dimensions, four velocity actions, ImageNet ResNet-18 initialization,
a fresh 512D transformer with eight heads, four encoder layers, one effective
decoder layer, a 32D CVAE latent and a 100-action chunk. `n_action_steps: 100`
is the stock inference default; closed-loop evaluation and execution-horizon
experiments are separate from this training request.

Training targets 40,000 updates using BF16, AdamW with learning rate 1e-5,
weight decay 1e-4 and gradient clipping at 10. The batch is eight per GPU on
GPUs 0 and 1, sixteen globally. A single-GPU two-update preflight uses the
same model and per-GPU batch, checks finite gradients, saves and strictly
reloads the stock checkpoint, then removes its temporary training artifacts.
The preflight must remain below 10 GiB of allocated CUDA memory before the
dual-GPU run starts. Existing simulator jobs are left running.

Recovery checkpoints are saved every 5,000 updates. After successful final
validation, reload and source-immutability checks, the controller removes only
this run's intermediate checkpoints and retains the final 40,000-step model,
processors, optimizer and RNG state. The full 20-episode validation split is
evaluated at the end. The 40 test episodes are not used for training or model
selection. Original-mode process groups allow two hours for synchronization so
the full rank-zero validation can finish while the other rank waits.

The controller stores a frozen copy of the project training sources, resolved
configuration, audit, train-only statistics, source checksums, numeric training
log, stage logs and a live `status.json`. Failures are reported explicitly; only
processes started by this controller are stopped on interruption.

```bash
PYTHON=/root/gpufree-data/conda/envs/ExcavatorVLA-current/bin/python
"$PYTHON" -B scripts/run_act_original_experiment.py --prepare-only
# Run this only once for a prepared experiment:
"$PYTHON" -B \
  outputs/experiments/act_original_repeat40k_20261117_01/run_act_original_experiment.py \
  --run-prepared outputs/experiments/act_original_repeat40k_20261117_01
```

The prepared experiment refuses to overwrite an existing run. Inspect its
status and logs before attempting any recovery. The trainer's current CLI
does not implement optimizer-state resume for this experiment.

## Closed-loop evaluation

The final 40k checkpoint is evaluated on the same frozen 40 test parents:

```bash
/root/gpufree-data/conda/envs/ExcavatorVLA-current/bin/python -B scripts/eval_smolvla_repeat40.py \
  --policy-type act --experiment outputs/experiments/act_original_repeat40k_20261117_01 \
  --output outputs/experiments/act_original_40k_closedloop_test40_20261117_01 --gpus 0,1
```

ACT uses its saved default of predict-100/execute-100 at 10 Hz, with a fresh
current observation for each chunk and immediate replanning after goal updates.
The shared physical-cycle and mass accounting is unchanged: ten cycles per scene,
5,000 actions maximum, fresh Newton coarse soil, model-only actions and a video
per scene. This execution horizon differs from SmolVLA's execute-three protocol.
Read the evaluation directory's `summary.json`, `REPORT.md`, worker progress and
`completion.json` for actual progress and completion. Paused SmolVLA processes
are preserved. Completed model failures remain in the denominator.
