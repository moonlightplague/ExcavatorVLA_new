# ACT training with executable repeated-task control labels

This experiment isolates the command-target label repair identified in
`act_return_diagnosis_20261118.md`. It preserves the ACT architecture, current
single-frame observation, native masked L1 plus 10 times KL loss, predict-ten /
execute-three horizons, seed 1000, global batch 16, and 40,000-update budget.
It starts a fresh ACT policy with ImageNet ResNet-18 initialization, rather
than resuming a checkpoint with different action/state normalization. History
and sampling changes are deferred to separate experiments.

The experimental `excavator_bucket_slew20_v1` contract changes bucket command
target slew from 6 to 20 rad/s; this is not a change to measured mechanical
joint speed or soil physics. Other joint command limits remain 6/2.5/2.5.
Existing checkpoints retain their original contracts. ACT checkpoint metadata
now records this contract, and the shared evaluators reject profile mismatches.

## Candidate data and source preservation

Candidate:
`outputs/experiments/act_repeat_control_repair_data_20261118_01/dataset/`.
Original immutable publication: `data/repeat/export_repeat`.

All 200 parents, 493,456 frames, camera timestamps, task text, measured joints,
velocities, tracking errors, geometry, fill observations and the 140/20/40
parent assignments are preserved. Actions are adjacent command-target
differences at 10 Hz, where command target is measured position plus tracking
error. Causal previous-action state fields are updated; labels are not clipped.
All other state fields preserve their source values and precision.

New tables and metadata are private copies. The 600 videos are read-only
hard-linked containers; training never writes them. Full inventories verify
original source bytes before and after preparation. State/action normalization
uses only the original 140 training parents. Image normalization remains fixed
ImageNet constants. The candidate audit also verifies all 600 camera/episode
timestamp pairs and checks per-parent Newton provenance. The new dataset
fingerprint is `3383b61dc7bce419e7b7e9b89fb0f7de591def9d6d7f5c39ee1b19ab347d468c`.

## Replay and launch gates

The pipeline waits for complete reconstructed expert replays of parents 0, 22
and 1 (two training parents, one validation parent; no test parents). Each must
execute every recorded action in all ten cycles, without clipping, and pass:

- Maximum command-target error <=0.001 rad.
- Maximum measured-joint error <=0.15 rad.
- Net delivered mass >=50 kg in every recorded cycle, using verified live mass.
- Exact agreement between replay controls and the actual candidate table.
- Preserved measured source observations and source checksums.

These are expert-replay/data-compatibility checks, not learned model success.
The replay holds the initial recorded task context while executing the complete
recorded expert actions; it validates control/physical trajectories, not live
model goal transitions. The later model evaluation updates goals and verifies
physical returns using the shared repeated-task evaluator.

The controller recomputes these checks from full traces, status, recorded
reference arrays and the candidate labels, then copies checksum-bound evidence
into the candidate. The ACT trainer rejects this experimental profile until
the three-replay gate is present and matches the candidate fingerprint.

After the gate passes, the existing frozen-source controller prepares a new
training run, performs two actual optimizer updates, checks finite loss and
gradients, strictly reloads the checkpoint, checks a 10 GiB CUDA memory budget,
and launches the 40k two-GPU run. Final validation covers all 20 validation
parents. Source inventories and checkpoint reload are checked again at the end.
Recovery saves occur every 5k updates; successful final completion retains the
40k checkpoint according to the existing controller's retention policy.

## Monitoring

- Data preparation: `outputs/experiments/act_repeat_control_repair_data_20261118_01/status.json`.
- Gate and launch state: `outputs/experiments/act_repeat_control_repair_data_20261118_01/pipeline_status.json`.
- Training state after preparation: `outputs/experiments/act_repaired_repeat40k_20261118_01/status.json`.
- Configuration: `training_config_act_repaired.yaml`.

The current pipeline owns these output names. Do not run the controller again
against existing artifacts. A pending replay is not a launched training job;
read `training_started` and the training status for actual progress.

## Verification

Three experimental-gate tests, ten existing action-relabel tests, two original
ACT training/reload tests and ten repeated-evaluation tests passed. Tests use
temporary directories and do not change existing datasets or checkpoints.
