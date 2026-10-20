# SmolVLA evaluation regression — 2026-10-20

The original batch `outputs/eval/success_20261019_205715_37970` scored 290/500
successes, including 180 retained-payload failures. Episode 0025 is scene seed
26. Its model curls a bucket in the pile and never initiates useful transport.

The corrected evaluator keeps the existing checkpoint, pure model actions,
10 Hz cadence, persistent command-target integration, truck geometry, and v2
completion rule. It changes sand initialization and the digging target supplied
to the policy, and makes fresh simulator processes the session default.

## Historical comparison

`a9f6cfd` changed the SmolVLA submodule from `46f34f9` to `4690878`. Besides
process/runtime work, that transition changed scene loading, lighting/ground,
the robot base placement, and measured-pose versus persistent command targets.
It is not a performance-only comparison.

The 500-episode batch already used `fresh_simulator: true`; episode 0025 records
`simulator_reused: false`. Restarting alone cannot correct that specific batch.
The session manager now restarts by default even if a caller forgets to close
the previous episode. Persistent reset is available only through explicit
`reuse_process=True`, used by `--no-fresh-simulator` diagnostics and the live
reset regression.

Reverting only the command target to the measured pose did not recover seed 26.
Moving only the robot back to the historical deployment position did not recover
it either. Those changes are not included in the production fix.

## Confirmed environment problems and fixes

1. Evaluation called `clean_sand_retaining_walls()` when the optional circular
   wall was disabled, before sand generation and settling. This also removed
   the rectangular generation walls. Collection removes those walls after
   settling. The corrected fresh and persistent paths retain/recreate the
   rectangular walls for settling and remove them before rollout. The initial
   source-particle count for seed 26 changes from about 12,600 to about 18,500,
   matching the approximate 18,200-particle median in training.
2. Evaluation used the 65th percentile of particle heights within a 1 m radius
   directly as the dig target. Collection supplies a below-surface target from
   local cut-path surface probes, subtracting planned depth and extra bite.
   Merely correcting the walls does not correct this semantic mismatch.
   `excavator_common/collection_target.py` now implements the local surface
   recipe using the deepest configured collection cut (0.58 m), extra bite
   (0.12 m), roughness margin, and a floor bound. Missing sand surface data is
   an error instead of a silently fabricated target. Exact reference targets
   remain available for recorded-episode replay.

The randomized target recipe follows collection's surface/depth convention; it
does not run the collection planner's target ranking, IK search, or retries.
Both methods are versioned in the evaluation manifest. Scene context records
the probe heights, planned surface, and depth to make differences inspectable.

## Controlled seed-26 model tests

All tests use the unchanged 040000 checkpoint from
`excavator_smolvla_20261014_190059_474909_268032`, a maximum of 400 actions,
100-particle threshold, replanning every three actions, a fresh simulator,
and no action supervisor.

| Variant | Result | Truck payload | Artifact under `outputs/eval/` |
| --- | --- | ---: | --- |
| Original saved run | Retained payload | 0 | `success_20261019_205715_37970/episodes/episode_0025` |
| Measured-pose command targets only | Retained payload | 0 | `regression_seed26_measured_target` |
| Correct wall settling only | Retained payload | 0 | `regression_seed26_wall_settling` |
| Historical robot placement plus wall correction | No scoop | 0 | `regression_seed26_legacy_base` |
| Wall correction plus below-surface cut target | Success, 242 actions / 24.2 simulated seconds | 305 | `regression_seed26_collection_target` |

These exploratory runs are labeled diagnostic because they use temporary
simulator variants. The successful model lifted, transported and dumped;
three successive observations met the existing open-bucket/payload rule.
This shows an evaluator-side correction can resolve the observed behavior
without training the model or adding recovery actions. It does not establish
the new 500-scene success rate.

## Production verification

The final source is tested separately through the ordinary evaluator in
`outputs/eval/smolvla_regression_fix_seeds24_27_20261020`, with seeds 24–27,
two workers, video enabled, 400 actions maximum, threshold 100, and fresh
simulators. Final outcomes:

| Seed | Result | Payload | Simulated cycle | Isaac PID | Worker |
| --- | --- | ---: | ---: | ---: | ---: |
| 24 | Success | 903 | 17.4 s | 751027 | 0 |
| 25 | Success | 434 | 18.3 s | 750988 | 1 |
| 26 | Success | 209 | 23.9 s | 753606 | 0 |
| 27 | Success | 234 | 20.4 s | 754031 | 1 |

All four episodes record `simulator_reused: false`; each worker's second
episode has a new Isaac PID. All measured bucket angles in these rollouts
remain at or above -120 degrees within numerical precision. Seed 26 now
completes with the same model that previously retained material for 40 seconds
without depositing any.

The independent saved-trace audit with `--verify-sources` passes all four
episodes, and the batch has `official_result_valid: true`. This validity applies
to the four-seed regression sample, not a 500-scene score. Checkpoint file hashes
were also checked against the original 500-episode manifest and match exactly.

Artifacts in that production-run directory:

- `audit.json`: independent completion, timing, accounting and source verification.
- `regression_comparison.json`: original versus corrected results, process IDs,
  minimum bucket angles, initial source counts, and target methods.
- `episodes/episode_0003/evaluation.mp4`: corrected scene seed 26, corresponding
  to episode 0025 of the original batch.

Evaluation identity:
`a4c5b6d962d07d691342090ceb1293c3adb6b83e44f772e32ea0a685a05e78c8`.

The optional persistent diagnostic reset also passes the live seed sequence
2 → 12 → 2 in one Isaac process (PID 756574), verifying fresh particle-transfer
baselines and truck bounds changing/restoring with the scene. Its result is
`outputs/eval/smolvla_regression_fix_reset_20261020/reset_validation.json`.
Standard evaluation continues to use a fresh process per episode.

Automated checks:

- Root test suite: 69 tests, 68 passes and one pre-existing unavailable-fixture skip.
- SmolVLA test suite: 72 tests, all pass.
- Regression coverage checks wall lifetime/order, cleanup on reset failure,
  separate simulator processes/sockets by default, explicit reset opt-in,
  below-surface targets, path direction, floor bounds, invalid particle data,
  and agreement with collection's probe/depth configuration.

The original 500-scene outputs and model weights remain available for comparison.
Scores from before this environment correction need a matched rerun; the small
regression batch is not a replacement benchmark.
