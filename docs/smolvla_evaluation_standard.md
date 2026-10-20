# Excavation evaluation standard v2

The standard identifier is `excavator_success_v2`. This is the reference protocol
for the current 28D SmolVLA policy. It intentionally changes the old deposit
counter and success rule; historical v1 scores are not directly comparable.

## Reference benchmark

Use the same checkpoint files, simulator assets/software, initial pose, and seed
set for comparisons. The reference benchmark is 500 scenes, seeds 2–501, 350
policy actions per scene, and at least 100 deposited source particles. A smaller
100-scene comparison uses seeds 2–101 and must be reported as such.

```bash
conda activate ExcavatorVLA-current
export POLICY_PYTHON="$(command -v python)"
bash eval_smolvla_success.sh \
  outputs/train/<run>/checkpoints/040000/pretrained_model \
  500 350 100 --gpus 0,1 --no-video
```

The launcher defaults to one episode for a quick check, with the same 350-action
limit and 100-particle threshold. The standard protocol permits explicitly
configured thresholds/horizons; their values and seed set are part of the saved
evaluation identity. Compare rates only with matched settings.
Fresh processes and fixed rendering increase runtime; the reference benchmark
can take many hours. `runtime.json` records total elapsed time, including startup.

## Simulation and observation protocol

- Each episode starts a new Isaac Sim process and loads the authored collection
  USD. Policies stay loaded per GPU worker. No previous episode's sand, solver,
  collision, target, or renderer history is reused in a standard run.
  Randomized scenes use the fixed `--sand-amount 1` setting; the sampler's nominal
  amount in `scene_profile` does not override that explicit choice.
- Temporary rectangular generation walls stay in place during sand settling
  and are removed before rollout, as in collection. Disabling the optional
  circular wall does not remove the generation walls early. The simulator
  session itself also defaults to a fresh process; persistent reuse requires
  explicit opt-in through the diagnostic `--no-fresh-simulator` path.
- Randomized digging targets use `collection_local_cut_surface_depth_v1`:
  query the local particle surface with collection's small 90th-percentile
  probes, take the low surface along the inward cut path, and subtract the
  deepest configured cut (0.58 m), extra bite (0.12 m), and surface uncertainty.
  The target remains at least 0.04 m above the sand floor. A percentile of
  the pile's particle heights is not used as the digging target. Reference
  episode replay retains the exact recorded target.
  `config.json` and the provenance protocol record both the target method and
  sand initialization method. Each scene records the surface probes and
  resulting depth in `scene_context.json`.
- Physics runs at 60 Hz; commands advance exactly the dataset action interval
  (10 Hz for the current dataset). Command velocities integrate into a persistent
  position target. There is no handwritten action supervisor or task recovery.
  Targets are clamped to the articulation's canonical joint limits every physics
  tick. Both legacy and current Isaac limit APIs are supported; failure to read
  limits is an evaluation error. The scene context records the resolved limits.
- Every physics tick renders. A paused observation capture advances at least five
  render updates, waiting longer only for pending GPU readbacks. The timeline is
  paused and committed before capture; rendering does not advance policy time.
  Headless mode performs no wall-clock idle refreshes while waiting for policy
  inference or video encoding. GUI diagnostics retain periodic UI updates.
- The three policy cameras render at 1024×1024 and downsample to 256×256. The
  overview viewport is captured whether or not MP4 recording is enabled. Thus
  `--no-video` controls overview transport/encoding rather than viewport topology.
- Seed `seed + episode_index - 1` controls scene sampling and policy RNGs,
  independently of worker assignment. Python, NumPy and Torch RNGs are seeded per
  episode. Torch deterministic algorithms are required, TF32 and cuDNN benchmark
  selection are disabled, and the cuBLAS workspace is fixed. Unsupported
  deterministic operations are infrastructure errors, not silent fallbacks.
- SmolVLA checkpoint loading is strict: missing or unexpected model weights are
  errors rather than silently leaving random initialization in the policy.
- Initial state, task, current truck bounds, and policy-camera SHA-256 hashes are
  recorded. Action traces include applied and predicted actions, state, simulator
  timing, current truck geometry/counts, camera hashes, and particle accounting.

Readback timeouts or invalid images are infrastructure errors. All-black frames
receive bounded retries at the same paused physics state. An opaque black result
can represent an occluded camera and is delivered without fabricated fallback
pixels; camera timing describes the final attempt, while bridge latency includes
all attempts. This can add rendering work in an occluded view.

`--no-fresh-simulator`, `--no-render-every-physics-step`, `--no-headless`, and
reference-scene/replay modes produce `evaluation_profile: diagnostic`. They are
useful for investigating the simulator but cannot produce an official rate.
The persistent reset path remains supported and now rebuilds truck counting
bounds and inverse transforms after every truck relocation. Fresh and reset
scenes are not asserted to have identical sand/solver/renderer states.

The 2026-10-20 initialization/target correction changes the environment seen
by the policy; rerun earlier scores before comparing them with corrected runs.
The completion/scoring rule remains v2. See the
[regression investigation](smolvla_evaluation_regression_20261020.md) for the
controlled seed-26 tests and live verification.

## Payload and completion

Only particles in the initial pile source mask can count. Payload is the number
of their identities currently in the truck, excluding identities already there
at the initial observation. This identity exclusion remains correct if initial
material subsequently leaves the truck; subtracting a scalar baseline does not.

Truck geometry uses `current_bed_local_box_excluding_bucket_v3`: the current truck
bed collision mesh's local bounding box, inset by up to 0.05 m in XY, with a
0.03 m tolerance below its lower Z and 0.05 m above its upper Z. The current
inverse transform makes the test follow the truck's rotation. Particles still
inside the current bucket are excluded. A world bounding box is only a broad
filter; its corners outside the oriented bed cannot count. Bounds and scene seed
are included in observations; the evaluator rejects stale-scene metrics.
The authored truck is stationary within an episode; bounds are constructed at
initialization and rebuilt on scene reset. Arbitrary truck motion during a
rollout is outside this protocol.
The mesh box is computed without the prim's own transform, and tolerances are
converted by its axis scales before the inverse-transform test. This avoids
double-applying USD mesh transforms and handles the scaled truck asset.

An episode succeeds when BOTH conditions hold for three consecutive action
observations:

1. New source payload is at or above the configured threshold.
2. Bucket joint angle is at least -0.8 radians (open).

Increasing payload is allowed. Falling below the threshold or closing the
bucket resets the window. The initial zero-action observation does not count.
The episode terminates immediately at completion. `success` and
`task_completed_success_and_bucket_open` have identical meaning. A closed bucket
alone cannot contribute to the official success rate.

At the current 10 Hz policy rate, the three observations are spaced by 0.1 s;
the first-to-third span is 0.2 s. This is an observation-window rule, not an
additional 0.3 s waiting period. The maximum 350 actions represent 35 seconds of
policy simulation, excluding startup, settling, initial observation and final
video hold. Cycle time is defined only for successful complete tasks.

## Results, auditability and errors

Read `official_result_valid` and `official_success_rate` in
`success_rate_summary.json`. An official rate exists only when every requested
episode completes, no infrastructure/validation errors occur, and all rows use
one standard evaluation identity and model execution. Incomplete and diagnostic
batches retain descriptive metrics but their official rate is null. Runtime
errors stop the batch. Do not turn an error into a model failure or discard it
and report a smaller denominator.

The summarizer rejects mixed evaluation identities, profiles, execution sources,
duplicate indices and indices outside the requested batch. It validates that
reported successes meet the versioned completion rule. Legacy status files
remain readable and are never promoted to official v2 results.

`provenance.json` records hashes of checkpoint files, evaluator/simulator sources,
the collection root USD, dataset metadata, protocol parameters, Isaac/software
versions and GPU/driver inventory. `evaluation_id` identifies this complete manifest. It does not hash
all transitively referenced USD meshes/textures or a training dataset; preserve
the installed assets and software environment when comparing benchmarks.
`config.json` records worker/GPU assignments and runtime choices. Videos are
optional; their encoding cost changes wall-time throughput, not simulation time.

Failure labels describe the terminal state:

| Label | Meaning |
| --- | --- |
| `no_scoop` | No source particle was observed in the bucket. |
| `retained_payload` | No deposit, but source material remains in the bucket. |
| `lost_payload` | Observed scooped material is outside both bucket and truck. |
| `insufficient_payload` | Positive deposit below the threshold. |
| `bucket_not_open` | Threshold payload is present but the bucket remains closed. |
| `completion_not_sustained` | Final payload/open state has not met the complete observation window. |
| `infrastructure_error` / `validation_error` | Evaluation did not yield a valid task result. |

Labels are a classification of measurements, not a diagnosis of the policy's
internal failure. Partial deposits take precedence over retained-load labels.

## Known limits

This protocol is self-consistent, not bitwise deterministic PhysX. GPU contact
solvers, sand settling and ray-traced images can still vary between launches,
GPUs, drivers or Isaac versions. Fixed render cadence and fresh scenes limit
history dependence; camera hashes expose changes without promising equality.
Repeat borderline cases and report confidence intervals rather than selecting
the best rerun. A seed identifies a sampled scene; it does not prove the task is
impossible or that every rollout must have the same outcome.
Command limits constrain drive targets. The contact solver can still produce
lag or overshoot in measured joint positions under load; tracking errors remain
visible in states/traces. No control supervisor compensates for these effects.

Bed occupancy is a geometric approximation, not a material-contact or rest
detector. In-flight particles inside the counted bed volume can count while the
completion window holds. Spillage includes material returned to the pile and
material still in flight. Particle IDs are sampled only at action observations,
so an entire scoop/transfer occurring between samples can be missed. The policy's
bucket-load observation retains its collection-compatible fallback, while
evaluation accounting always uses the strict source IDs.

Some world-valid sampled targets fall outside observed robot-relative training
radii. They remain in the benchmark; do not filter them after seeing outcomes.
The initial robot articulation comes from dataset episode 0 unless explicitly
overridden and recorded. This is a simulator task benchmark, not a guarantee of
generalization outside the collection environment.

## Verification

```bash
"$POLICY_PYTHON" -m unittest discover -s tests
PYTHONPATH=ExcavatorVLA-smolvla:scripts \
  "$POLICY_PYTHON" -m unittest discover -s ExcavatorVLA-smolvla/tests
# Independently replay the completion rule and check trace/geometry/timing invariants:
"$POLICY_PYTHON" scripts/audit_smolvla_success.py \
  --batch-dir outputs/eval/<completed-v2-run> --verify-sources
# Optional live geometry/reset regression (one GPU, no policy):
"$POLICY_PYTHON" tests/run_success_eval_reset_live.py \
  --gpu 0 --output outputs/eval/<new-reset-test-directory>
```

The archived raw-reference integration test is skipped when its original
October 12 fixture is unavailable. The reset, geometric counting, identity,
completion, rollout/report integration and process-lifecycle regressions are
self-contained. Live validation results are documented separately from these
unit tests; neither a small successful smoke run nor a unit test establishes a
500-scene checkpoint success rate.
