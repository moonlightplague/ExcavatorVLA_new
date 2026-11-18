# Evaluate SmolVLA

## Offline action error

From the repository root, set `CHECKPOINT` to the pretrained model directory you want
to evaluate. The root evaluation launchers use the single-cycle `data/export`
dataset. For repeated digging and unloading, see the [repeat evaluation workflow](smolvla_history.md).

Evaluate the first N episodes in `data/export` (default: 1):

```bash
cd "$BASE"
bash eval_smolvla.sh "$CHECKPOINT" 1
```

Results: `outputs/eval/smolvla_*/{metrics.json,actions.npz}`.
Reports action MAE/RMSE in rad/s. Training-data evaluation is a fitting check,
not a measure of generalization or simulator success.

## Closed-loop success rate

The final protocol is **`excavator_success_v2`**, specified in
[the evaluation standard](smolvla_evaluation_standard.md). It runs headlessly
with one policy worker per visible GPU, capped by episode count. Policies remain
loaded, while each episode starts a fresh Isaac Sim process and scene. Every
physics tick renders. Historical v1 scores use a different counting/completion
rule and must be reported separately.

```bash
cd "$BASE"
# Checkpoint, episodes, maximum policy steps, minimum deposited particles
bash eval_smolvla_success.sh "$CHECKPOINT" 10 350 100
# Reference benchmark: seeds 2–501, 350 actions, 100 source particles
bash eval_smolvla_success.sh "$CHECKPOINT" 500 350 100 --gpus 0,1 --no-video
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
export REFERENCE="data/raw/<run>/episode_<id>"
bash eval_smolvla_success.sh "$CHECKPOINT" 1 350 100 --reference-episode "$REFERENCE"
# Test exported expert actions independently of the policy:
bash eval_smolvla_success.sh "$CHECKPOINT" 1 350 100 --reference-episode "$REFERENCE" --replay-reference
# Audit causal alignment, cadence, limits and normalization:
"$POLICY_PYTHON" scripts/analysis/audit_smolvla_data.py --dataset data/export --chunk-size 10 --output outputs/analysis/data_audit.json
```

Reference evaluation fails when the initial target-relative geometry differs
from collection by more than 5 cm. `initial_observation.json` and `steps.jsonl`
record state and commanded positions for debugging. Expert replay is explicitly
marked `execution_source: expert_replay`; its success is not model success.
Particle regeneration and contact physics can still differ between launches.
