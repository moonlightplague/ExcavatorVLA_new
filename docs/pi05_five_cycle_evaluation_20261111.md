# PI05 C five-cycle checkpoint evaluation — 2026-11-11

The bucket20 history-C training run completed 20,000 optimizer updates at
09:37 China time. All ten requested 2k milestones are retained in
`outputs/experiments/pi05_repeat_history_c_bucket20_20k_20261109_01/train/`.

The cancelled single-cycle batch is preserved at
`outputs/eval/pi05_repeat_history_c_bucket20_checkpoints_dualgpu_20261111_01/`.
Its interruptions are not repeated-task policy outcomes.

## Active comparison

The replacement five-cycle comparison is
[`pi05_repeat_history_c_bucket20_full5_checkpoints_dualgpu_20261111_01`](../outputs/eval/pi05_repeat_history_c_bucket20_full5_checkpoints_dualgpu_20261111_01/).
Use its [`status.json`](../outputs/eval/pi05_repeat_history_c_bucket20_full5_checkpoints_dualgpu_20261111_01/status.json)
for the controller state and
[`summary.csv`](../outputs/eval/pi05_repeat_history_c_bucket20_full5_checkpoints_dualgpu_20261111_01/summary.csv)
for completed checkpoint results. A successful process exit means evaluation
completed; task success is reported separately.

| Setting | Value |
| --- | --- |
| Scene | Training reference 0, source-recorded PhysX setup, seed 2 |
| Task | Five model-controlled scoop/dump/physical-return cycles, one continuous scene |
| GPU 0 queue | 20k, 16k, 12k, 8k, 4k |
| GPU 1 queue | 18k, 14k, 10k, 6k, 2k |
| Action execution | Predict 10 actions, execute 3 before replanning |
| Budget | At most 500 actions per cycle, 2500 total, at 10 Hz |
| History | Causal image/state steps [-6, -3, 0], continuous across goals |
| Action profile | `excavator_bucket_slew20_v1`; bucket command-target slew limit 20 rad/s |
| Collection result | Every cycle scoops at least 5 new IDs, deposits at least 3 IDs, and returns |
| Strict result | Additionally deposits at least 100 IDs in every cycle |

Goals come from the recorded task coordinates. Completion uses observed
particle transfers and physical return; it never advances on expert timing.
An incomplete cycle ends the task and later cycles remain `not_attempted`.
This is a reference-scene diagnostic with recorded goals, not a held-out
success-rate or live fill-planner benchmark. The 20 rad/s setting describes
command-target slew, not measured bucket speed.

Each completed checkpoint writes
`gpu_N/step_NNNNNN/reference_0000/episodes/episode_0001/evaluation.mp4`, plus
`episode_status.json`, `repeat_metrics.json`, `repeat_audit.json`,
`repeat_task_spec.json`, `goal_updates.json`, and `steps.jsonl`.
Videos display the collection and strict thresholds and their separate outcomes.
Every accepted comparison row requires a completed model rollout, a nonempty
MP4 with a readable duration, and an independent repeated-trace audit.

## Validation before launch

All 29 focused tests passed: seven repeated-task tests (including a synthetic
full shared-evaluator rollout), seven checkpoint-comparison tests, three video
tests, and twelve runtime/lifecycle tests. Video captions fit the frame width.

`tests/run_repeat_goal_update_live.py --gpu 0` also passed against real PhysX:
four goal updates used zero physics ticks and preserved camera pixels,
controller targets, source IDs, previous actions, and subsequent 10 Hz action
cadence. The probe executed only two zero commands; it is transport validation,
not learned policy performance. Its temporary simulator artifacts were removed.

## Launch command

The current controller already owns the comparison directory. A separate run
must use a new output name.

```bash
.venv-pi05/bin/python -B -u ExcavatorVLA-vlm/scripts/eval_pi05_checkpoints.py \
  --train-dir outputs/experiments/pi05_repeat_history_c_bucket20_20k_20261109_01/train \
  --output outputs/eval/pi05_repeat_history_c_bucket20_full5_checkpoints_dualgpu_20261111_01 \
  --checkpoint-steps 2000 4000 6000 8000 10000 12000 14000 16000 18000 20000 \
  --gpus 0,1 --reference-episodes 0 --repeat-cycles 5 --repeat-cycle-steps 500 \
  --rollout-steps 2500 --replan-interval 3 --min-particles 100 --latest-first
```
