# Collection-matched PI05 evaluation

The first random-100 batch selected the truck-bed center and used one fixed
initial pose. It is a different scene/target protocol from the 20 training
episodes and must not be pooled with this evaluation.

The replacement prepares new scenes with the actual collection runtime:

- Collection `area_uniform_polar_v2` scene sampling, PhysX, sand amount 1.0.
- `baseline.repeat_runtime.install(..., RepeatConfig(scoops=5))`, as in the
  source collection. Its initial pose faces the current pile.
- The original first-cycle planning path, including `RepeatController.choose_landing`,
  the 5×3 measured-fill grid, safe inset, radial reach gate, exact dump IK,
  loaded-carry/collision checks, and digging-target planning.
- Stop before recording or executing a scoop. Capture only the scene, initial
  joints, feasible dig/unload targets, scores and feasibility evidence.
- Execute the trained model in a fresh deployment simulator for each captured
  setup, with the same single-cycle instruction as the pilot training set.

Preparation rejects geometrically infeasible or unplannable scenes before
policy evaluation, matching the collection planning path. `attempts.jsonl`
records these exclusions; they are not model failures. Selection never uses
model outcomes, measured model payload, or an expert execution result. This
matches collection **preparation and planning**, not the retrospective filter
that keeps only successful expert demonstrations.

Each GPU prepares and then evaluates half the scenes. The runner first finishes
both preparation workers, freezes a combined manifest, then launches persistent
policy workers with a fresh simulator per scene. Status can therefore show
`preparing_collection_scenes` before any model evaluation/video exists.

```bash
.venv-pi05/bin/python -u scripts/run_collection_matched_eval.py \
  --checkpoint outputs/experiments/pi05_singlecycle20_physx_10k_20261102_02/train/checkpoints/006000/pretrained_model \
  --dataset data/pi05_singlecycle_physx_pilot_v2 \
  --episodes 100 --seed 2026110300 --gpus 0,1 \
  --job outputs/experiments/pi05_6k_collection_matched100_dualgpu_20261103_02 \
  --output outputs/eval/pi05_singlecycle20_physx_6k_collection_matched100_dualgpu_20261103_02
```

The job directory contains `status.json`, `source_snapshot.json`, per-worker
preparation logs, `scene_manifest.json` and `evaluation.log`. The evaluation
directory contains `success_rate_summary.json`, `episode_results.csv` and
`episodes/episode_XXXX/evaluation.mp4`.

Every episode also stores `collection_setup.json` and `scene_profile.json`.
The evaluator validates selector/backend consistency, unique seeds, collection
reach/IK evidence, target equality, and the actual initial joints. The manifest
hash and target-selection method enter the evaluation provenance, preventing
this batch from being aggregated with the earlier center-target batch.

The model still predicts 10 actions, executes 3 before replanning, and has a
350-step limit. Completion still requires at least 100 deposited particles
and an open bucket for three consecutive observations. This is a single-cycle
evaluation; it does not establish repeated-digging performance.
