# Evaluation v2 validation — 2026-10-19

The accepted release is described in [the evaluation standard](smolvla_evaluation_standard.md).
Validation used the existing 040000 checkpoint from
`excavator_smolvla_20261014_190059_474909_268032` without changing its weights.

## Automated tests

```text
python -m unittest discover -s tests
Ran 64 tests: OK, 1 skipped

PYTHONPATH=ExcavatorVLA-smolvla:scripts python -m unittest discover -s ExcavatorVLA-smolvla/tests
Ran 68 tests: OK
```

There are 131 passes and one skip in total. The skip is the pre-existing
integration check whose archived October 12 raw episode is unavailable. No
evaluation regression is skipped. The tests include actual Isaac USD bindings
for translated, rotated and scaled bed geometry, rather than relying entirely
on geometry mocks.

Coverage includes source-particle identity accounting, bucket exclusion, oriented
bed counting, transform scaling and coordinate conventions, changing reset
bounds, canonical joint-limit APIs, headless idle rendering, physics cadence,
camera readback ownership, growing-payload completion, threshold/open-window
resets, invalid report rejection, process cleanup, interruption/video errors,
and independent saved-trace audits. The archived legacy launcher test now checks
that configuration controls and download commands are exposed, without pinning
settings or ports from a different historical experiment.

## Live release checks

Environment: two RTX 4090 GPUs, NVIDIA driver 580.126.09, Torch 2.10.0+cu128,
Isaac Sim `6.1.0-rc.26+release.49347.2d230af4.gl`.
GPU 1 was also hosting a separate Pi0.5 training job during the release checks.
The scoring/trace audits passed, but wall-time and inference latency from these
checks measure shared-load operation. Use idle GPUs for throughput comparisons;
that other job was left running.

All checks used the release source, 350 actions maximum, threshold 100,
replanning every three actions, and 14 simulator CPU threads per worker.

| Seed | GPU | Video | Result | Final payload | Actions | Simulator PID |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 8 | 0 | Off | Complete | 222 | 323 | 3455909 |
| 9 | 0 | Off | Complete | 537 | 216 | 3487018 |
| 12 | 1 | On | Complete | 594 | 231 | 3455923 |
| 13 | 1 | On | Complete | 688 | 229 | 3493772 |

Both workers started a different simulator for their second episode; every
episode records `simulator_reused: false`. Strict checkpoint loading and
deterministic Torch operation checks succeeded. Video output is 1024×864 at
10 FPS. Seed 12 has 252 frames and seed 13 has 250, matching one initial frame,
the action count, and the two-second final hold.

The independent auditor replayed completion from saved observations and checked
particle-count relationships, current-scene geometry, bounded command targets,
physical action durations, capture budgets, camera hashes, manifest identities,
and source hashes against the current release. Both batch audits passed, and
both complete batches have `official_result_valid: true`.

Accepted artifacts:

- `outputs/eval/standard_v2_release_seeds8_9_gpu0/`, including `audit.json`.
- `outputs/eval/standard_v2_release_seeds12_13_gpu1/`, including `audit.json` and MP4s.
- `outputs/eval/standard_v2_release_reset_validation/reset_validation.json`.

The evaluation identities for the two model batches are respectively
`011e048e463572e118e90f2062fec25e7caf8e4d925364f00fbed7c337831b4b`
and `fe3f3ed10d9bf18668fd78581658d1d364bbe1baa8d3fc6486f5d4659035c606`.
Their identities differ because their configured seed sets differ.

The live reset regression used one simulator PID (3512591) for seeds 2 → 12 → 2.
It verified source-tracker initialization, matching context/observation counting
bounds, replacement of bounds for seed 12, and restoration of the original seed-2
bounds. This validates the persistent diagnostic path without claiming exact
fresh/reset sand-state parity.

Earlier `standard_v2_validation_*`, `standard_v2_acceptance_*` and
`standard_v2_final_*` directories are development runs, retained for inspection.
The accepted release artifacts are the `standard_v2_release_*` directories above.

These four episodes validate integration. Use the complete fixed seed set for
checkpoint accuracy estimates; historical v1 rates require a v2 rerun for a
comparison under the new scoring rules.
