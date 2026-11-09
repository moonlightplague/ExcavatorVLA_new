# Changelog

## 2026-11-09 — Newton ten-cycle repeat tuning

- Add reachable side-loading and source-ID episode acceptance options to the repeat baseline, with a fixed-attempt acceptance benchmark and retained strict counterfactual decisions.
- Repair Newton thin static-shell tunneling using actual collision triangles and sync the collection and deployment soil adapters.
- Document repeat collection failures, parameter choices and acceptance semantics in `baseline/TUNING.md`; preserve the original user collection data.
- Measure 2/4 validated ten-round Newton exports under explicit episode efficiency acceptance (1/4 strict-equivalent); validate both publications and the LeRobot loader, retaining the full outcome/uncertainty report in `baseline/VALIDATION.md`.

## 2026-11-09 — PI05 repeated-data repair experiment

- Record experimental action-limit profiles in evaluation provenance and reject silent model/profile mismatches.
- Add explicit full-reference expert replay without first-scoop early termination; keep ordinary model and single-cycle evaluation unchanged.
- Prepare the versioned PI05 bucket20 repeated-data candidate and gated dual-GPU C training at 20k updates with 2k checkpoint retention; actual gate/training outcomes remain in ignored experiment outputs.

## 2026-11-08 — PI05 PhysX training and evaluation

- Maintain episode-local image/state windows for PI05 scheme C during chunk execution and record the history contract/window indices in evaluation provenance and traces.
- Update the VLM submodule with verified-backend single-cycle PI05 training, continuation, and checkpoint-comparison tools.
- Support source-recorded reference setups and collection-prepared scene manifests in the shared evaluator, including per-scene initial poses, targets, and provenance hashes.
- Add the collection-matched multi-GPU evaluation runner, scene validation, regression coverage, and protocol documentation; update the baseline scene-preparation dependency.
- Ignore the standalone generated output video; datasets, model weights, checkpoints, logs, and evaluation outputs remain outside these commits.

## 2026-10-30 — Repeated digging and repository maintenance

- Update the baseline to use continuous collision-checked cuts and record per-episode transfer, timing, throughput, and bed-fill metrics, including failures.
- Update the Isaac exporter to retain physical-return supervision for both baseline metadata versions.
- Validate with 170 regression tests, the baseline training dry-run, and the live three-scoop collection/export (690 frames per camera); remove the temporary validation artifacts.
- Audit `data/`, `models/`, and `outputs/`; retain published datasets, base models, research results, and one final checkpoint in each of 15 training runs. Remove 1.12 GiB of unused temporary checkpoints, smoke-test artifacts, logs, and caches.
- Maintain generated-artifact ignore rules in the main repository and all submodules.
- Stop tracking local editor settings while keeping the local file, and remove obsolete tracked simulator backup copies from the SmolVLA submodule.

## 2026-10-30

- Record and validate simulator soil configuration in evaluation provenance, and report live progress and rollout diagnostics.
- Update simulation evaluation fixtures and document collection and evaluation validation.
- Update SmolVLA and VLM submodules with simulation settings, PI05 LoRA training, and rollout selection tools.
- Retain final training checkpoints while removing intermediate checkpoints and completed smoke/probe artifacts locally. Historical evaluations at earlier steps retain their metrics but cannot be rerun from removed checkpoints.
