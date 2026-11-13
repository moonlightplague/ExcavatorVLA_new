# Changelog

## 2026-11-13 — Output artifact cleanup

- Reclaim 21.03 GiB of allocated space in `outputs/`, reducing usage from 142.30 to 121.27 GiB.
- Remove 1,624 obsolete rollout videos, 108 evaluation RGB frame caches and 14 Python cache files. Shorten 1,394 inactive text logs to their first 32 KiB and final 96 KiB with an explicit truncation marker.
- Preserve datasets, raw recovery/training captures, numeric traces and metrics, frozen sources, complete final checkpoints, and the explicitly retained PI05 comparison milestones (ten repeated-data checkpoints and five single-cycle checkpoints).
- Preserve the archived ACT baseline and its original evaluation videos, the original combined-random evaluation, and all PI05 comparison outputs. Historical summaries remain; removed visual artifacts require rerunning their evaluations.
- Verify that all 31,238 retained regular files keep their original inode, size and modification time, all 41 numeric checkpoints remain across 28 checkpoint groups, and no symlinks become broken. Pass the preserved ACT baseline checksum audit before and after cleanup, including 826 archived files and 80 original videos.
- Confirm no running job uses the outputs selected for cleanup. Existing root/submodule ignore rules continue to exclude generated outputs, and all submodules remain unchanged.

## 2026-11-13 — Latest coarse MPM repeat pipeline

- Integrate the baseline's explicit soil-profile checks, mass-aware coarse planning and actual-source surface queries.
- Enforce the requested 50 kg minimum delivered load in every coarse dig, including the final rounds, with independent export and retained-metadata audits.
- Add an explicit coarse mass policy and convert native-grain control thresholds to mass units, retaining legacy quality decisions and physical geometry checks.
- Retain all feasible coarse targets before ranking and align early cut completion with the requested delivered load.
- Record 9/16 validated ten-dig exports (56.25%); all 90 accepted loads are at least 74.56 kg. The 95% Wilson interval is 33.18–76.90%, so a general above-50% rate guarantee remains unproven.
- Preserve the historical fine recipe and document the provisional amount-9 coarse recipe, 4.52 qualifying exports/hour on two RTX 4090 GPUs, independent reader validation and failed prototypes in `baseline/COARSE_MPM_VALIDATION.md`.
- Pass 196 CPU regression checks and remove 15.06 GB of task-generated datasets/diagnostics; retain ignored shared coarse experiment artifacts and existing training/evaluation assets.

## 2026-11-12 — Repository artifact maintenance

- Maintain generated-artifact ignore rules in the root and all four submodules, including local tooling state, renderer caches, experiment tracking and coverage reports.
- Unstage and ignore 1,971 generated coarse MPM experiment files (888 MiB), preserving local results, Python probes and required texture inputs.
- Reclaim 6.01 GiB by removing regenerated caches and the superseded 20k/40k/60k repeated SmolVLA checkpoints; retain the complete final 80k checkpoint and its `last` link.
- Audit `data/`, `models/` and `outputs/`; preserve datasets, base weights, recorded research evidence and PI05 milestone checkpoints explicitly retained by the comparison configurations.
- Update all four submodule pins with maintenance-only commits and preserve pending application/evaluation changes separately.
- Validate with 155 CPU regression tests covering collection/export, training, deployment and evaluation contracts.

## 2026-11-11 — Five-cycle model evaluation

- Add opt-in, revision-checked goal updates without physics ticks, controller resets, new particle baselines or camera changes; keep legacy bridge clients immutable.
- Evaluate five model-controlled scoop/dump/physical-return cycles using recorded per-cycle task coordinates, per-cycle source-ID accounting, persistent C history and an independent trace/goal-transition audit.
- Distinguish archived collection thresholds from the separate 100-particle-per-cycle strict result; label recorded-goal evaluations as diagnostics, not live fill-planner generalization.
- Run disjoint checkpoint/video queues on both GPUs with live combined summaries, preserve cancelled single-cycle outputs, and cover the new path with synthetic end-to-end and real PhysX bridge regressions.
- Resume the ten-checkpoint five-cycle comparison after 29 focused tests and the live PhysX goal-update probe pass; display collection and strict thresholds/outcomes throughout evaluation videos and record the protocol in `docs/pi05_five_cycle_evaluation_20261111.md`.

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
