# Changelog

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
