# Changelog

## 2026-10-30

- Record and validate simulator soil configuration in evaluation provenance, and report live progress and rollout diagnostics.
- Update simulation evaluation fixtures and document collection and evaluation validation.
- Update SmolVLA and VLM submodules with simulation settings, PI05 LoRA training, and rollout selection tools.
- Retain final training checkpoints while removing intermediate checkpoints and completed smoke/probe artifacts locally. Historical evaluations at earlier steps retain their metrics but cannot be rerun from removed checkpoints.
