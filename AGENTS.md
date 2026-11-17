# Training defaults

- Unless the user explicitly selects another dataset, use `/root/gpufree-data/ExcavatorVLA_new/data/repeat` as the source for new ExcavatorVLA training experiments. Its current LeRobot publication is `export_repeat/`.
- Keep source datasets immutable. Save splits, fitted statistics, audits and other preparation artifacts in a new experiment output directory.
- Preserve explicitly requested checkpoint intervals; do not prune checkpoints the user asks to retain.
