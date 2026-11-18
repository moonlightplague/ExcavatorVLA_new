# ExcavatorVLA

Collect digging and unloading demonstrations in Isaac Sim, train excavator policies on LeRobot datasets, and evaluate them in closed-loop simulation.

## Getting started

Clone the repository with its simulator and policy submodules, then follow the [environment setup guide](docs/setup.md).

```bash
git clone --recursive https://github.com/moonlightplague/ExcavatorVLA_new.git
cd ExcavatorVLA_new
```

From a configured checkout, `bash train.sh --dry-run` previews the default SmolVLA training configuration for `data/repeat/export_repeat`.

| Task | Guide |
| --- | --- |
| Collect and export demonstrations | [Collection and export](docs/collection.md) |
| Train a policy | [SmolVLA training](docs/training.md) · [ACT and other policies](ExcavatorVLA-vlm/README.md) |
| Evaluate a checkpoint | [Offline and simulator evaluation](docs/evaluation.md) · [Collection-matched scenes](docs/collection_matched_evaluation.md) |
| Find tools and experiment records | [Repository guide](docs/README.md) |
