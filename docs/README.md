# Repository guide

## Workflows

Start with [environment setup](setup.md). Follow [collection and export](collection.md) to create a LeRobot dataset, [training](training.md) to fit SmolVLA, and [evaluation](evaluation.md) to measure offline error or simulator success. The [evaluation standard](smolvla_evaluation_standard.md) defines the official single-cycle score.

## Source layout

| Location | Contents |
| --- | --- |
| [Repository root](../) | Collection, training and evaluation launchers, the Isaac Python wrapper, and training YAML files |
| [scripts](../scripts/) | Policy training and evaluation entry points, shared runtime modules and ACT experiment runners |
| [scripts/analysis](../scripts/analysis/) | Standalone dataset audits, rollout diagnostics and experiment summaries |
| [scripts/experiments/r5](../scripts/experiments/r5/) | Historical Best-of-N fine-tuning recipe and paired checkpoint evaluation |
| [scripts/setup](../scripts/setup/) | Environment preparation |
| [ExcavatorVLA-isaac](../ExcavatorVLA-isaac/) | Simulation assets and demonstration collection |
| [baseline](../baseline/README.md) | Repeated-cycle expert and dataset acceptance rules |
| [ExcavatorVLA-smolvla](../ExcavatorVLA-smolvla/) | Simulation bridge and SmolVLA deployment |
| [ExcavatorVLA-vlm](../ExcavatorVLA-vlm/README.md) | Policy implementations, training and benchmarks |
| [tests](../tests/) | Integration and regression tests for the top-level workflows |

Standalone analysis commands use `python scripts/analysis/<filename>.py` from the repository root. The shared success auditor stays at `scripts/audit_smolvla_success.py` because evaluation runners load it from frozen source snapshots.

## Experiment records

- Repeated-cycle experiments include [SmolVLA observation history](smolvla_history.md), [original-objective ACT](act_original_repeat.md), and [π0.5 five-cycle evaluation](pi05_five_cycle_evaluation_20261111.md).
- The [ACT experiment archive](experiments/act/) contains dated reports on transport, spillage, recovery and unloading. The [retained ACT baseline](experiments/act/act_baseline_275_20261105.md) links to its source and checkpoint manifest.
- Simulator checks include [pipeline validation](simulation_pipeline_validation_20261029.md), [SmolVLA evaluation validation](smolvla_evaluation_validation_20261019.md), and the [retained-payload regression investigation](smolvla_evaluation_regression_20261020.md).
