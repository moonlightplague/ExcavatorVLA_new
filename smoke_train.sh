#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
BASE="$PWD"
POLICY_PYTHON="${POLICY_PYTHON:-$(command -v python)}"
echo "Python: $POLICY_PYTHON"
export HF_HOME="$BASE/models/hf-cache"
OUTPUT="$BASE/outputs/train/smolvla_smoke_$(date +%Y%m%d_%H%M%S)_$$"

INPUT_FEATURES="$("$POLICY_PYTHON" - <<'PYCODE'
import json
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
if "stage_loss_weight" in SmolVLAConfig.__dataclass_fields__:
    raise SystemExit("This environment has stage patches. Activate a clean Conda environment and export POLICY_PYTHON=$(command -v python). See README.md for setup.")
features = json.load(open("data/export/meta/info.json"))["features"]
inputs = {"observation.state": {"type": "STATE", "shape": features["observation.state"]["shape"]}}
for key, spec in features.items():
    if key.startswith("observation.images."):
        h, w, c = spec["shape"]
        inputs[key] = {"type": "VISUAL", "shape": [c, h, w]}
print(json.dumps(inputs))
PYCODE
)"

"$POLICY_PYTHON" -m lerobot.scripts.lerobot_train \
    --policy.path="$BASE/models/smolvla_base" \
    --policy.load_vlm_weights=false --policy.input_features="$INPUT_FEATURES" \
    --policy.device=cuda --policy.push_to_hub=false \
    --policy.scheduler_warmup_steps=1 --policy.scheduler_decay_steps=20 \
    --dataset.repo_id=local/excavator \
    --dataset.root="$BASE/data/export" --dataset.video_backend=pyav \
    --output_dir="$OUTPUT" --job_name=excavator_smolvla_smoke \
    --batch_size=1 --steps=20 --num_workers=0 \
    --log_freq=1 --save_freq=20 --eval_freq=0 --wandb.enable=false

echo "Checkpoint: $OUTPUT/checkpoints/last/pretrained_model"
