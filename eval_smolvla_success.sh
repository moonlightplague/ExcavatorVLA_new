#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export HF_HOME="$PWD/models/hf-cache"
POLICY_PYTHON="${POLICY_PYTHON:-$(command -v python)}"
echo "Python: $POLICY_PYTHON"
CHECKPOINT="${1:?Usage: bash eval_smolvla_success.sh <pretrained_model_dir> [episodes] [steps] [min_particles]}"
"$POLICY_PYTHON" scripts/eval_smolvla_success.py \
    --checkpoint "$CHECKPOINT" --dataset "$PWD/data/export" \
    --episodes "${2:-1}" --steps "${3:-350}" --min-particles "${4:-1}" \
    --output "$PWD/outputs/eval/success_$(date +%Y%m%d_%H%M%S)_$$" "${@:5}"
