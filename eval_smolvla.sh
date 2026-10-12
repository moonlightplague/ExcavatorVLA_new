#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export HF_HOME="$PWD/models/hf-cache"
POLICY_PYTHON="${POLICY_PYTHON:-$(command -v python)}"
echo "Python: $POLICY_PYTHON"
CHECKPOINT="${1:?Usage: bash eval_smolvla.sh <pretrained_model_dir> [num_episodes]}"
"$POLICY_PYTHON" scripts/eval_smolvla.py \
    --checkpoint "$CHECKPOINT" --dataset "$PWD/data/export" \
    --num-episodes "${2:-1}" \
    --output "$PWD/outputs/eval/smolvla_$(date +%Y%m%d_%H%M%S)_$$"
