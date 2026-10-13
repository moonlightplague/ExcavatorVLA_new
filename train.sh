#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
POLICY_PYTHON="${POLICY_PYTHON:-$(command -v python)}"
echo "Python: $POLICY_PYTHON"
exec "$POLICY_PYTHON" scripts/train_smolvla.py "$@"
