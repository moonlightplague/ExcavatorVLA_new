#!/usr/bin/env bash
set -euo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_REPO="$BASE/ExcavatorVLA-isaac"
ISAAC_PYTHON="$BASE/run_isaac_python.sh"
POLICY_PYTHON="${POLICY_PYTHON:-/root/gpufree-data/conda/envs/ExcavatorVLA/bin/python}"
DATA_ROOT="$BASE/data"
COUNT="${1:?Usage: bash collect.sh <success_count>}"
[[ "$COUNT" =~ ^[1-9][0-9]*$ ]] || exit 2
"$POLICY_PYTHON" -c 'import pandas, pyarrow, imageio_ffmpeg'

export EXCAVATOR_AUTO_COLLECT_RESULT_FILE="$(mktemp)"
trap 'rm -f "$EXCAVATOR_AUTO_COLLECT_RESULT_FILE"' EXIT

cd "$ISAAC_REPO"
REMAINING="$COUNT"
while (( REMAINING > 0 )); do
    "$ISAAC_PYTHON" run_vla_train_scene.py \
        --headless --auto-collect --no-bridge \
        --success-count "$REMAINING" --max-attempts "$((REMAINING * 5))" \
        --sand-amount 1.0 --dataset-root "$DATA_ROOT/raw" --disable-export

    RESULT="$("$POLICY_PYTHON" - "$EXCAVATOR_AUTO_COLLECT_RESULT_FILE" <<'PYCODE'
import json, sys
result = json.load(open(sys.argv[1]))
if result.get("loop_cancelled"):
    raise SystemExit("Collection cancelled.")
print(result["run_dir"], result["successes"])
PYCODE
)"
    read -r RUN SUCCESSES <<< "$RESULT"
    for (( EPISODE=0; EPISODE<SUCCESSES; EPISODE++ )); do
        "$POLICY_PYTHON" excavator_dataset_tools.py "$RUN" \
            --export-lerobot-v3 --export-append \
            --export-offset "$EPISODE" --export-limit 1 \
            --export-dir "$DATA_ROOT/export" \
            --export-require-standard --export-require-vla \
            2>&1 | tee "$RUN/export_episode_${EPISODE}.log"
    done
    REMAINING=$((REMAINING - SUCCESSES))
    echo "Exported $((COUNT - REMAINING))/$COUNT successful episodes to $DATA_ROOT/export"
done
