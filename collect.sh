#!/usr/bin/env bash
set -euo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_REPO="$BASE/ExcavatorVLA-isaac"
ISAAC_PYTHON="${ISAAC_PYTHON:-$BASE/run_isaac_python.sh}"
POLICY_PYTHON="${POLICY_PYTHON:-$(command -v python)}"
DATA_ROOT="${DATA_ROOT:-$BASE/data}"
COUNT="${1:?Usage: bash collect.sh <success_count> [workers] [gpu_ids, e.g. 0,1,2]}"
WORKERS="${2:-${WORKERS:-1}}"
[[ "$COUNT" =~ ^[1-9][0-9]*$ && "$WORKERS" =~ ^[1-9][0-9]*$ ]] || {
    echo "Success count and workers must be positive integers." >&2
    exit 2
}
GPU_IDS="${3:-${GPU_IDS:-}}"
if [[ -z "$GPU_IDS" ]]; then
    GPU_IDS=0
    for (( GPU=1; GPU<WORKERS; GPU++ )); do GPU_IDS+=",$GPU"; done
fi
[[ "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
    echo "GPU IDs must be comma-separated non-negative integers." >&2
    exit 2
}
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
command -v setsid >/dev/null
"$POLICY_PYTHON" -c 'import pandas, pyarrow, imageio_ffmpeg'
# Resolve before changing directory; callers may supply a relative DATA_ROOT.
mkdir -p "$DATA_ROOT"
DATA_ROOT="$(cd "$DATA_ROOT" && pwd)"
BATCH_ID="collection_$(date +%Y%m%d_%H%M%S)_$$"
LOG_ROOT="$DATA_ROOT/raw/.collection_logs/$BATCH_ID"
mkdir -p "$LOG_ROOT"
PIDS=()
cleanup() {
    local pid
    for pid in "${PIDS[@]}"; do
        kill -TERM -- "-$pid" 2>/dev/null || true
    done
    if (( ${#PIDS[@]} > 0 )); then
        sleep 2
        for pid in "${PIDS[@]}"; do
            kill -KILL -- "-$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        done
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Collecting $COUNT successful episodes with $WORKERS worker(s), GPUs=$GPU_IDS"
echo "Worker logs and results: $LOG_ROOT"
cd "$ISAAC_REPO"
REMAINING="$COUNT"
ROUND=0
while (( REMAINING > 0 )); do
    ROUND=$((ROUND + 1))
    ACTIVE_WORKERS=$(( REMAINING < WORKERS ? REMAINING : WORKERS ))
    ROUND_DIR="$LOG_ROOT/round_$ROUND"
    mkdir -p "$ROUND_DIR"
    PIDS=()
    for (( WORKER=0; WORKER<ACTIVE_WORKERS; WORKER++ )); do
        TARGET=$(( REMAINING / ACTIVE_WORKERS + (WORKER < REMAINING % ACTIVE_WORKERS) ))
        GPU="${GPUS[$((WORKER % ${#GPUS[@]}))]}"
        echo "Round $ROUND worker $WORKER: GPU=$GPU target=$TARGET"
        EXCAVATOR_AUTO_RUN_ID_SUFFIX="${BATCH_ID}_r${ROUND}_w${WORKER}" \
        EXCAVATOR_AUTO_COLLECT_RESULT_FILE="$ROUND_DIR/worker_$WORKER.result.json" \
        PYTHONUNBUFFERED=1 \
        setsid "$ISAAC_PYTHON" run_vla_train_scene.py \
            --headless --auto-collect --no-bridge \
            --active-gpu "$GPU" --physics-gpu "$GPU" \
            --success-count "$TARGET" --max-attempts "$((TARGET * 5))" \
            --sand-amount 1.0 --dataset-root "$DATA_ROOT/raw" --disable-export \
            >"$ROUND_DIR/worker_$WORKER.log" 2>&1 &
        PIDS+=("$!")
    done

    FAILED=0
    for PID in "${PIDS[@]}"; do
        if wait "$PID"; then :; else FAILED=1; fi
    done
    PIDS=()
    GAINED=0
    for (( WORKER=0; WORKER<ACTIVE_WORKERS; WORKER++ )); do
        RESULT_FILE="$ROUND_DIR/worker_$WORKER.result.json"
        if [[ ! -s "$RESULT_FILE" ]]; then
            echo "Worker $WORKER did not produce a result; see $ROUND_DIR/worker_$WORKER.log" >&2
            FAILED=1
            continue
        fi
        if RESULT="$("$POLICY_PYTHON" - "$RESULT_FILE" <<'PYCODE'
import json, sys
result = json.load(open(sys.argv[1]))
if result.get("loop_cancelled"):
    raise SystemExit("Collection cancelled.")
successes = result["successes"]
if type(successes) is not int or successes < 0:
    raise SystemExit("Invalid collection success count.")
print(result["run_dir"])
print(successes)
PYCODE
        )"; then
            mapfile -t RESULT_LINES <<< "$RESULT"
            RUN="${RESULT_LINES[0]}"
            SUCCESSES="${RESULT_LINES[1]}"
        else
            FAILED=1
            continue
        fi
        # Only the parent exports, so workers never contend for the append lock.
        for (( EPISODE=0; EPISODE<SUCCESSES; EPISODE++ )); do
            "$POLICY_PYTHON" excavator_dataset_tools.py "$RUN" \
                --export-lerobot-v3 --export-append \
                --export-offset "$EPISODE" --export-limit 1 \
                --export-dir "$DATA_ROOT/export" \
                --export-require-standard --export-require-vla \
                2>&1 | tee "$RUN/export_episode_${EPISODE}.log"
        done
        GAINED=$((GAINED + SUCCESSES))
    done
    REMAINING=$((REMAINING - GAINED))
    echo "Exported $((COUNT - REMAINING))/$COUNT successful episodes to $DATA_ROOT/export"
    if (( FAILED != 0 || GAINED == 0 )); then
        echo "Collection stopped after worker failure or no progress; raw data and logs remain in $DATA_ROOT/raw." >&2
        exit 1
    fi
done
