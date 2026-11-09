#!/usr/bin/env bash
set -euo pipefail

# Parse the whole collection routine before starting any long-running work.
# Otherwise, edits during collection can shift Bash's next read into old code.
main() {
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_REPO="$BASE/ExcavatorVLA-isaac"
ISAAC_PYTHON="${ISAAC_PYTHON:-$BASE/run_isaac_python.sh}"
POLICY_PYTHON="${POLICY_PYTHON:-$(command -v python)}"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    echo 'Usage: ./collect_repeat.sh <successful_episodes> [workers=1] [gpu_ids=0] [scoops=10] [baseline/Isaac options...]'
    echo 'Example: ./collect_repeat.sh 10 2 0,1 10'
    echo 'Defaults: Newton MPM, reachable side loading, sand amount 3, episode efficiency acceptance; see baseline/TUNING.md.'
    echo 'Default output: data/repeat/{raw,export_repeat}; override DATA_ROOT, EXPORT_DIR and POLICY_PYTHON as needed.'
    echo 'Each successful episode exports immediately; raw retains metrics only after validated export.'
    echo 'Combined batch metrics and averages: data/repeat/metrics/repeat_*_summary.json.'
    echo 'See baseline/README.md; baseline/collect_repeat.py --help lists baseline options.'
    return
fi
DATA_ROOT="${DATA_ROOT:-$BASE/data/repeat}"
COUNT="${1:?Usage: ./collect_repeat.sh <successful_episodes> [workers] [gpu_ids] [scoops]}"
WORKERS="${2:-${WORKERS:-1}}"
SCOOPS="${4:-${SCOOPS_PER_EPISODE:-10}}"
STATUS_SECONDS="${COLLECT_STATUS_SECONDS:-15}"
[[ "$COUNT" =~ ^[1-9][0-9]*$ && "$WORKERS" =~ ^[1-9][0-9]*$ ]] || {
    echo "Success count and workers must be positive integers." >&2
    exit 2
}
[[ "$STATUS_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
    echo "COLLECT_STATUS_SECONDS must be a positive integer." >&2
    exit 2
}
[[ "$SCOOPS" =~ ^[1-9][0-9]*$ ]] && (( SCOOPS >= 2 && SCOOPS <= 64 )) || {
    echo "Scoops per episode must be an integer between 2 and 64." >&2
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
EXTRA_ARGS=()
if (( $# > 4 )); then EXTRA_ARGS=("${@:5}"); fi
command -v setsid >/dev/null
if ! "$POLICY_PYTHON" -c 'import numpy, pandas, pyarrow, imageio_ffmpeg'; then
    echo "Set POLICY_PYTHON to your training Python with numpy, pandas, pyarrow and imageio-ffmpeg installed." >&2
    exit 2
fi
POLICY_ENV_JSON="$("$POLICY_PYTHON" -c 'import json,os; print(json.dumps({key:os.environ.get(key) for key in ("PATH", "PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LD_PRELOAD")}))')"
# Resolve before changing directory; callers may supply a relative DATA_ROOT.
mkdir -p "$DATA_ROOT"
DATA_ROOT="$(cd "$DATA_ROOT" && pwd)"
EXPORT_DIR="$("$POLICY_PYTHON" -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${EXPORT_DIR:-$DATA_ROOT/export_repeat}")"
BATCH_ID="repeat_$(date +%Y%m%d_%H%M%S)_$$"
LOG_ROOT="$DATA_ROOT/raw/.collection_logs/$BATCH_ID"
METRICS_FILE="$DATA_ROOT/metrics/${BATCH_ID}_summary.json"
mkdir -p "$LOG_ROOT"
PIDS=()
BATCH_STATUS=failed
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
finish_collection() {
    local status=$?
    trap - EXIT
    cleanup
    if (( status == 130 || status == 143 )); then BATCH_STATUS=interrupted; fi
    if ! "$POLICY_PYTHON" "$BASE/baseline/metrics_summary.py" \
        --raw-root "$DATA_ROOT/raw" --batch-id "$BATCH_ID" \
        --export-dir "$EXPORT_DIR" --requested "$COUNT" \
        --status "$BATCH_STATUS" --output "$METRICS_FILE"; then
        echo "Failed to write combined metrics: $METRICS_FILE" >&2
        status=1
    fi
    exit "$status"
}
trap finish_collection EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Collecting $COUNT successful episodes, $SCOOPS dig/unload/physical-return cycles each, with $WORKERS worker(s), GPUs=$GPU_IDS"
echo "Worker logs and results: $LOG_ROOT"
echo "Streaming export: $EXPORT_DIR; combined metrics: $METRICS_FILE"
echo "Progress is reported every ${STATUS_SECONDS}s; scene/RTX initialization can take several minutes."
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
        EXCAVATOR_KIT_LOG_FILE="$ROUND_DIR/worker_$WORKER.kit.log" \
        REPEAT_POLICY_ENV="$POLICY_ENV_JSON" \
        PYTHONUNBUFFERED=1 \
        setsid "$ISAAC_PYTHON" "$BASE/baseline/collect_repeat.py" \
            --max-attempts "$((TARGET * 5))" \
            --sand-amount "${SAND_AMOUNT:-3.0}" "${EXTRA_ARGS[@]}" \
            --active-gpu "$GPU" --physics-gpu "$GPU" --success-count "$TARGET" \
            --dataset-root "$DATA_ROOT/raw" --scoops-per-episode "$SCOOPS" \
            --stream-export-dir "$EXPORT_DIR" --policy-python "$POLICY_PYTHON" \
            >"$ROUND_DIR/worker_$WORKER.log" 2>&1 &
        PIDS+=("$!")
    done

    # Keep reporting even when a native Kit update blocks Python's heartbeat.
    # Poll in the parent so there is no extra monitor process to clean up.
    ROUND_STARTED=$SECONDS
    NEXT_STATUS=$SECONDS
    while :; do
        RUNNING=0
        for PID in "${PIDS[@]}"; do
            if kill -0 "$PID" 2>/dev/null; then RUNNING=1; fi
        done
        (( RUNNING != 0 )) || break
        if (( SECONDS >= NEXT_STATUS )); then
            NOW=$(date +%s)
            for (( WORKER=0; WORKER<ACTIVE_WORKERS; WORKER++ )); do
                PID="${PIDS[$WORKER]}"
                if ! kill -0 "$PID" 2>/dev/null; then continue; fi
                WORKER_LOG="$ROUND_DIR/worker_$WORKER.log"
                LAST_LINE=$(tail -n 1 "$WORKER_LOG" 2>/dev/null || true)
                LOG_UPDATED=$(stat -c %Y "$WORKER_LOG" 2>/dev/null || echo "$NOW")
                echo "[worker $WORKER pid=$PID elapsed=$((SECONDS - ROUND_STARTED))s log_age=$((NOW - LOG_UPDATED))s] ${LAST_LINE:-Starting Isaac Sim...}"
            done
            NEXT_STATUS=$((SECONDS + STATUS_SECONDS))
        fi
        sleep 1
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
if successes > result["requested"]:
    raise SystemExit("Collection returned more successes than requested.")
if not result.get("run_dir"):
    raise SystemExit("Collection result has no raw run directory.")
if str(result.get("loop_exit_reason", "")).startswith("exception:"):
    raise SystemExit("Collection failed: " + result["loop_exit_reason"])
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
        # Workers already exported and pruned before recording another episode.
        # A simulation success alone cannot count as a committed export.
        if ! "$POLICY_PYTHON" "$BASE/baseline/stream_export.py" \
            --run-dir "$RUN" --export-dir "$EXPORT_DIR" --verify-successes "$SUCCESSES"; then
            echo "Worker $WORKER has missing export/cleanup receipts; raw files retained." >&2
            FAILED=1
            continue
        fi
        GAINED=$((GAINED + SUCCESSES))
    done
    REMAINING=$((REMAINING - GAINED))
    echo "Exported $((COUNT - REMAINING))/$COUNT successful episodes to $EXPORT_DIR"
    if (( FAILED != 0 || GAINED == 0 )); then
        echo "Collection stopped after worker failure or no progress; raw data and logs remain in $DATA_ROOT/raw." >&2
        exit 1
    fi
done
BATCH_STATUS=complete
}

# Parse the exit together with the call so Bash never rereads an edited file.
main "$@"; exit
