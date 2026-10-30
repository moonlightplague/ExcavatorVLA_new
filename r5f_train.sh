#!/usr/bin/env bash
# r5f_train: chained launcher for the Best-of-N supervised fine-tune.
#
# Waits for r5f_build.sh to finish the dataset, verifies it independently, waits
# for the evaluation pool to release both GPUs, re-validates the real config,
# then trains to completion and reports. Stops before r5g: choosing which
# checkpoints to score on the 100 held-out seeds is a decision that should see
# the loss curve first, and each 100-seed Isaac evaluation costs ~2 hours
# against ~17 minutes of training.
#
# Every gate aborts rather than proceeds. The expensive failure mode here is not
# a crash but a silent 2-hour evaluation of a policy trained on the wrong data.
set -u
ROOT=/root/gpufree-data/ExcavatorVLA_new
PY=/root/gpufree-data/conda/envs/ExcavatorVLA-current/bin/python
DS=$ROOT/outputs/datasets/r5f_bon_t400
SEL=$ROOT/outputs/selection/selection_r5dbig_t400.json
YAML=$ROOT/training_config_r5f.yaml
BUILD_PID=1702904
BUILD_LOG=/tmp/r5f_build.log
STDOUT_LOG=$ROOT/outputs/rl/r5f_train_stdout.log
mkdir -p "$ROOT/outputs/rl"
cd "$ROOT"
# The box has no external network. Without the offline trio every HF cache
# lookup burns 5 connect-timeout retries (~90 s per file); the 13:59Z launch
# stalled 15+ min walking tokenizer files before reaching a single step.
# Cache is complete (040000 baseline trained here; probe60 dry-run loaded
# the pretrained model offline), so pure-cache mode is safe.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

fail() { echo "ABORT: $*" >&2; exit 1; }

echo '=== 1. wait for r5f_build.sh to finish the dataset ==='
# r5f_build.sh itself waits on the rollout pool, so this waits transitively on
# r5dbig. It ends with a fixed banner and leaves AUDIT_OK / SMOKE_OK in its log;
# require all three so a build that died halfway cannot look like success.
deadline=$(( $(date +%s) + 7200 ))
while :; do
  if ! kill -0 "$BUILD_PID" 2>/dev/null; then
    echo "$(date -u '+%H:%M:%SZ') build runner $BUILD_PID exited"
    break
  fi
  echo "$(date -u '+%H:%M:%SZ') build runner alive ($(ps -p "$BUILD_PID" -o etime= | tr -d ' ')); $(tail -n 1 "$BUILD_LOG" 2>/dev/null)"
  [ "$(date +%s)" -ge "$deadline" ] && fail "build runner still alive after 120 min"
  sleep 60
done

echo '--- build log, sections 2 onward ---'
sed -n '/=== 2\./,$p' "$BUILD_LOG" 2>/dev/null | head -200
echo '--- build gate ---'
grep -q '=== 10\. done' "$BUILD_LOG" || fail "build did not reach section 10 (see $BUILD_LOG)"
grep -q '^AUDIT_OK$' "$BUILD_LOG" || fail 'contract.audit_dataset did not report AUDIT_OK'
grep -q '^SMOKE_OK$' "$BUILD_LOG" || fail 'LeRobotDataset smoke did not report SMOKE_OK'
grep -qE '^RC=0$' "$BUILD_LOG" || fail 'some build step returned nonzero'
nfail=$(grep -c '^RC=' "$BUILD_LOG")
nzero=$(grep -c '^RC=0$' "$BUILD_LOG")
echo "build steps returning RC: $nzero/$nfail were 0"
[ "$nfail" -eq "$nzero" ] || fail "build had $((nfail - nzero)) nonzero RC step(s)"
echo BUILD_GREEN

echo
echo '=== 2. independent dataset verification (do not trust the build log alone) ==='
[ -d "$DS" ] || fail "dataset dir missing: $DS"
"$PY" - "$DS" "$SEL" <<'PY'
import json
import sys
from pathlib import Path

root, sel_path = Path(sys.argv[1]), Path(sys.argv[2])
info = json.loads((root / 'meta' / 'info.json').read_text())
prov = json.loads((root / 'rl_provenance.json').read_text())
sel = json.loads(sel_path.read_text())
eps, frames = info['total_episodes'], info['total_frames']
print(f'info: episodes={eps} frames={frames} tasks={info["total_tasks"]} '
      f'fps={info["fps"]} schema={info["codebase_version"]}')
# The selection file calls it 'criteria'; only rl_provenance.json calls it
# 'selection_criteria'. probe59: using the provenance name here raised KeyError
# and aborted the runner with the dataset perfectly healthy.
crit = sel['criteria']
print(f'selection: schema={sel["schema_version"]} mode={crit["mode"]} '
      f'min_score={crit["min_score"]} selected={len(sel["selected"])}')
print(f'provenance: weight_mode={prov["weight_mode"]} stats_mode={prov["stats_mode"]} '
      f'state={prov["state_source"]} action={prov["action_source"]} '
      f'episodes={prov["total_episodes"]} frames={prov["total_frames"]} skipped={len(prov["skipped"])}')

problems = []
# The dataset must be built from exactly the threshold-400 selection.
if Path(prov['source_selection']).name != sel_path.name:
    problems.append(f'source_selection is {prov["source_selection"]}, expected {sel_path}')
if prov['total_episodes'] != eps or prov['total_frames'] != frames:
    problems.append('provenance totals disagree with info.json')
# info.json counts what landed; provenance counts what was selected. They differ
# only if build_finetune_dataset skipped an episode, which must never be silent.
if eps != len(sel['selected']):
    problems.append(f'info has {eps} episodes but the selection has {len(sel["selected"])}')
if prov['skipped']:
    problems.append(f'build skipped {len(prov["skipped"])} episode(s): {prov["skipped"][:3]}')
if prov['weight_mode'] != 'uniform':
    problems.append(f'weight_mode is {prov["weight_mode"]}, expected uniform')
if prov['stats_mode'] != 'reference':
    problems.append(f'stats_mode is {prov["stats_mode"]}, expected reference')
# info.json carries the LeRobot on-disk layout version ('v3.0'). 'current-v4' is
# the *contract* schema reported by audit_dataset -- a different namespace. probe60
# confirmed v3.0 on the real dataset.
if info['codebase_version'] != 'v3.0':
    problems.append(f'schema is {info["codebase_version"]}, expected v3.0')
if info['fps'] != 10.0:
    problems.append(f'fps is {info["fps"]}, expected 10.0')
for key, shape in (('observation.state', [28]), ('action', [4])):
    got = info['features'][key]['shape']
    if got != shape:
        problems.append(f'{key} shape {got} != {shape}')
if 'action_loss_weight' not in info['features']:
    problems.append('action_loss_weight feature missing -- excavator_forward would ignore the weights')
# 82 seeds were rolled out; a threshold-400 cut should keep most of them. Fewer
# than 30 surviving episodes means the pool or the gate went wrong.
if not 30 <= eps <= 82:
    problems.append(f'episode count {eps} outside the plausible 30..82 band')
if frames < eps * 100:
    problems.append(f'only {frames / max(eps, 1):.0f} frames per episode on average; rollouts run ~180 steps')
# Held-out seeds must not have leaked into training.
holdout = set(range(402, 502))
leaked = sorted({e['scene_seed'] for e in prov['episodes'] if e['scene_seed'] in holdout})
if leaked:
    problems.append(f'HOLDOUT SEEDS LEAKED INTO TRAINING: {leaked}')
train_seeds = sorted(e['scene_seed'] for e in prov['episodes'])
print(f'training seeds ({len(train_seeds)}): {train_seeds}')
print(f'stats_warnings ({len(prov["stats_warnings"])}):')
for w in prov['stats_warnings']:
    print(f'  {w}')
if problems:
    for p in problems:
        print(f'PROBLEM: {p}')
    raise SystemExit(1)
print('DATASET_VERIFIED')
PY
[ $? -eq 0 ] || fail 'dataset verification found problems'
du -sh "$DS"
find "$DS" -type f | wc -l
df -h "$ROOT" | tail -1
avail_kb=$(df --output=avail -k "$ROOT" | tail -1 | tr -d ' ')
[ "$avail_kb" -gt 20971520 ] || fail "only $((avail_kb / 1024 / 1024))G free; 4 checkpoints need ~6G plus headroom"

echo
echo '=== 3. wait for both GPUs to be released by the rollout pool ==='
# The pool runs two lanes, each a parent plus a worker, so pgrep reports 4 while
# healthy and flickers to 3 or 0 between seeds. Require three consecutive quiet
# polls AND low memory on both devices before touching the GPUs.
stable=0
deadline=$(( $(date +%s) + 5400 ))
while :; do
  procs=$(pgrep -c -f 'eval_smolvla_success.py' 2>/dev/null || true)
  procs=${procs:-0}
  mem0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
  mem1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
  echo "$(date -u '+%H:%M:%SZ') eval_procs=$procs mem=${mem0}/${mem1}MiB quiet_streak=$stable"
  if [ "$procs" -eq 0 ] && [ "$mem0" -lt 1500 ] && [ "$mem1" -lt 1500 ]; then
    stable=$(( stable + 1 ))
    [ "$stable" -ge 3 ] && { echo GPUS_FREE; break; }
  else
    stable=0
  fi
  [ "$(date +%s)" -ge "$deadline" ] && fail "GPUs still busy after 90 min"
  sleep 30
done
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader

echo
echo '=== 4. dry-run the REAL config on both GPUs ==='
# probe51 validated this file with the dataset root and device patched out. This
# runs it unpatched, which is the only check that the r5f dataset satisfies
# build_train_config's 28D/4D and state/action name-order contract.
"$PY" scripts/train_smolvla.py --config "$YAML" --dry-run > /tmp/r5f_dryrun_real.json 2>/tmp/r5f_dryrun_real.err
rc=$?
echo "DRYRUN_RC=$rc"
[ -s /tmp/r5f_dryrun_real.err ] && { echo '--- stderr ---'; cat /tmp/r5f_dryrun_real.err; }
[ "$rc" -eq 0 ] || fail 'real dry-run rejected the config'
"$PY" - /tmp/r5f_dryrun_real.json "$DS" "$ROOT" <<'PY'
import json
import sys

d = json.load(open(sys.argv[1]))
ds_want, root = sys.argv[2], sys.argv[3]
launch, train, pol = d['launcher'], d['training'], d['training']['policy']
bad = []


def want(name, got, exp):
    if got != exp:
        bad.append(f'{name}: got {got!r}, want {exp!r}')
    print(f'  {name:26s} = {got!r}')


want('dataset.root', train['dataset']['root'], ds_want)
want('launcher.gpu_ids', launch['gpu_ids'], [0, 1])
want('launcher.mixed_precision', launch['mixed_precision'], 'bf16')
want('policy.device', pol['device'], 'cuda')
want('steps', train['steps'], 4000)
want('batch_size', train['batch_size'], 16)
want('seed', train['seed'], 2000)
want('save_freq', train['save_freq'], 1000)
want('eval_freq', train['eval_freq'], 0)
want('policy.chunk_size', pol['chunk_size'], 10)
want('policy.n_action_steps', pol['n_action_steps'], 3)
want('policy.num_steps', pol['num_steps'], 10)
want('policy.optimizer_lr', pol['optimizer_lr'], 2e-5)
want('policy.scheduler_warmup_steps', pol['scheduler_warmup_steps'], 100)
want('policy.scheduler_decay_steps', pol['scheduler_decay_steps'], 4000)
want('state shape', pol['input_features']['observation.state']['shape'], [28])
want('output_features', pol['output_features'], {'action': {'type': 'ACTION', 'shape': [4]}})
print('  output_dir                =', train['output_dir'])
print('  pretrained_path           =', pol['pretrained_path'])
if not str(train['output_dir']).startswith(root + '/outputs/train/excavator_smolvla_r5f_bon_'):
    bad.append(f'output_dir {train["output_dir"]} is not under the expected job prefix')
# NOTE: this output_dir is NOT the one training will use. load_settings stamps it
# with datetime.now() and os.getpid() at call time, so the real run dir is only
# knowable after launch. Only the job_name prefix is a stable expectation.
if bad:
    for b in bad:
        print('MISMATCH:', b)
    raise SystemExit(1)
print('DRYRUN_MATCHES_INTENT')
PY
[ $? -eq 0 ] || fail 'resolved config does not match intent'

echo
echo '=== 5. launch training (detached; self-re-execs into accelerate on 2 GPUs) ==='
PREFIX="$ROOT/outputs/train/excavator_smolvla_r5f_bon_"
before=$(ls -1d "$PREFIX"* 2>/dev/null | sort)
setsid nohup "$PY" scripts/train_smolvla.py --config "$YAML" > "$STDOUT_LOG" 2>&1 &
LAUNCH_PID=$!
disown "$LAUNCH_PID" 2>/dev/null || true
echo "launcher pid = $LAUNCH_PID, stdout = $STDOUT_LOG"
echo "$LAUNCH_PID" > /tmp/r5f_train.pid

echo
echo '=== 6. confirm it survived startup ==='
# Multi-GPU means the parent immediately re-execs into accelerate.commands.launch,
# so the pid recorded above is not the pid that trains. Find the run directory it
# actually created and watch that.
RUN=''
for i in $(seq 1 24); do
  sleep 10
  # comm against the pre-launch listing, with empty lines stripped: `grep -vxF ""`
  # would match everything, so an empty `before` has to be filtered out first.
  RUN=$(comm -13 <(printf '%s\n' "$before" | sed '/^$/d' | sort) <(ls -1d "$PREFIX"* 2>/dev/null | sort) | tail -1)
  if [ -n "$RUN" ] && [ -f "$RUN/resolved_training_config.json" ]; then
    echo "run dir appeared after ~$((i * 10)) s: $RUN"
    break
  fi
  echo "  ${i}: waiting for run dir ($(tail -n 1 "$STDOUT_LOG" 2>/dev/null))"
done
[ -n "$RUN" ] || { echo '--- stdout ---'; cat "$STDOUT_LOG"; fail 'no run directory appeared within 240 s'; }
echo "$RUN" > /tmp/r5f_run_dir.txt
ls -la "$RUN"
echo '--- resolved config actually used ---'
"$PY" - "$RUN/resolved_training_config.json" "$DS" <<'PY'
import json
import sys

d = json.load(open(sys.argv[1]))
ds_want = sys.argv[2]
# resolved_training_config.json is NESTED ({'launcher':..,'training':{...}}),
# same shape the section-4 dry-run JSON has. The flat-key version of this check
# read all-None and aborted a perfectly healthy run (13:59Z incident).
t = d['training']
pol = t['policy']
print(f'  steps={t.get("steps")} batch_size={t.get("batch_size")} seed={t.get("seed")} '
      f'save_freq={t.get("save_freq")} output_dir={t.get("output_dir")}')
print(f'  dataset.root={t.get("dataset", {}).get("root")}')
print(f'  pretrained_path={pol.get("pretrained_path")}')
print(f'  lr={pol.get("optimizer_lr")} warmup={pol.get("scheduler_warmup_steps")} '
      f'decay={pol.get("scheduler_decay_steps")}')
assert t.get('dataset', {}).get('root') == ds_want, 'run is NOT training on the r5f dataset'
print('RUN_CONFIG_CONFIRMED')
PY
[ $? -eq 0 ] || fail 'the live run is not configured as intended'

echo '--- stdout tail ---'
tail -n 25 "$STDOUT_LOG"
# Only a Python traceback is fatal here. A looser regex would abort on benign
# transformer warnings and orphan a run that is training perfectly well.
if grep -q 'Traceback (most recent call last)' "$STDOUT_LOG"; then
  grep -n -A 20 'Traceback (most recent call last)' "$STDOUT_LOG" | head -40
  fail 'training raised during startup'
fi
echo '--- non-fatal error/warning lines (informational) ---'
grep -inE 'error|warn' "$STDOUT_LOG" | head -15
echo '--- processes ---'
pgrep -af 'train_smolvla.py|accelerate.commands.launch' | head -10
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
echo STARTUP_OK

echo
echo '=== 7. wait for training to finish ==='
# 4000 steps at the measured 0.259 s/step is ~17 min; allow 60 min. "End of
# training" in train.log is the authoritative signal; the pgrep check is the
# fallback for a crash that never wrote it.
deadline=$(( $(date +%s) + 3600 ))
while :; do
  if grep -q 'End of training' "$RUN/train.log" 2>/dev/null; then
    echo "$(date -u '+%H:%M:%SZ') train.log reports End of training"
    break
  fi
  if ! pgrep -f 'accelerate.commands.launch' >/dev/null 2>&1 && ! pgrep -f 'train_smolvla.py' >/dev/null 2>&1; then
    echo "$(date -u '+%H:%M:%SZ') no training process left (crash or completion without the banner)"
    break
  fi
  echo "$(date -u '+%H:%M:%SZ') $(tail -n 1 "$RUN/train.log" 2>/dev/null || tail -n 1 "$STDOUT_LOG")"
  [ "$(date +%s)" -ge "$deadline" ] && { echo 'TRAIN_TIMEOUT_60MIN'; break; }
  sleep 60
done

echo
echo '=== 8. result ==='
RUN=$(cat /tmp/r5f_run_dir.txt)
echo "run dir = $RUN"
ls -la "$RUN"
echo '--- checkpoints ---'
ls -la "$RUN/checkpoints/" 2>/dev/null
readlink -f "$RUN/checkpoints/last" 2>/dev/null
for c in "$RUN"/checkpoints/*/pretrained_model; do
  [ -d "$c" ] || continue
  echo "  $(basename "$(dirname "$c")"): $(du -sh "$c" | cut -f1), model.safetensors=$(stat -c %s "$c/model.safetensors" 2>/dev/null || echo MISSING)"
done
echo '--- loss curve (every 5th logged step) ---'
grep -oE 'step:[^ ]+ .*' "$RUN/train.log" 2>/dev/null | awk 'NR % 5 == 1' | head -40
echo '--- final logged step ---'
grep -oE 'step:[^ ]+ .*' "$RUN/train.log" 2>/dev/null | tail -1
echo '--- checkpoint events ---'
grep -i 'Checkpoint policy\|End of training' "$RUN/train.log" 2>/dev/null
echo '--- supervised_action_fraction present in train.log? ---'
grep -c 'supervised_action_fraction' "$RUN/train.log" 2>/dev/null || echo 0
echo '  (expected 0: the fixed INFO line format does not carry extra forward metrics,'
echo '   and under --weight-mode uniform every weight is 1.0 anyway)'
echo '--- tracebacks in stdout? ---'
grep -c 'Traceback (most recent call last)' "$STDOUT_LOG" || echo 0
echo '--- error/warning line count in stdout ---'
grep -icE 'error|warn' "$STDOUT_LOG" || echo 0
echo '--- last 15 stdout lines ---'
tail -n 15 "$STDOUT_LOG"
echo
echo '=== 9. done; r5g NOT launched ==='
echo "checkpoints under $RUN/checkpoints"
df -h "$ROOT" | tail -1
date -u '+finished %Y-%m-%dT%H:%M:%SZ'
