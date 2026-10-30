#!/usr/bin/env bash
# r5g: score one Best-of-N fine-tuned checkpoint on the 100 held-out seeds and
# compare it, paired seed by seed, against the 040000 baseline that is already
# on disk.
#
# Usage: bash run_r5g.sh <checkpoint-dir|auto> [output-dir]
#   <checkpoint-dir>  .../checkpoints/NNNNNN/pretrained_model
#   auto              wait for the r5f training run to finish, then use its
#                     checkpoints/last
#
# Why this is a fair comparison (probe53/54/55 established all of it):
#
# 1. The baseline arm is free. outputs/eval/success_20261020_151616_800466 ran
#    --seed 2 --episodes 500, so scene_seeds 402..501 are its last 100 episodes
#    and all 100 episode_status.json files are on disk: 83/100 succeeded. Their
#    scores are recoverable from the `rejected` list (reject_reason=excluded_seed)
#    of outputs/rl/selection_smolvla_top100/selection.json, produced by this same
#    select_rollouts.py, so both arms are scored by one identical reward function.
#
# 2. --seed 402 --episodes 100 reproduces those exact scenes. eval_smolvla_success.py
#    uses args.seed in exactly one functional place,
#        status['scene_seed'] = args.seed + index - 1
#    and then, inside every episode and before session.prepare(), re-seeds all
#    three RNG streams from that scene_seed:
#        torch.manual_seed(scene_seed); random.seed(scene_seed)
#        np.random.seed(scene_seed % 2**32)
#    So the per-episode RNG state depends on scene_seed alone, not on where in the
#    1..N range the episode sat. The flow-matching noise that SmolVLA draws during
#    predict_action_chunk comes from that same global torch RNG, so it is replayed
#    too. Every other RNG in the submodule is a fixed literal (4107, 4108,
#    410700+attempt*7919) or derived from wall-clock inside the planner, neither
#    of which args.seed reaches.
#
# 3. Every other flag is copied verbatim from the baseline config.json, including
#    --workers 0 --gpus 0,1 (resolved_workers=2, one process, two Isaac sims --
#    the same shape as the baseline) and --cpu-threads 0 (resolved to 14).
#
# The one thing that cannot be verified from artifacts: config.json and
# provenance.json carry no soil-backend key, so the baseline's PhysX sand can only
# be inferred from its 10-20 date predating submodule a034789's newton_mpm default.
# EXCAVATOR_SOIL_BACKEND=physx below is the whole guarantee.
#
# Cost: runtime.json for the baseline records 43962.6 s for 500 episodes on two
# workers = 87.9 s/episode, so 100 episodes is ~2.4 h. There is no resume: the
# episode loop does folder.mkdir(parents=True) without exist_ok, so a re-run into
# the same output dir dies immediately. Point it at a fresh dir, or at one that
# already holds success_rate_summary.json, which this script treats as "eval done,
# just compare".
set -uo pipefail

ROOT=/root/gpufree-data/ExcavatorVLA_new
VLM=$ROOT/ExcavatorVLA-vlm
PY=/root/gpufree-data/conda/envs/ExcavatorVLA-current/bin/python
REF=/root/gpufree-data/data/export
BASELINE_RUN=$ROOT/outputs/eval/success_20261020_151616_800466
BASELINE_SEL=$ROOT/outputs/rl/selection_smolvla_top100/selection.json
TRAIN_RUN_FILE=/tmp/r5f_run_dir.txt
SEED0=402
EPISODES=100

# Trajectory capture is OFF by default. The baseline has no save_trajectory key at
# all (the flag postdates it), and holdout rollouts must never become training
# data, so the NPZs would have no reuse value; failure analysis is served by the
# evaluation.mp4 that --video already writes on both arms. Set to 1 only if the
# per-step frames are wanted anyway -- it costs ~7.9 G and does not perturb the
# score, since the store is written after the action is applied and, per the
# comment at eval_smolvla_success.py:438, after the latency sample.
SAVE_TRAJ=0

CKPT_ARG=${1:-auto}
fail() { echo "ABORT: $*" >&2; exit 1; }

for required in "$PY" "$REF/meta/info.json" "$BASELINE_RUN/config.json" "$BASELINE_SEL"; do
  [ -e "$required" ] || fail "MISSING: $required"
done

# --- 0. resolve the checkpoint ------------------------------------------------
if [ "$CKPT_ARG" = auto ]; then
  echo '=== 0. waiting for the r5f training run to finish ==='
  [ -f "$TRAIN_RUN_FILE" ] || fail "no $TRAIN_RUN_FILE; pass a checkpoint dir instead of auto"
  RUN=$(cat "$TRAIN_RUN_FILE")
  [ -d "$RUN" ] || fail "recorded run dir is gone: $RUN"
  deadline=$(( $(date +%s) + 7200 ))
  while ! grep -q 'End of training' "$RUN/train.log" 2>/dev/null; do
    if ! pgrep -f 'accelerate.commands.launch|train_smolvla.py' >/dev/null 2>&1; then
      fail "training processes are gone but train.log never reported End of training"
    fi
    echo "$(date -u '+%H:%M:%SZ') $(tail -n 1 "$RUN/train.log" 2>/dev/null)"
    [ "$(date +%s)" -ge "$deadline" ] && fail 'training still going after 120 min'
    sleep 60
  done
  CKPT=$RUN/checkpoints/last/pretrained_model
  echo "training finished; using $CKPT"
  echo '--- loss curve ---'
  grep -oE 'step:[^ ]+ .*' "$RUN/train.log" | awk 'NR % 10 == 1' | head -30
  grep -oE 'step:[^ ]+ .*' "$RUN/train.log" | tail -1
  echo '--- available checkpoints ---'
  ls -1 "$RUN/checkpoints/"
else
  CKPT=$(readlink -f "$CKPT_ARG")
fi
[ -f "$CKPT/config.json" ] || fail "not a checkpoint dir (no config.json): $CKPT"
[ -f "$CKPT/model.safetensors" ] || fail "no model.safetensors in $CKPT"
CKPT_NAME=$(basename "$(dirname "$CKPT")")
[ "$CKPT_NAME" = pretrained_model ] && CKPT_NAME=$(basename "$(dirname "$(dirname "$CKPT")")")
# `last` alone says nothing about which run it came from, and two aborted runs
# would silently collide in one output dir (which then trips the no-resume guard).
[ "$CKPT_NAME" = last ] && [ -n "${RUN:-}" ] && CKPT_NAME="last_${RUN##*_bon_}"
OUT=${2:-$ROOT/outputs/eval/r5g_$CKPT_NAME}
ARM_SEL=$ROOT/outputs/rl/r5g_$CKPT_NAME/selection.json
REPORT=$ROOT/outputs/rl/r5g_$CKPT_NAME/comparison.json
mkdir -p "$(dirname "$ARM_SEL")"

echo
echo "=== r5g ==="
echo "  checkpoint : $CKPT"
echo "  output     : $OUT"
echo "  seeds      : $SEED0..$(( SEED0 + EPISODES - 1 )) ($EPISODES episodes)"
echo "  baseline   : $BASELINE_RUN (already on disk, not re-run)"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
df -h "$ROOT" | tail -1

# Refuse to contaminate the comparison with a checkpoint trained on holdout data.
"$PY" - "$CKPT" "$REF" <<'PY' || fail 'checkpoint preflight failed'
import json, sys
from pathlib import Path
ckpt, ref = Path(sys.argv[1]), Path(sys.argv[2])
cfg = json.loads((ckpt / 'config.json').read_text())
print(f'  policy class = {cfg.get("type")}, chunk_size = {cfg.get("chunk_size")}, '
      f'n_action_steps = {cfg.get("n_action_steps")}')
tc = ckpt / 'train_config.json'
if tc.exists():
    t = json.loads(tc.read_text())
    ds = t.get('dataset', {}).get('root')
    print(f'  trained on   = {ds}')
    print(f'  steps        = {t.get("steps")}, pretrained_path = '
          f'{t.get("policy", {}).get("pretrained_path")}')
    if ds and Path(ds).resolve() == Path(ref).resolve():
        print('  NOTE: this checkpoint was trained on the reference set, not an r5f dataset')
pre = ckpt / 'policy_preprocessor_step_5_normalizer_processor.safetensors'
post = ckpt / 'policy_postprocessor_step_0_unnormalizer_processor.safetensors'
for p in (pre, post):
    print(f'  {"OK " if p.exists() else "MISSING"} {p.name} ({p.stat().st_size if p.exists() else 0} B)')
    if not p.exists():
        raise SystemExit(1)
print('CKPT_PREFLIGHT_OK')
PY

# --- 1. evaluate (skipped when a finished run is already there) ---------------
export HF_HOME="$ROOT/models/hf-cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export EXCAVATOR_SOIL_BACKEND=physx
export OMNI_KIT_ALLOW_ROOT=1
export EXCAVATOR_ISAAC_DIR=/root/gpufree-data/isaacsim
export PYTHONUNBUFFERED=1

if [ -f "$OUT/success_rate_summary.json" ]; then
  echo
  echo "=== 1. $OUT already holds a summary; skipping the ~2.4 h evaluation ==="
else
  echo
  echo "=== 1. evaluating $(date -u +%FT%TZ) (expect ~2.4 h) ==="
  # Both GPUs must be idle: the baseline shape is two Isaac sims in one process,
  # and a stray third sim would change nothing about the score but would slow the
  # wall clock enough to make the run look hung.
  # In `auto` mode this runs seconds after train.log printed End of training, when
  # accelerate is still tearing down (rank processes linger, VRAM is released last),
  # so a single instant check would abort a run that is 10 s from being ready. Wait
  # for three consecutive quiet polls with both devices under 1.5 G instead.
  idle_deadline=$(( $(date +%s) + 1800 ))
  idle_streak=0
  while :; do
    busy=$(pgrep -c -f 'eval_smolvla_success.py|accelerate.commands.launch|train_smolvla.py' 2>/dev/null || true)
    busy=${busy:-0}
    mem0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    mem1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
    echo "$(date -u '+%H:%M:%SZ') busy_procs=$busy mem=${mem0}/${mem1}MiB quiet_streak=$idle_streak"
    if [ "$busy" -eq 0 ] && [ "$mem0" -lt 1500 ] && [ "$mem1" -lt 1500 ]; then
      idle_streak=$(( idle_streak + 1 ))
      [ "$idle_streak" -ge 3 ] && { echo GPUS_FREE; break; }
    else
      idle_streak=0
    fi
    [ "$(date +%s)" -ge "$idle_deadline" ] && \
      fail "GPUs still busy after 30 min (busy=$busy mem=${mem0}/${mem1}MiB); wait for them manually"
    sleep 30
  done
  [ -e "$OUT" ] && fail "$OUT exists but has no summary; move it aside (there is no resume)"
  mkdir -p "$OUT"
  cd "$ROOT" || fail "cannot cd $ROOT"
  extra=()
  [ "$SAVE_TRAJ" = 1 ] && extra+=(--save-trajectory)
  t0=$(date +%s)
  "$PY" scripts/eval_smolvla_success.py \
    --checkpoint "$CKPT" \
    --policy-type smolvla \
    --dataset "$REF" \
    --output "$OUT" \
    --episodes "$EPISODES" \
    --steps 400 \
    --min-particles 100 \
    --seed "$SEED0" \
    --replan-interval 3 \
    --initial-episode 0 \
    --workers 0 \
    --gpus 0,1 \
    --headless \
    --video \
    ${extra[@]+"${extra[@]}"} \
    --fresh-simulator \
    --render-every-physics-step \
    --cpu-threads 0 \
    --startup-timeout 600 \
    > "$OUT/../r5g_$CKPT_NAME.log" 2>&1
  rc=$?
  t1=$(date +%s)
  echo "eval exit=$rc wall=$(( t1 - t0 ))s ($(( (t1 - t0) / 60 )) min)"
  tail -n 25 "$OUT/../r5g_$CKPT_NAME.log"
  [ "$rc" -eq 0 ] || fail "evaluation returned $rc; see $OUT/../r5g_$CKPT_NAME.log"
  [ -f "$OUT/success_rate_summary.json" ] || fail 'no summary written despite exit 0'
fi

# --- 2. gate: the run must match the baseline protocol ------------------------
echo
echo '=== 2. protocol gate (every flag must equal the baseline config.json) ==='
"$PY" - "$OUT/config.json" "$BASELINE_RUN/config.json" "$CKPT" <<'PY' || fail 'protocol gate failed'
import json, sys
new = json.load(open(sys.argv[1]))
old = json.load(open(sys.argv[2]))
ckpt = sys.argv[3]
# Keys whose whole purpose is to differ, plus bookkeeping the harness stamps itself.
ALLOWED = {'checkpoint', 'checkpoint_sha256', 'output', 'evaluation_id', 'seed', 'episodes',
           'resolved_workers', 'resolved_gpu_ids', 'resolved_cpu_threads_per_simulator',
           'initial_episode', 'metric_definitions', 'save_trajectory'}
bad = []
for key in sorted(set(old) | set(new)):
    if key in ALLOWED:
        continue
    if old.get(key) != new.get(key):
        bad.append(f'{key}: baseline={old.get(key)!r} r5g={new.get(key)!r}')
print(f'  compared {len(set(old) | set(new)) - len(ALLOWED)} shared protocol keys')
print(f'  seed {old["seed"]}..{old["seed"] + old["episodes"] - 1} -> '
      f'{new["seed"]}..{new["seed"] + new["episodes"] - 1}')
print(f'  workers {old["workers"]}->{old["resolved_workers"]} vs '
      f'{new["workers"]}->{new["resolved_workers"]}')
print(f'  cpu_threads {old["cpu_threads"]}->{old["resolved_cpu_threads_per_simulator"]} vs '
      f'{new["cpu_threads"]}->{new["resolved_cpu_threads_per_simulator"]}')
print(f'  checkpoint_sha256 baseline={old["checkpoint_sha256"][:16]} r5g={new["checkpoint_sha256"][:16]}')
if old['checkpoint_sha256'] == new['checkpoint_sha256']:
    bad.append('r5g evaluated the SAME checkpoint as the baseline; the comparison would be vacuous')
for line in bad:
    print('MISMATCH:', line)
if bad:
    raise SystemExit(1)
print('PROTOCOL_MATCHES_BASELINE')
PY

echo
echo '=== 3. score the r5g arm with the same select_rollouts.py ==='
cd "$VLM" || fail "cannot cd $VLM"
# --mode success keeps every successful rollout and records the rest in `rejected`
# with their full reward dict, so selected+rejected covers all 100 seeds with
# scores. No threshold: the point is to measure the arm, not to filter it.
"$PY" -m excavator_vlm.rl.select_rollouts \
  --runs "$OUT" \
  --output "$ARM_SEL" \
  --mode success \
  --policy-type smolvla \
  --print-limit 100
[ -f "$ARM_SEL" ] || fail 'select_rollouts wrote nothing'

# --- 4. paired comparison ----------------------------------------------------
echo
echo '=== 4. paired comparison against the baseline arm ==='
"$PY" - "$ARM_SEL" "$BASELINE_SEL" "$REPORT" "$SEED0" "$EPISODES" "$CKPT" <<'PY' || fail 'comparison failed'
import json, math, statistics as st, sys
from pathlib import Path

try:
    from scipy import stats as sps
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

arm_path, base_path, report_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
seed0, n_epi, ckpt = int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
holdout = set(range(seed0, seed0 + n_epi))
FIELDS = ('score', 'payload_particles', 'scooped_particles', 'spillage_particles',
          'spillage_fraction', 'max_bucket_particles')


def sign_test_p(diffs):
    """Exact two-sided binomial sign test, stdlib only (McNemar-style tail)."""
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_p(diffs):
    """Wilcoxon signed-rank when scipy exists, else sign-test fallback."""
    if all(d == 0 for d in diffs):
        return 1.0, 'none(all-zero)'
    if HAS_SCIPY:
        try:
            return float(sps.wilcoxon(diffs, alternative='two-sided').pvalue), 'wilcoxon'
        except ValueError:
            pass
    return sign_test_p(diffs), 'sign'


def boot_median_ci(diffs, B=10000, seed=4107):
    """Percentile bootstrap CI for the median paired difference, stdlib only."""
    import random
    rng = random.Random(seed)
    n = len(diffs)
    meds = sorted(st.median([diffs[rng.randrange(n)] for _ in range(n)]) for _ in range(B))
    return meds[int(0.025 * B)], meds[int(0.975 * B) - 1]


def wilson_ci(k, n, z=1.96):
    d = 1 + z * z / n
    c = (k + z * z / 2) / d
    h = z * math.sqrt(k * (n - k) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def collect(sel, seeds):
    """seed -> reward dict, from both the selected and the rejected lists."""
    out = {}
    for entry in list(sel.get('selected', [])) + list(sel.get('rejected', [])):
        seed = entry.get('scene_seed')
        if seed not in seeds:
            continue
        reward = entry.get('reward') or {}
        row = {'success': bool(reward.get('success')),
               'failure_mode': reward.get('failure_mode') or entry.get('failure_mode'),
               'status': reward.get('episode_status') or entry.get('status')}
        for f in FIELDS:
            # spillage_fraction lives on the entry, not inside reward (probe56:
            # all 100 baseline reward dicts lack it), so fall back per field.
            v = reward.get(f)
            row[f] = entry.get(f) if v is None else v
        if seed in out:
            raise SystemExit(f'seed {seed} appears twice in {sel["schema_version"]}')
        out[seed] = row
    return out


arm, base = collect(json.load(open(arm_path)), holdout), collect(json.load(open(base_path)), holdout)
missing_arm, missing_base = sorted(holdout - set(arm)), sorted(holdout - set(base))
print(f'  r5g arm  : {len(arm)}/{n_epi} holdout seeds scored'
      + (f'  MISSING {missing_arm}' if missing_arm else ''))
print(f'  baseline : {len(base)}/{n_epi} holdout seeds scored'
      + (f'  MISSING {missing_base}' if missing_base else ''))
if missing_arm or missing_base:
    raise SystemExit('cannot pair: one arm is incomplete')

both = sorted(holdout)
a_ok = [s for s in both if arm[s]['success']]
b_ok = [s for s in both if base[s]['success']]
print()
print(f'  success rate   baseline {len(b_ok)}/{n_epi} = {100 * len(b_ok) / n_epi:.1f}%')
print(f'                 r5g      {len(a_ok)}/{n_epi} = {100 * len(a_ok) / n_epi:.1f}%')
print(f'                 delta    {len(a_ok) - len(b_ok):+d} episodes '
      f'({100 * (len(a_ok) - len(b_ok)) / n_epi:+.1f} pp)')
b_lo, b_hi = wilson_ci(len(b_ok), n_epi)
a_lo, a_hi = wilson_ci(len(a_ok), n_epi)
print(f'                 Wilson 95% CI  baseline [{100 * b_lo:.1f}, {100 * b_hi:.1f}]%'
      f'  r5g [{100 * a_lo:.1f}, {100 * a_hi:.1f}]%')

# McNemar: only the discordant pairs carry information about a change.
gain = sorted(s for s in both if arm[s]['success'] and not base[s]['success'])
loss = sorted(s for s in both if base[s]['success'] and not arm[s]['success'])
nd = len(gain) + len(loss)
print()
print(f'  discordant     baseline-fail -> r5g-success : {len(gain)}  {gain}')
print(f'                 baseline-success -> r5g-fail : {len(loss)}  {loss}')
print(f'                 concordant (both same)       : {n_epi - nd}')
if nd == 0:
    print('  McNemar        exact p = 1.000 (no discordant pair)')
    mcnemar_p = 1.0
else:
    k = min(len(gain), len(loss))
    tail = sum(math.comb(nd, i) for i in range(k + 1)) / 2 ** nd
    mcnemar_p = min(1.0, 2 * tail)
    print(f'  McNemar        exact two-sided p = {mcnemar_p:.4f} on b={len(gain)}, c={len(loss)}')
    print(f'                 {"significant at 0.05" if mcnemar_p < 0.05 else "NOT significant at 0.05"}')

# Paired score deltas. PhysX settle noise alone moves a single seed's score by
# up to ~900 points (probe47: seed 239 scored 897/461/-1 across three identical
# runs), so the median and the sign test say more here than the mean.
deltas = [(s, arm[s]['score'] - base[s]['score']) for s in both
          if arm[s]['score'] is not None and base[s]['score'] is not None]
if deltas:
    vals = [d for _, d in deltas]
    pos = sum(1 for v in vals if v > 0)
    neg = sum(1 for v in vals if v < 0)
    tie = len(vals) - pos - neg
    print()
    print(f'  score delta    n={len(vals)} mean={st.mean(vals):+.1f} median={st.median(vals):+.1f}')
    q = st.quantiles(vals, n=4) if len(vals) >= 4 else [min(vals), st.median(vals), max(vals)]
    print(f'                 q1={q[0]:+.0f} q3={q[2]:+.0f} min={min(vals):+d} max={max(vals):+d}')
    print(f'                 improved {pos} / worsened {neg} / unchanged {tie}')
    top = sorted(deltas, key=lambda t: t[1])
    print(f'                 worst  5: {[(s, d) for s, d in top[:5]]}')
    print(f'                 best   5: {[(s, d) for s, d in top[-5:]]}')
    base_scores = [base[s]['score'] for s in both if base[s]['score'] is not None]
    arm_scores = [arm[s]['score'] for s in both if arm[s]['score'] is not None]
    print(f'  score level    baseline mean={st.mean(base_scores):.1f} median={st.median(base_scores):.1f}')
    print(f'                 r5g      mean={st.mean(arm_scores):.1f} median={st.median(arm_scores):.1f}')

# Continuous paired stats. probe58 power analysis: with n=100 the binary McNemar
# cannot resolve effects below ~+7pp (single-arm 95% CI is +/-7.4pp), but the
# paired continuous tests use all 100 diffs, not just the discordant pairs --
# and score is significantly decoupled from success (probe57).
cont = {}
print()
print('  continuous paired diffs (median [bootstrap 95% CI], two-sided p)')
for f in FIELDS:
    d = [arm[s][f] - base[s][f] for s in both
         if isinstance(arm[s][f], (int, float)) and isinstance(base[s][f], (int, float))]
    if len(d) < 10:
        print(f'    {f:20s} n={len(d):3d}  too few numeric pairs, skipped')
        continue
    p, method = paired_p(d)
    lo, hi = boot_median_ci(d)
    med = st.median(d)
    cont[f] = {'n': len(d), 'median_diff': med, 'mean_diff': st.mean(d),
               'boot_ci95': [lo, hi], 'p': p, 'test': method}
    star = ' *' if p < 0.05 else ''
    print(f'    {f:20s} n={len(d):3d}  median {med:+.3f} [{lo:+.3f}, {hi:+.3f}]'
          f'  p={p:.4f} ({method}){star}')
if not HAS_SCIPY:
    print('    (scipy unavailable: exact sign test used instead of Wilcoxon)')

print()
print('  failure modes  baseline -> r5g')
modes = sorted({arm[s]['failure_mode'] for s in both} | {base[s]['failure_mode'] for s in both})
for m in modes:
    nb = sum(1 for s in both if base[s]['failure_mode'] == m)
    na = sum(1 for s in both if arm[s]['failure_mode'] == m)
    print(f'    {str(m):24s} {nb:3d} -> {na:3d}  ({na - nb:+d})')

print()
print('  per-seed table (seed: baseline_success/score -> r5g_success/score)')
for i, s in enumerate(both):
    mark = '  ' if arm[s]['success'] == base[s]['success'] else ('++' if arm[s]['success'] else '--')
    print(f'  {mark} {s}: {"Y" if base[s]["success"] else "n"}/{base[s]["score"]} '
          f'-> {"Y" if arm[s]["success"] else "n"}/{arm[s]["score"]} '
          f'[{arm[s]["failure_mode"]}]')
    if i == 9 and len(both) > 20:
        print(f'  ... ({len(both) - 10} more rows in {report_path})')
        break

report_path.write_text(json.dumps({
    'schema_version': 'excavator_r5g_comparison_v2',
    'checkpoint': ckpt,
    'arm_run': str(arm_path.parent),
    'baseline_run': json.load(open(base_path))['criteria']['runs'][0],
    'seeds': [seed0, seed0 + n_epi - 1],
    'n_episodes': n_epi,
    'scipy_available': HAS_SCIPY,
    'success': {'baseline': len(b_ok), 'r5g': len(a_ok), 'delta': len(a_ok) - len(b_ok),
                'wilson_ci95': {'baseline': [b_lo, b_hi], 'r5g': [a_lo, a_hi]}},
    'mcnemar': {'gained': gain, 'lost': loss, 'n_discordant': nd, 'exact_two_sided_p': mcnemar_p},
    'continuous': cont,
    'score_delta': ({'mean': st.mean(vals), 'median': st.median(vals), 'min': min(vals),
                     'max': max(vals), 'improved': pos, 'worsened': neg, 'unchanged': tie}
                    if deltas else None),
    'per_seed': {str(s): {'baseline': base[s], 'r5g': arm[s]} for s in both},
}, indent=2, sort_keys=True))
print()
print(f'  wrote {report_path}')
PY

echo
du -sh "$OUT" 2>/dev/null
df -h "$ROOT" | tail -1
echo "R5G_DONE $(date -u +%FT%TZ)"
