"""Read-only audit of the ten post-dig stalls identified by the user."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from act_stall_recovery import PostDigStallDetector, settings_dict, mode_settings

ROOT = Path(__file__).resolve().parents[1]
REVIEWED = [1, 2, 4, 9, 10, 15, 18, 24, 29, 31]


def audit(experiment, mode='secure_lift'):
    results = json.loads((experiment / 'results.json').read_text())
    records, protected = [], {}
    for key, result in sorted(results.items(), key=lambda x: int(x[0])):
        path = experiment / 'evaluation/episodes' / f'episode_{int(key):04d}/steps.jsonl'
        raw = path.read_bytes()
        rows = [json.loads(line) for line in raw.splitlines()]
        protected[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
        detector = PostDigStallDetector(mode_settings(mode))
        first = None
        for row in rows:
            trigger, details = detector.observe(row['input_state'], row['predicted'])
            if trigger:
                first = dict(step=row['step'], seconds=row['step'] / 10, details=details)
                break
        stable = rows[100:400]
        actions = np.asarray([r['predicted'] for r in stable])
        state = np.asarray([r['input_state'] for r in stable])
        records.append(dict(scene=int(key), user_reviewed_postdig_stall=int(key) in REVIEWED,
            original_success=result['original_task_completed'], trigger=first,
            mid_rollout_parent_action_mean_rad_s=float(np.linalg.norm(actions[:, 1:3], axis=1).mean()),
            mid_rollout_fill_median=float(np.median(state[:, 26])),
            mid_rollout_guard_modified_steps=sum(bool(r['carry_guard']['modified']) for r in stable),
            original_terminal=result['transport']['terminal']))
    return dict(source=str(experiment.resolve()), user_reviewed_scenes=REVIEWED,
        user_label='postdig_stall', detector_mode=mode, detector=settings_dict(mode), cases=records,
        reviewed_triggered=[r['scene'] for r in records if r['user_reviewed_postdig_stall'] and r['trigger']],
        successful_triggered=[r['scene'] for r in records if r['original_success'] and r['trigger']],
        other_triggered=[r['scene'] for r in records if not r['user_reviewed_postdig_stall'] and not r['original_success'] and r['trigger']],
        source_steps_sha256=protected, scope='Viewed development trajectories only; no new labels/actions or score changes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=ROOT / 'outputs/experiments/act_transport_validation40_20261104_01')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recovery-mode', choices=['secure_lift', 'secure_align', 'secure_align_scoop'], default='secure_lift')
    args = parser.parse_args()
    result = audit(args.experiment, args.recovery_mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2); f.write('\n')
    print(json.dumps({k: result[k] for k in ('user_reviewed_scenes', 'reviewed_triggered', 'successful_triggered', 'other_triggered')}))
