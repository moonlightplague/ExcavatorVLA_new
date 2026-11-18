"""Read recorded ACT actions to distinguish loaded stalls from guard intervention."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def audit(experiment):
    records = json.loads((experiment / 'results.json').read_text())
    cases = []
    source_hashes = {}
    for key, record in sorted(records.items(), key=lambda item: int(item[0])):
        if record['status'] != 'completed' or record['transport']['available']:
            continue
        path = experiment / 'evaluation/episodes' / f'episode_{int(key):04d}/steps.jsonl'
        raw = path.read_bytes()
        source_hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        rows = [json.loads(line) for line in raw.splitlines()]
        loaded = [row for row in rows if row['particle_transfer_metrics']['retained_particles'] >= 100]
        if not loaded:
            continue
        def mean(values):
            return statistics.mean(values)
        cases.append(dict(scene=int(key), seed=record['scene_seed'],
            loaded_observations=len(loaded),
            loaded_observation_seconds=len(loaded) / 10,
            # A count across observations is not necessarily one contiguous interval.
            at_least_100_loaded_observations=len(loaded) >= 100,
            max_retained=max(row['particle_transfer_metrics']['retained_particles'] for row in rows),
            raw_parent_action_norm_mean_rad_s=mean(math.hypot(*r['predicted'][1:3]) for r in loaded),
            applied_parent_action_norm_mean_rad_s=mean(math.hypot(*r['applied'][1:3]) for r in loaded),
            raw_swing_abs_mean_rad_s=mean(abs(r['predicted'][0]) for r in loaded),
            raw_bucket_abs_mean_rad_s=mean(abs(r['predicted'][3]) for r in loaded),
            guard_modified_observations=sum(bool(r['carry_guard'].get('modified')) for r in loaded),
            guard_infeasible_observations=sum(bool(r['carry_guard'].get('infeasible')) for r in loaded)))
    return dict(scope='Read-only diagnosis of previously viewed development trajectories; original scoring unchanged.',
        limitations='Loaded frames may be noncontiguous. Particle count is not the policy fill fraction. '
                    'Small commanded actions do not identify why the model chose them; no reachability conclusion.',
        loaded_without_valid_transport=len(cases),
        persistent_loaded_cases=sum(c['at_least_100_loaded_observations'] for c in cases),
        guard_modified_loaded_observations=sum(c['guard_modified_observations'] for c in cases),
        cases=cases, source_sha256=source_hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.experiment.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({k: result[k] for k in ('loaded_without_valid_transport',
                     'persistent_loaded_cases', 'guard_modified_loaded_observations')}))


if __name__ == '__main__':
    main()
