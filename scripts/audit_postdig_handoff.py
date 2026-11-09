"""Separate closure/lift overrides from subsequent autonomous loaded transport."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np


def audit(folder):
    from excavator_reachability import TruckBedColumn
    scene = json.loads((folder / 'scene_context.json').read_text())
    bed = TruckBedColumn(scene, 0.)
    rows = [json.loads(line) for line in (folder / 'steps.jsonl').read_text().splitlines()]
    handbacks = [(i, r['carry_guard'].get('stall_recovery') or {}) for i, r in enumerate(rows)
                 if (r['carry_guard'].get('stall_recovery') or {}).get('handback')]
    if not handbacks:
        return dict(available=False, reason='No recovery handback')
    index, details = handbacks[0]
    start = index + 1
    if start >= len(rows):
        return dict(available=False, reason='No post-handback policy observations')
    samples = []
    for i in range(start, len(rows)):
        row = rows[i]
        trace = row['carry_guard'].get('stall_recovery') or {}
        if trace.get('override'):
            raise ValueError('Recovery overrides continued after handback')
        point = np.asarray(row['input_bucket_landmarks_world']['load'])
        local = bed.local(point)
        excess = np.maximum(np.maximum(bed.minimum[:2] - local[:2], local[:2] - bed.maximum[:2]), 0.)
        particles = rows[i - 1]['particle_transfer_metrics']
        samples.append(dict(step=i, bed_gap_m=float(np.linalg.norm(excess * bed.scales[:2])),
            retained_particles=particles['retained_particles'], payload_particles=particles['payload_particles'],
            load_world=point.tolist()))
    loaded = [s for s in samples if s['retained_particles'] >= 50]
    window = samples[:81]
    empty_before_transport = any(all(s['retained_particles'] <= 10 and s['payload_particles'] < 100
        and s['bed_gap_m'] > .5 for s in window[i:i + 3]) for i in range(max(0, len(window) - 2)))
    minimum = min((s['bed_gap_m'] for s in loaded), default=samples[0]['bed_gap_m'])
    window_minimum = min((s['bed_gap_m'] for s in window if s['retained_particles'] >= 50),
                         default=samples[0]['bed_gap_m'])
    checkpoint = samples[min(80, len(samples) - 1)]
    material_threshold = max(10., min(50., samples[0]['retained_particles'] * .5))
    material_minimum = min((s['bed_gap_m'] for s in samples if s['retained_particles'] >= material_threshold),
                           default=samples[0]['bed_gap_m'])
    improved = (details.get('phase') == 'completed_secure_lift' and len(samples) >= 81
                and samples[0]['bed_gap_m'] - window_minimum >= 3.
                and checkpoint['retained_particles'] >= 50 and not empty_before_transport)
    return dict(available=True, handback_step=index, handback_phase=details.get('phase'),
        post_handback_action_steps=len(samples), no_recovery_override_after_handback=True,
        start=samples[0], at_eight_seconds=checkpoint, terminal=samples[-1],
        loaded_bed_gap_reduction_m=samples[0]['bed_gap_m'] - minimum,
        eight_second_loaded_bed_gap_reduction_m=samples[0]['bed_gap_m'] - window_minimum,
        low_load_diagnostic_threshold_particles=material_threshold,
        material_bed_gap_reduction_m=samples[0]['bed_gap_m'] - material_minimum,
        maximum_new_deposit_after_handback=max(s['payload_particles'] for s in samples) - samples[0]['payload_particles'],
        pretruck_empty_within_eight_seconds=empty_before_transport,
        sustained_autonomous_loaded_transport=improved,
        scope='Stage repair only; task success remains the original scoring result')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    args = parser.parse_args()
    results = {}
    for folder in args.experiment.glob('evaluation/*/episodes/episode_0001'):
        if (folder / 'episode_status.json').exists():
            results[folder.parents[1].name] = audit(folder)
    (args.experiment / 'handoff_audit.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    for name, result in results.items():
        if result['available']:
            print(name, result['handback_phase'], round(result['loaded_bed_gap_reduction_m'], 3),
                  result['sustained_autonomous_loaded_transport'])
