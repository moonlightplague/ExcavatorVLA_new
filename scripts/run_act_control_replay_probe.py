"""Paired expert-only replay of original versus reconstructed ACT controls.

This short probe validates one recorded training cycle, not model success or
full repeated-task compatibility. The source publication remains read-only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BASE / 'scripts'), str(BASE / 'ExcavatorVLA-vlm')]
from eval_smolvla_repeat40 import evaluation_environment, atomic_json, now, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--cycles', type=int, choices=[1, 10], default=1)
    parser.add_argument('--variant', choices=['both', 'original', 'reconstructed'], default='both')
    parser.add_argument('--gpu', choices=['0', '1'])
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import pyarrow.parquet as pq
    from excavator_vlm.data.relabel import relabel_episode
    from excavator_vlm.data.action_limits import BUCKET20_PROFILE, PROFILES

    source = BASE / 'data/repeat/export_repeat'
    checkpoint = BASE / 'outputs/experiments/act_shortchunk_repeat40k_20261117_01/train/checkpoints/040000/pretrained_model'
    split = json.loads((checkpoint / 'benchmark_provenance.json').read_text())['split_manifest']
    if args.episode not in split['train'] + split['validation']:
        raise ValueError('Compatibility probes must not use test parents')
    if args.variant == 'both' and args.gpu is not None:
        raise ValueError('The paired probe uses both GPUs; --gpu selects one unpaired variant')
    receipt_path = None
    for path in (BASE / 'data/repeat/raw').glob('run_*/episode_*/export.json'):
        receipt = json.loads(path.read_text())
        if (receipt.get('validated') and receipt.get('export_episode_index') == args.episode
                and Path(receipt['export_dir']).resolve() == source.resolve()):
            if receipt_path is not None:
                raise ValueError('Duplicate parent identity')
            receipt_path = path
    if receipt_path is None:
        raise ValueError('Missing original collection receipt')
    meta_path = receipt_path.with_name('meta.json')
    meta = json.loads(meta_path.read_text())
    if meta['repeat_baseline']['soil_backend'] != 'newton_mpm':
        raise ValueError('Wrong physical backend')
    frame = pq.read_table(source / f'data/chunk-{args.episode:03d}/file-000.parquet').to_pandas()
    frame = frame[frame['observation.scoop_index'] < args.cycles].sort_values('frame_index')
    states = np.stack(frame['observation.state'])
    original_actions = np.stack(frame.action)
    revised, repaired, numerical = relabel_episode(states, original_actions, PROFILES[BUCKET20_PROFILE])
    info = json.loads((source / 'meta/info.json').read_text())
    applied = meta['scene_randomization']['applied']
    goal = meta['scoop_results'][0]
    mass = meta['repeat_baseline']['soil']['mpm']['metric_point_mass_kg']
    gate_path = receipt_path.parent.parent / 'quality_gate_config.json'
    minimum_kg = json.loads(gate_path.read_text())['repeat_min_delivered_kg']
    task = str(pq.read_table(source / 'meta/tasks.parquet').to_pandas().index[0])
    common = dict(schema_version='excavator_reference_setup_v1', fps=10,
        action_units='rad/s', state_names=info['features']['observation.state']['names'],
        action_names=info['features']['action']['names'], task=task,
        initial_joints_deg=np.degrees(states[0, :4].astype(float)).tolist(),
        expected_initial_state=states[0].tolist(), dig_target_xyz=goal['target_xyz'],
        sand_amount=applied['sand_amount_multiplier'], profile=dict(
            sand_xy=applied['sand_center'][:2], truck_translation_xyz=applied['truck_translation_xyz'],
            truck_yaw_deg=applied['truck_yaw_deg'], unload_landing_xyz=goal['unload_landing_xyz']),
        provenance=dict(source_export_episode=args.episode,
            source_split='train' if args.episode in split['train'] else 'validation',
            source_meta=str(meta_path), source_meta_sha256=digest(meta_path),
            scope='Expert-only recorded-cycle control compatibility probe; not model success'))
    atomic_json(out / 'numerical_repair.json', numerical)
    np.savez_compressed(out / 'reference_controls.npz', states=states,
                        original_actions=original_actions, repaired_actions=repaired)
    atomic_json(out / 'protocol.json', dict(source=str(source.resolve()), source_episode=args.episode,
        source_info_sha256=digest(source / 'meta/info.json'), source_meta=str(meta_path),
        source_meta_sha256=digest(meta_path), steps=len(states), cycles=args.cycles, created_utc=now(),
        minimum_delivered_kg=minimum_kg, recorded_metric_point_mass_kg=mass,
        scope='Numerical and physical compatibility probe; model success is not evaluated; '
              'three distinct complete ten-cycle replay gates remain required before training',
        variants=['original controls / bucket6', 'reconstructed controls / bucket20']))
    variants = []
    for name, actions, state, gpu in [('original', original_actions, states, '0'),
                                      ('reconstructed', repaired, revised, '1')]:
        if args.variant != 'both' and args.variant != name:
            continue
        if args.gpu is not None:
            gpu = args.gpu
        folder = out / name
        folder.mkdir()
        contract = folder / 'probe_contract_only/meta'
        contract.mkdir(parents=True)
        declaration = copy.deepcopy(info)
        if name == 'reconstructed':
            declaration.update(action_limit_profile=BUCKET20_PROFILE,
                               action_limits_rad_s=PROFILES[BUCKET20_PROFILE])
        # This directory is metadata for an expert probe, not a training dataset.
        atomic_json(contract / 'info.json', declaration)
        atomic_json(folder / 'reference.json', dict(common, actions=actions.tolist(),
                     expected_initial_state=state[0].tolist()))
        command = [sys.executable, '-B', str(BASE / 'scripts/eval_smolvla_success.py'),
            '--policy-type', 'act', '--checkpoint', str(checkpoint), '--dataset', str(contract.parent),
            '--output', str(folder / 'evaluation'), '--reference-episode', str(folder / 'reference.json'),
            '--replay-reference', '--replay-full-reference', '--episodes', '1',
            '--steps', str(len(states)), '--min-particles', str(math.ceil(minimum_kg / mass)),
            '--replan-interval', '3', '--seed', '2026111800', '--gpus', gpu, '--workers', '1',
            '--soil-backend', 'newton_mpm', '--cpu-threads', '12', '--video', '--headless',
            '--fresh-simulator', '--render-every-physics-step', '--progress-interval', '15']
        atomic_json(folder / 'launch.json', dict(command=command))
        variants.append((folder, command, name))
    def run(variant):
        folder, command, name = variant
        with (folder / 'launch.log').open('x') as handle:
            result = subprocess.run(command, cwd=BASE, env=evaluation_environment(out),
                                    stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
        atomic_json(folder / 'launch_result.json', dict(exit_code=result.returncode, finished_utc=now()))
        if result.returncode:
            raise RuntimeError(f'{name} probe infrastructure failed; see {folder / "launch.log"}')
        episode = folder / 'evaluation/episodes/episode_0001'
        rows = [json.loads(line) for line in (episode / 'steps.jsonl').open()]
        status = json.loads((episode / 'episode_status.json').read_text())
        if not status.get('full_reference_executed'):
            raise ValueError('Probe did not execute the complete selected recorded sequence')
        observed = np.array([r['input_state'][:4] for r in rows])
        target_observed = np.array([r['joint_command_positions'] for r in rows[:-1]])
        expected_target = states[1:, :4] + states[1:, 8:12]
        joint_errors = observed - states[:, :4]
        metrics = dict(status='completed', source_episode=args.episode, variant=name,
            expert_replay=True, model_success_evaluated=False, steps=len(rows),
            measured_joint_max_error_rad=float(abs(joint_errors).max()),
            measured_joint_mae_rad=abs(joint_errors).mean(0).tolist(),
            final_joint_error_deg=np.degrees(joint_errors[-1]).tolist(),
            command_target_max_error_rad=float(abs(target_observed - expected_target).max()),
            delivered_particles=status['final_newly_deposited_particle_count'],
            video=str(episode / 'evaluation.mp4'))
        cycle_results = []
        prior_count = 0
        for scoop, group in frame.groupby('observation.scoop_index', sort=True):
            end = int(np.flatnonzero(frame['observation.scoop_index'].to_numpy() == scoop)[-1])
            count = rows[end]['deposited_particles']
            delivered_kg = (count - prior_count) * mass
            cycle_results.append(dict(cycle_index=int(scoop), end_step=end,
                net_delivered_particles=count - prior_count, net_delivered_kg=delivered_kg,
                delivery_passed=delivered_kg >= minimum_kg,
                return_joint_error_rad=abs(joint_errors[end]).tolist()))
            prior_count = count
        metrics.update(cycles=cycle_results, cycles_tested=len(cycle_results),
                       numerical_target_parity_passed=metrics['command_target_max_error_rad'] <= .001,
                       measured_joint_parity_passed=metrics['measured_joint_max_error_rad'] <= .15,
                       all_cycle_deliveries_passed=all(r['delivery_passed'] for r in cycle_results))
        atomic_json(folder / 'compatibility.json', metrics)
        return metrics
    results = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, variant) for variant in variants]
        while not all(f.done() for f in futures):
            time.sleep(5)
        for future in futures:
            results.append(future.result())
    if digest(meta_path) != common['provenance']['source_meta_sha256']:
        raise ValueError('Source evidence changed during the probe')
    atomic_json(out / 'comparison.json', dict(status='completed', expert_only=True,
                 training_ready=False,
                 reason='Three distinct complete ten-cycle replay gates and a separately published candidate remain required',
                 variants=results))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
