"""Build immutable repaired controls, verify three replay gates, then train ACT."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BASE / 'scripts'), str(BASE / 'ExcavatorVLA-vlm')]
from run_act_original_experiment import write, digest, now


def assess_and_publish(dataset, probes):
    import numpy as np
    import pandas as pd
    from excavator_vlm.data.action_limits import action_contract
    from excavator_vlm.data.control_repair import require_control_repair_gate
    from excavator_vlm.data.contract import audit_dataset
    import re

    info = json.loads((dataset / 'meta/info.json').read_text())
    audit, _ = audit_dataset(dataset, 10)
    split = json.loads((dataset / 'split_manifest.json').read_text())
    if split['dataset_fingerprint'] != audit['fingerprint']:
        raise ValueError('Candidate fingerprint changed after preparation')
    results = []
    for probe in probes:
        protocol = json.loads((probe / 'protocol.json').read_text())
        eid = protocol['source_episode']
        if eid not in split['train'] + split['validation'] or protocol.get('cycles') != 10:
            raise ValueError('Replay must cover a distinct non-test ten-cycle parent')
        source = Path(info['relabel']['source'])
        if (Path(protocol['source']).resolve() != source.resolve()
                or digest(source / 'meta/info.json') != protocol['source_info_sha256']
                or digest(protocol['source_meta']) != protocol['source_meta_sha256']):
            raise ValueError('Replay source identity/checksum differs from original publication')
        folder = probe / 'reconstructed/evaluation/episodes/episode_0001'
        status = json.loads((folder / 'episode_status.json').read_text())
        rows = [json.loads(line) for line in (folder / 'steps.jsonl').open()]
        controls = np.load(probe / 'reference_controls.npz')
        states, actions = controls['states'], controls['repaired_actions']
        if (status.get('status') != 'completed' or status.get('execution_source') != 'expert_replay'
                or not status.get('full_reference_executed') or status.get('clipped_action_dimensions') != 0
                or len(rows) != len(states) or status['policy_steps_executed'] != len(states)):
            raise ValueError('Full unclipped expert-only replay was not completed')
        frame = pd.read_parquet(dataset / f'data/chunk-{eid:03d}/file-000.parquet')
        frame = frame.sort_values('frame_index').reset_index(drop=True)
        if len(frame) != len(states) or not np.array_equal(np.stack(frame.action), actions):
            raise ValueError('Replay controls differ from the actual candidate labels')
        candidate_states = np.stack(frame['observation.state'])
        if not np.array_equal(candidate_states[:, :12], states[:, :12]) or not np.array_equal(candidate_states[:, 16:], states[:, 16:]):
            raise ValueError('Candidate changed measured source observations')
        if any(r['action_source'] != 'expert_replay' for r in rows):
            raise ValueError('A replay action was not an expert control')
        applied = np.array([r['applied'] for r in rows])
        if not np.allclose(applied, actions, rtol=0, atol=1e-6):
            raise ValueError('Applied replay controls differ from candidate labels')
        observed = np.array([r['input_state'][:4] for r in rows])
        joint_error = np.abs(observed - states[:, :4])
        observed_target = np.array([r['joint_command_positions'] for r in rows[:-1]])
        target_error = np.abs(observed_target - states[1:, :4] - states[1:, 8:12])
        maximum_joint, maximum_target = float(joint_error.max()), float(target_error.max())
        if maximum_joint > .15 or maximum_target > .001:
            raise ValueError(f'Physical/numerical target parity failed for parent {eid}')
        simulator_log = probe / 'reconstructed/evaluation/workers/worker_00/simulator.log'
        match = re.search(r'\[MPM READY\].*?metric_points=(\d+).*?mass_kg=([0-9.]+)', simulator_log.read_text())
        if not match:
            raise ValueError('Missing live Newton metric mass evidence')
        live_mass = float(match[2]) / int(match[1])
        if not math.isclose(live_mass, protocol['recorded_metric_point_mass_kg'], rel_tol=1e-6):
            raise ValueError('Live metric mass differs from collection')
        gate_path = Path(protocol['source_meta']).parent.parent / 'quality_gate_config.json'
        minimum_kg = json.loads(gate_path.read_text())['repeat_min_delivered_kg']
        if minimum_kg != 50 or protocol['minimum_delivered_kg'] != minimum_kg:
            raise ValueError('Replay mass criterion differs from original collection')
        cycles, prior = [], 0
        for scoop, group in frame.groupby('observation.scoop_index', sort=True):
            end = int(group.index[-1])
            count = rows[end]['deposited_particles']
            mass_kg = (count - prior) * live_mass
            cycles.append(dict(cycle_index=int(scoop), end_step=end, net_delivered_kg=mass_kg))
            if mass_kg < minimum_kg:
                raise ValueError(f'Parent {eid}, scoop {scoop}: net delivered mass below collection criterion')
            prior = count
        if [c['cycle_index'] for c in cycles] != list(range(10)):
            raise ValueError('Replay is missing one or more complete cycles')
        evidence_folder = dataset / f'provenance/control_replay/episode_{eid:04d}'
        evidence_folder.mkdir(parents=True, exist_ok=False)
        files = {'status': folder / 'episode_status.json', 'trace': folder / 'steps.jsonl',
                 'reference_controls': probe / 'reference_controls.npz',
                 'compatibility': probe / 'reconstructed/compatibility.json',
                 'protocol': probe / 'protocol.json', 'simulator_log': simulator_log}
        evidence = []
        for kind, path in files.items():
            dest = evidence_folder / path.name
            shutil.copy2(path, dest)
            evidence.append(dict(kind=kind, path=str(dest.relative_to(dataset)), sha256=digest(dest)))
        result = dict(episode_index=eid, cycles_tested=10, full_reference_executed=True,
            clipped_action_dimensions=0, numerical_target_parity_passed=True,
            measured_joint_parity_passed=True, all_cycle_deliveries_passed=True,
            measured_joint_max_error_rad=maximum_joint, command_target_max_error_rad=maximum_target,
            live_metric_point_mass_kg=live_mass, minimum_delivered_kg=minimum_kg,
            cycles=cycles, evidence=evidence)
        write(evidence_folder / 'independent_assessment.json', result)
        results.append(result)
    if len(results) < 3 or len({r['episode_index'] for r in results}) != len(results):
        raise ValueError('Three distinct replay parents are required')
    gate = dict(schema_version='newton_control_repair_gate_v1', status='passed',
        source_fingerprint=info['relabel']['source_fingerprint'], dataset_fingerprint=audit['fingerprint'],
        soil_backend='newton_mpm', action_contract=action_contract(info), assessed_utc=now(), episodes=results,
        scope='Three full recorded ten-cycle expert replays; not model success or held-out task success rate')
    write(dataset / 'control_replay_gate.json', gate)
    require_control_repair_gate(dataset, audit['fingerprint'])
    report_path = dataset / 'repair_report.json'
    report = json.loads(report_path.read_text())
    report.update(status='numerical_dataset_and_three_full_replay_gates_passed', training_ready=True,
                  control_replay_gate_sha256=digest(dataset / 'control_replay_gate.json'))
    write(report_path, report)
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-output', type=Path, default=BASE / 'outputs/experiments/act_repeat_control_repair_data_20261118_01')
    parser.add_argument('--config', type=Path, default=BASE / 'training_config_act_repaired.yaml')
    args = parser.parse_args()
    root = args.data_output.resolve()
    lock = root.with_suffix('.controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    from prepare_act_control_repair import build
    probes = [BASE / f'outputs/experiments/act_control_replay10_ep{e:04d}_20261118_01' for e in (0, 22, 1)]
    child = None
    def update(phase, **values):
        write(root / 'pipeline_status.json', dict(phase=phase, updated_at=now(), controller_pid=os.getpid(), **values))
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        dataset = build(BASE / 'data/repeat/export_repeat', root,
            BASE / 'outputs/experiments/act_shortchunk_repeat40k_20261117_01/prepared/split_manifest.json')
        while not all((probe / 'comparison.json').is_file() for probe in probes):
            completed = []
            for probe in probes:
                f = probe / 'reconstructed/compatibility.json'
                if f.exists():
                    result = json.loads(f.read_text())
                    if not all(result.get(k) for k in ('numerical_target_parity_passed',
                            'measured_joint_parity_passed', 'all_cycle_deliveries_passed')):
                        raise ValueError(f'Replay gate failed: {probe}')
                    completed.append(result['source_episode'])
                result_path = probe / 'reconstructed/launch_result.json'
                if result_path.exists() and json.loads(result_path.read_text()).get('exit_code'):
                    raise RuntimeError(f'Replay infrastructure error: {probe}')
            update('waiting_for_three_complete_replays', passed_replay_parents=completed,
                   dataset=str(dataset), training_started=False)
            time.sleep(5)
        update('independently_verifying_replay_evidence', training_started=False)
        assess_and_publish(dataset, probes)
        update('preparing_and_launching_training', training_started=False)
        command = [sys.executable, '-B', '-u', str(BASE / 'scripts/run_act_original_experiment.py'),
                   '--config', str(args.config.resolve()), '--split-source', str(dataset / 'split_manifest.json')]
        with (root / 'training_controller.log').open('x') as handle:
            child = subprocess.Popen(command, cwd=BASE, stdin=subprocess.DEVNULL,
                                     stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            while child.poll() is None:
                training_status = BASE / 'outputs/experiments/act_repaired_repeat40k_20261118_01/status.json'
                status = json.loads(training_status.read_text()) if training_status.exists() else {}
                update('training_controller_running', training_controller_pid=child.pid,
                       training_started=status.get('phase') in ('training', 'verifying_final', 'completed'),
                       training_phase=status.get('phase', 'preparing'), step=status.get('step', 0),
                       target_steps=40000)
                time.sleep(5)
        if child.returncode:
            raise RuntimeError('Training controller failed; see training_controller.log')
        update('completed', training_started=True, step=40000)
    except BaseException as error:
        if root.exists():
            update('failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait(timeout=30)


if __name__ == '__main__':
    main()
