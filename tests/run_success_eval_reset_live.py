"""Explicit live reset regression; not part of automatic unit-test discovery."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BASE / 'scripts'), str(BASE / 'ExcavatorVLA-smolvla')]
from excavator_common.bridge_protocol import read_json, write_json
from excavator_common.deployment_contract import build_client_contract, OBSERVATION_SCHEMA_28D_V4_PLUS_EFFORT
from excavator_common.evaluation_standard import TRUCK_COUNT_METHOD
from excavator_diagnostics import dataset_initial_pose
from success_eval_runtime import SimulatorSession, atomic_json, resolve_gpu_ids


def exchange(sock, command):
    write_json(sock, command)
    reply = read_json(sock)
    if reply.get('ok') is False or reply.get('type') == 'error':
        raise RuntimeError(reply)
    return reply


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--dataset', type=Path, default=BASE / 'data/export')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['CUDA_VISIBLE_DEVICES'] = resolve_gpu_ids(args.gpu)[0]
    pose = dataset_initial_pose(args.dataset)
    fps = json.loads((args.dataset / 'meta/info.json').read_text())['fps']
    command = [str(BASE / 'run_isaac_python.sh'), str(BASE / 'ExcavatorVLA-smolvla/run_simulation.py'),
               '--headless', '--render-every-physics-step', '--active-cuda-gpu', '0',
               '--cpu-threads', '14', '--idle-ui-hz', '0.1', '--sand-amount', '1',
               '--robot-initial-joints-deg', *map(str, pose)]
    session = SimulatorSession(command, BASE / 'ExcavatorVLA-smolvla',
                               args.output / 'simulator.log', False, exchange)
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    results = []
    try:
        for seed in (2, 12, 2):
            preparation = session.prepare(seed)
            scene = exchange(session.sock, {'type': 'scene_context'})
            contract = build_client_contract(fps, hashlib.sha256(b'reset-test-no-policy').hexdigest(),
                                            OBSERVATION_SCHEMA_28D_V4_PLUS_EFFORT, scene['observation_context'])
            acknowledgment = exchange(session.sock, contract)
            if acknowledgment.get('type') != 'handshake_ack':
                raise AssertionError('Missing deployment handshake')
            reply = exchange(session.sock, {'joint_velocities': [0.] * 4})
            metric = reply['truck_bed_load_metrics']
            volume = scene['truck_counting_volume']
            if not (scene['scene_seed'] == volume['scene_seed'] == metric['scene_seed'] == seed
                    and metric['method'] == TRUCK_COUNT_METHOD and metric['available']
                    and metric['source_tracking_valid'] and reply['observation_32d_ready']):
                raise AssertionError('Invalid reset scene or metric')
            for key in ('world_min', 'world_max', 'local_min', 'local_max'):
                np.testing.assert_allclose(metric['counting_volume'][key], volume[key], atol=1e-6)
            if reply['particle_transfer_metrics']['payload_particles'] != 0:
                raise AssertionError('Reset retained a previous particle-transfer baseline')
            results.append(dict(seed=seed, simulator_pid=scene['simulator_pid'],
                                reused=preparation['simulator_reused'], counting_volume=volume))
        if len({result['simulator_pid'] for result in results}) != 1:
            raise AssertionError('Reset test restarted the simulator')
        if [result['reused'] for result in results] != [False, True, True]:
            raise AssertionError('Reset lifecycle was not exercised')
        if np.allclose(results[0]['counting_volume']['world_min'], results[1]['counting_volume']['world_min']):
            raise AssertionError('Truck bounds did not change with the new scene')
        np.testing.assert_allclose(results[0]['counting_volume']['world_min'],
                                   results[2]['counting_volume']['world_min'], atol=1e-5)
        result = dict(status='passed', evaluation_profile='diagnostic', scenes=results)
        atomic_json(args.output / 'reset_validation.json', result)
        print(json.dumps(result, indent=2))
    finally:
        session.close()


if __name__ == '__main__':
    main()
