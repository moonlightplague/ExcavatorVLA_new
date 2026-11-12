"""Explicit real-PhysX goal-update regression, with no model/expert motion."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'ExcavatorVLA-smolvla')]
from excavator_common.bridge_protocol import read_json, write_json
from excavator_common.deployment_contract import build_client_contract, OBSERVATION_SCHEMA_28D_V4_PLUS_EFFORT
from excavator_common.simulation_settings import resolve_simulation_settings, apply_simulation_settings
from success_eval_runtime import SimulatorSession, resolve_gpu_ids
from repeat_eval import load_repeat_spec, verify_goal_ack


def exchange(sock, command):
    write_json(sock, command)
    reply = read_json(sock)
    if reply.get('ok') is False or reply.get('type') == 'error':
        raise RuntimeError(reply)
    return reply


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = resolve_gpu_ids(args.gpu)[0]
    apply_simulation_settings(resolve_simulation_settings('physx', True))
    dataset = ROOT / 'outputs/datasets/pi05_repeat_physx_bucket20_v2'
    reference_path = dataset / 'references/episode_0000.json'
    reference = json.loads(reference_path.read_text())
    spec = load_repeat_spec(dataset, reference_path, 5, 500, 100)
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    with tempfile.TemporaryDirectory(prefix='pi05_repeat_goal_probe_', dir='/tmp') as directory:
        folder = Path(directory)
        profile = folder / 'scene_profile.json'
        profile.write_text(json.dumps(reference['profile']))
        command = [str(ROOT / 'run_isaac_python.sh'), str(ROOT / 'ExcavatorVLA-smolvla/run_simulation.py'),
            '--headless', '--render-every-physics-step', '--active-cuda-gpu', '0', '--cpu-threads', '8',
            '--idle-ui-hz', '.1', '--sand-amount', str(reference['sand_amount']),
            '--robot-initial-joints-deg', *map(str, reference['initial_joints_deg']),
            '--scene-profile', str(profile), '--state27-dig-target-world', *map(str, reference['dig_target_xyz']),
            '--state27-unload-target-world', *map(str, reference['profile']['unload_landing_xyz']),
            '--task-text', reference['task']]
        session = SimulatorSession(command, ROOT / 'ExcavatorVLA-smolvla', folder / 'simulator.log', False, exchange)
        try:
            session.prepare(2)
            scene = exchange(session.sock, {'type': 'scene_context'})
            context = {**scene['observation_context'], 'allow_task_goal_updates': True, 'goal_revision': 0}
            contract = build_client_contract(10, 'goal-update-test-no-policy',
                                             OBSERVATION_SCHEMA_28D_V4_PLUS_EFFORT, context)
            exchange(session.sock, contract)
            reply = exchange(session.sock, {'joint_velocities': [0.] * 4})
            if not reply['observation_32d_ready'] or not reply.get('evaluation_particle_ids'):
                raise AssertionError('Missing v4 observation or source-particle identity payload')
            for revision, goal in enumerate(spec['goals'][1:], 1):
                updated = exchange(session.sock, {'type': 'set_task_goals', 'revision': revision, **goal})
                verify_goal_ack(reply, updated, goal, revision)
                heading = updated['observation_context']['initial_heading_rad'] + updated['joint_positions'][0]
                c, s = np.cos(heading), np.sin(heading)
                expected = []
                for name, landmark in [('dig_target_xyz', 'tip'), ('unload_landing_xyz', 'load')]:
                    delta = np.asarray(goal[name]) - updated['bucket_landmarks_world'][landmark]
                    expected.extend([c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1], delta[2]])
                np.testing.assert_allclose(updated['observation_state_28d'][18:24], expected, atol=2e-5, rtol=0)
                reply = updated
            following = exchange(session.sock, {'joint_velocities': [0.] * 4})
            if following['bridge_timing']['physics_ticks'] != 6 or following['bridge_timing']['simulated_seconds'] != .1:
                raise AssertionError('Task-goal updates changed subsequent action cadence')
            np.testing.assert_array_equal(following['joint_command_positions'], reply['joint_command_positions'])
            np.testing.assert_array_equal(following['observation_state_28d'][12:16], [0.] * 4)
            print(json.dumps({'status': 'passed', 'real_physx_goal_updates': 4,
                'physics_ticks_during_goal_updates': 0, 'preserved': ['camera pixels', 'controller targets',
                'source IDs', 'previous-action state', '10Hz action cadence'],
                'expert_or_policy_actions_executed': False, 'test_motion': 'two zero commands only'}), flush=True)
        except BaseException:
            print('\n'.join(session.log_path.read_text(errors='replace').splitlines()[-35:]), flush=True)
            raise
        finally:
            session.close()


if __name__ == '__main__':
    main()
