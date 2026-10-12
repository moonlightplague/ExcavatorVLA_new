"""Evaluate the current 28D policy in the legacy lockstep Isaac simulator."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

BASE = Path(__file__).resolve().parents[1]
REPO = BASE / 'ExcavatorVLA-smolvla'
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/evaluation'))
from excavator_common.bridge_protocol import read_json, write_json, decode_camera_images
from excavator_common.deployment_contract import (
    build_client_contract, OBSERVATION_SCHEMA_28D_V4_PLUS_EFFORT, STATE_NAMES_28D,
    ACTION_NAMES_4D, sha256_files,
)
from summarize_isaacsim_success_rate import load_episode_rows, build_summary, write_outputs


def exchange(sock, command):
    write_json(sock, command)
    reply = read_json(sock)
    if reply.get('type') == 'error' or reply.get('ok') is False:
        raise RuntimeError(reply)
    return reply


def deposited_count(reply):
    metric = reply['truck_bed_load_metrics']
    if (not metric['available'] or not metric['source_tracking_valid']
            or metric['method'] != 'world_aabb_excluding_bucket_v2'):
        raise ValueError(f'Invalid truck load metric: {metric}')
    return int(metric['count'])


def update_success(count, previous, stable_steps, minimum):
    stable_steps = stable_steps + 1 if count >= minimum and previous >= minimum and count <= previous else 0
    return stable_steps, count >= minimum and stable_steps >= 3


def observation(reply):
    if not reply.get('observation_32d_ready'):
        raise ValueError(reply.get('observation_32d_error', '28D observation unavailable'))
    if reply['observation_contract']['observation.state']['names'] != list(STATE_NAMES_28D):
        raise ValueError('Simulator state feature order differs from current-v4')
    state = np.asarray(reply['observation_state_28d'], dtype=np.float32)
    if state.shape != (28,) or not np.isfinite(state).all():
        raise ValueError('Invalid 28D state')
    batch = {'observation.state': torch.from_numpy(state).unsqueeze(0), 'task': [reply['task_text']]}
    images = decode_camera_images(reply, np_module=np)
    for camera in ('0', '1', '2'):
        rgb = images[camera]
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or not np.any(rgb):
            raise ValueError(f'Invalid camera {camera}')
        tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255
        batch[f'observation.images.{camera}'] = F.interpolate(tensor, (256, 256), mode='bilinear', align_corners=False)
    return batch


def connect_simulator(process, port, timeout=300):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'Simulator exited with {process.returncode}; see simulator.log')
        try:
            sock = socket.create_connection(('127.0.0.1', port), timeout=1)
            sock.settimeout(120)
            return sock
        except OSError:
            time.sleep(1)
    raise TimeoutError('Simulator startup timed out; see simulator.log')


def stop_simulator(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--steps', type=int, default=350)
    parser.add_argument('--min-particles', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--replan-interval', type=int, default=3)
    args = parser.parse_args()
    if min(args.episodes, args.steps, args.min_particles, args.replan_interval) < 1:
        parser.error('Episode, step, particle and replanning counts must be positive')
    info = json.loads((args.dataset / 'meta/info.json').read_text())
    action_limits = np.asarray(info['action_limits_rad_s'], dtype=np.float32)
    if action_limits.shape != (4,) or not np.isfinite(action_limits).all() or np.any(action_limits <= 0):
        raise ValueError('Invalid dataset action limits')
    if info['features']['observation.state']['names'] != list(STATE_NAMES_28D):
        raise ValueError('Expected current-v4 dataset feature order')
    if info['features']['action']['names'] != list(ACTION_NAMES_4D):
        raise ValueError('Expected canonical four-joint velocity actions')
    policy = SmolVLAPolicy.from_pretrained(str(args.checkpoint)).to('cuda').eval()
    if list(policy.config.input_features['observation.state'].shape) != [28]:
        raise ValueError('Expected a 28D checkpoint')
    if list(policy.config.output_features['action'].shape) != [4]:
        raise ValueError('Expected a 4D action checkpoint')
    preprocess, postprocess = make_pre_post_processors(
        policy.config, pretrained_path=str(args.checkpoint),
        preprocessor_overrides={'device_processor': {'device': 'cuda'}},
    )
    normalization_hash = sha256_files(sorted(args.checkpoint.glob('*processor*')))
    args.output.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(training_fps=info['fps'], supervisor=False, bucket_open_threshold=-0.80,
                  success_no_growth_steps=3, simulator=str(REPO / 'run_simulation.py'))
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    failed = False
    for index in range(1, args.episodes + 1):
        folder = args.output / 'episodes' / f'episode_{index:04d}'
        folder.mkdir(parents=True)
        status = {'schema_version': 'excavator_success_rate_episode_v1', 'status': 'error',
                  'success': False, 'task_completed_success_and_bucket_open': False,
                  'policy_steps_inferred': 0, 'policy_steps_executed': 0,
                  'early_terminated': False, 'error': None, 'scene_seed': args.seed + index - 1}
        # A fresh simulator resets sand, the robot, truck counts and observation history.
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        env = dict(os.environ, EXCAVATOR_PROJECT_ROOT=str(REPO), EXCAVATOR_BRIDGE_PORT=str(port))
        torch.manual_seed(status['scene_seed'])
        started = time.monotonic()
        with (folder / 'simulator.log').open('w') as log:
            process = subprocess.Popen([str(BASE / 'run_isaac_python.sh'), str(REPO / 'run_simulation.py'),
                                        '--scene-seed', str(status['scene_seed']), '--sand-amount', '1'],
                                       cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                with connect_simulator(process, port) as sock, (folder / 'steps.jsonl').open('w') as trace:
                    scene = exchange(sock, {'type': 'scene_context'})
                    contract = build_client_contract(info['fps'], normalization_hash,
                        OBSERVATION_SCHEMA_28D_V4_PLUS_EFFORT, scene['observation_context'])
                    ack = exchange(sock, contract)
                    if (ack.get('type') != 'handshake_ack' or ack.get('timing_mode') != 'deterministic_lockstep'
                            or ack.get('state_names') != list(STATE_NAMES_28D)
                            or ack.get('training_fps') != info['fps']):
                        raise ValueError(f'Invalid deployment handshake: {ack}')
                    (folder / 'contract.json').write_text(json.dumps({'request': contract, 'reply': ack}, indent=2))
                    reply = exchange(sock, {'joint_velocities': [0.0] * 4})
                    baseline = deposited_count(reply)
                    previous = stable_steps = 0
                    policy.reset()
                    actions = []
                    policy_start = time.monotonic()
                    for step in range(args.steps):
                        # Validate every returned observation, including between replans.
                        batch = observation(reply)
                        if not actions:
                            with torch.inference_mode():
                                chunk = policy.predict_action_chunk(preprocess(batch))
                                actions = [postprocess(chunk[:, i]).cpu().numpy()[0]
                                           for i in range(min(args.replan_interval, chunk.shape[1]))]
                            status['policy_steps_inferred'] += len(actions)
                        predicted = actions.pop(0)
                        if predicted.shape != (4,) or not np.isfinite(predicted).all():
                            raise ValueError('Invalid model action')
                        applied = np.clip(predicted, -action_limits, action_limits)
                        reply = exchange(sock, {'joint_velocities': applied.tolist()})
                        count = max(0, deposited_count(reply) - baseline)
                        stable_steps, success = update_success(count, previous, stable_steps, args.min_particles)
                        previous = count
                        bucket_q = float(reply['joint_positions'][3])
                        bucket_open = bucket_q >= -0.80
                        elapsed = time.monotonic() - policy_start
                        status.update(success=success, task_completed_success_and_bucket_open=success and bucket_open,
                                      policy_steps_executed=step + 1, final_newly_deposited_particle_count=count,
                                      final_bucket_position_rad=bucket_q, final_bucket_open=bucket_open)
                        if success and 'success_time_from_policy_start_seconds' not in status:
                            status['success_time_from_policy_start_seconds'] = elapsed
                            status['success_time_from_simulation_launch_seconds'] = time.monotonic() - started
                        trace.write(json.dumps({'step': step, 'predicted': predicted.tolist(), 'applied': applied.tolist(),
                                                'deposited_particles': count, 'stable_steps': stable_steps,
                                                'bucket_position': bucket_q, 'bridge_timing': reply['bridge_timing']}) + '\n')
                        trace.flush()
                        if success and bucket_open:
                            status.update(early_terminated=True, termination_reason='success_and_bucket_open',
                                          task_completion_time_from_policy_start_seconds=elapsed,
                                          task_completion_time_from_simulation_launch_seconds=time.monotonic() - started)
                            break
                    status.update(status='completed', rollout_time_from_policy_start_seconds=time.monotonic() - policy_start,
                                  rollout_time_from_simulation_launch_seconds=time.monotonic() - started)
                    status.setdefault('termination_reason', 'step_limit')
                    (folder / 'rollout_validation.json').write_text(json.dumps({
                        'status': 'passed', 'pure_model_execution_validated': True,
                        'supervisor_enabled': False, 'action_velocity_limits': action_limits.tolist(),
                    }, indent=2))
            except Exception as exc:
                status['error'] = f'{type(exc).__name__}: {exc}'
                failed = True
            finally:
                stop_simulator(process)
                (folder / 'episode_status.json').write_text(json.dumps(status, indent=2) + '\n')
        print(f"Episode {index}: {status['status']}, success={status['success']}, error={status['error']}", flush=True)
        summary = build_summary(load_episode_rows(args.output / 'episodes'), args.episodes)
        write_outputs(args.output, load_episode_rows(args.output / 'episodes'), summary)
        if failed:
            break
    print(f'Results: {args.output}', flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
