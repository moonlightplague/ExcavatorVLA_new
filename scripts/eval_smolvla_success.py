"""Evaluate 28D policy success with persistent, GPU-isolated Isaac workers."""
import argparse
import json
from pathlib import Path
import signal
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
from eval_video import EpisodeVideo
from success_eval_runtime import (
    SimulatorSession, atomic_json, episode_indices, resolve_gpu_ids, run_workers,
)
from excavator_diagnostics import dataset_initial_pose, reference_episode


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


def validate_state(reply):
    if not reply.get('observation_32d_ready'):
        raise ValueError(reply.get('observation_32d_error', '28D observation unavailable'))
    if reply['observation_contract']['observation.state']['names'] != list(STATE_NAMES_28D):
        raise ValueError('Simulator state feature order differs from current-v4')
    state = np.asarray(reply['observation_state_28d'], dtype=np.float32)
    if state.shape != (28,) or not np.isfinite(state).all():
        raise ValueError('Invalid 28D state')
    return state


def observation(reply):
    state = validate_state(reply)
    batch = {'observation.state': torch.from_numpy(state).unsqueeze(0), 'task': [reply['task_text']]}
    images = decode_camera_images(reply, np_module=np)
    for camera in ('0', '1', '2'):
        rgb = images[camera]
        # A completed renderer readback may be black when the policy points a
        # camera into an occluding surface. Validate the delivered image shape;
        # the simulator validates readback completion and opaque alpha.
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f'Invalid camera {camera}')
        tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255
        batch[f'observation.images.{camera}'] = F.interpolate(tensor, (256, 256), mode='bilinear', align_corners=False)
    return batch


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
    parser.add_argument('--initial-episode', type=int, default=0,
                        help='Use this dataset episode initial joint pose for randomized scenes.')
    parser.add_argument('--reference-episode', type=Path,
                        help='Raw successful episode directory; reproduce its scene, pose, targets and task.')
    parser.add_argument('--replay-reference', action='store_true',
                        help='Execute reference expert actions instead of model actions to test the simulator.')
    parser.add_argument('--workers', type=int, default=0,
                        help='Persistent workers; 0 uses one per visible GPU (capped by episodes).')
    parser.add_argument('--gpus', help='Physical GPU indices or UUIDs, e.g. 0,1; respects CUDA_VISIBLE_DEVICES.')
    parser.add_argument('--headless', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--video', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--fresh-simulator', action='store_true',
                        help='Restart Isaac every episode for reset/parity diagnosis.')
    parser.add_argument('--render-every-physics-step', action='store_true',
                        help='Render intermediate physics ticks for comparison with the old evaluator.')
    parser.add_argument('--cpu-threads', type=int, default=0,
                        help='Simulator CPU threads per worker; 0 splits the available CPU quota.')
    parser.add_argument('--startup-timeout', type=float, default=600)
    parser.add_argument('--worker-index', type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--worker-count', type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument('--gpu-label', default='', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.episodes, args.steps, args.min_particles, args.replan_interval) < 1:
        parser.error('Episode, step, particle and replanning counts must be positive')
    if args.workers < 0 or args.cpu_threads < 0 or args.startup_timeout <= 0:
        parser.error('Worker/thread counts must be nonnegative and startup timeout positive')
    if not args.headless:
        if args.workers > 1:
            parser.error('Multiple GPU workers require headless mode; use --workers 1 for a GUI')
        args.workers = 1
    args.checkpoint = args.checkpoint.resolve()
    args.dataset = args.dataset.resolve()
    args.output = args.output.resolve()
    if not args.checkpoint.is_dir():
        parser.error(f'Checkpoint directory does not exist: {args.checkpoint}')
    info = json.loads((args.dataset / 'meta/info.json').read_text())
    if args.replay_reference and args.reference_episode is None:
        parser.error('--replay-reference requires --reference-episode')
    reference = reference_episode(args.reference_episode, info['fps']) if args.reference_episode else None
    initial_pose = reference['initial_joints_deg'] if reference else dataset_initial_pose(args.dataset, args.initial_episode)
    action_limits = np.asarray(info['action_limits_rad_s'], dtype=np.float32)
    if action_limits.shape != (4,) or not np.isfinite(action_limits).all() or np.any(action_limits <= 0):
        raise ValueError('Invalid dataset action limits')
    if info['features']['observation.state']['names'] != list(STATE_NAMES_28D):
        raise ValueError('Expected current-v4 dataset feature order')
    if info['features']['action']['names'] != list(ACTION_NAMES_4D):
        raise ValueError('Expected canonical four-joint velocity actions')
    normalization_hash = sha256_files(sorted(args.checkpoint.glob('*processor*')))
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k not in ('worker_index', 'worker_count', 'gpu_label')}
    config.update(training_fps=info['fps'], supervisor=False, bucket_open_threshold=-0.80,
                  success_no_growth_steps=3, simulator=str(REPO / 'run_simulation.py'),
                  initial_joints_deg=initial_pose, execution_source='expert_replay' if args.replay_reference else 'model')
    if args.worker_index is None:
        args.output.mkdir(parents=True, exist_ok=False)
        atomic_json(args.output / 'config.json', config)
    if reference and args.worker_index is None:
        (args.output / 'scene_profile.json').write_text(json.dumps(reference['profile'], indent=2) + '\n')
    if args.worker_index is None:
        gpu_ids = resolve_gpu_ids(args.gpus)
        if args.workers > len(gpu_ids):
            parser.error('--workers cannot exceed the selected GPU count')
        last_count = -1

        def summarize():
            nonlocal last_count
            rows = load_episode_rows(args.output / 'episodes')
            if len(rows) != last_count:
                write_outputs(args.output, rows, build_summary(rows, args.episodes))
                last_count = len(rows)
            return rows

        code = run_workers(args, Path(__file__).resolve(), gpu_ids, summarize)
        print(f'Results: {args.output}', flush=True)
        if code:
            print('Evaluation incomplete; see workers/worker_*/worker.log and simulator.log.', flush=True)
        raise SystemExit(code)

    torch.set_num_threads(min(4, args.cpu_threads or 4))
    policy = None
    if not args.replay_reference:
        policy = SmolVLAPolicy.from_pretrained(str(args.checkpoint)).to('cuda').eval()
        if list(policy.config.input_features['observation.state'].shape) != [28]:
            raise ValueError('Expected a 28D checkpoint')
        if list(policy.config.output_features['action'].shape) != [4]:
            raise ValueError('Expected a 4D action checkpoint')
        preprocess, postprocess = make_pre_post_processors(
            policy.config, pretrained_path=str(args.checkpoint),
            preprocessor_overrides={'device_processor': {'device': 'cuda'}},
        )
    worker_dir = args.output / 'workers' / f'worker_{args.worker_index:02d}'
    command = [str(BASE / 'run_isaac_python.sh'), str(REPO / 'run_simulation.py'),
               '--sand-amount', str(reference['sand_amount'] if reference else 1),
               '--robot-initial-joints-deg', *map(str, initial_pose),
               '--active-cuda-gpu', '0', '--cpu-threads', str(args.cpu_threads or 16)]
    if args.headless:
        command.extend(['--headless', '--idle-ui-hz', '0.1'])
    command.append('--render-every-physics-step' if args.render_every_physics_step
                   else '--no-render-every-physics-step')
    if reference:
        command += ['--scene-profile', str(args.output / 'scene_profile.json'),
                    '--state27-dig-target-world', *map(str, reference['dig_target_xyz']),
                    '--state27-unload-target-world', *map(str, reference['profile']['unload_landing_xyz']),
                    '--task-text', reference['task']]
    session = SimulatorSession(command, REPO, worker_dir / 'simulator.log',
                               args.video, exchange, args.startup_timeout)
    try:
        run_episodes(args, info, reference, action_limits, policy,
                     preprocess if policy is not None else None,
                     postprocess if policy is not None else None,
                     normalization_hash, config, session)
    finally:
        session.close()


def run_episodes(args, info, reference, action_limits, policy,
                 preprocess, postprocess, normalization_hash, config, session):
    failed = False
    for index in episode_indices(args.episodes, args.worker_index, args.worker_count):
        folder = args.output / 'episodes' / f'episode_{index:04d}'
        folder.mkdir(parents=True)
        status = {'schema_version': 'excavator_success_rate_episode_v1', 'status': 'error',
                  'success': False, 'task_completed_success_and_bucket_open': False,
                  'policy_steps_inferred': 0, 'policy_steps_executed': 0,
                  'early_terminated': False, 'error': None, 'scene_seed': args.seed + index - 1}
        status.update(worker_index=args.worker_index, gpu_id=args.gpu_label)
        status['timing_origin'] = 'episode_preparation_start'
        video = (EpisodeVideo(folder / 'evaluation.mp4', index, status['scene_seed'],
                              info['fps'], args.steps, args.min_particles) if args.video else None)
        status['video_path'] = str(video.path) if video is not None else None
        torch.manual_seed(status['scene_seed'])
        started = time.monotonic()
        try:
            status.update(session.prepare(status['scene_seed']))
            (folder / 'simulator.log').symlink_to(session.log_path)
            sock = session.sock
            with (folder / 'steps.jsonl').open('w') as trace:
                scene = exchange(sock, {'type': 'scene_context'})
                if scene.get('scene_seed') != status['scene_seed']:
                    raise ValueError(f'Unexpected scene seed: {scene}')
                atomic_json(folder / 'scene_context.json', scene)
                status['kit_pid'] = scene['simulator_pid']
                contract = build_client_contract(info['fps'], normalization_hash,
                    OBSERVATION_SCHEMA_28D_V4_PLUS_EFFORT, scene['observation_context'])
                ack = exchange(sock, contract)
                if (ack.get('type') != 'handshake_ack' or ack.get('timing_mode') != 'deterministic_lockstep'
                        or ack.get('state_names') != list(STATE_NAMES_28D)
                        or ack.get('training_fps') != info['fps']):
                    raise ValueError(f'Invalid deployment handshake: {ack}')
                (folder / 'contract.json').write_text(json.dumps({'request': contract, 'reply': ack}, indent=2))
                reply = exchange(sock, {'joint_velocities': [0.0] * 4})
                initial_state = validate_state(reply)
                initial_report = {'state': initial_state.tolist(), 'task': reply['task_text']}
                if reference:
                    initial_report['reference_state_error'] = (initial_state - reference['expected_initial_state']).tolist()
                (folder / 'initial_observation.json').write_text(json.dumps(initial_report, indent=2) + '\n')
                if reference and np.max(np.abs(initial_report['reference_state_error'][16:26])) > 0.05:
                    raise ValueError('Reference scene geometry differs by more than 0.05 from collection; see initial_observation.json')
                baseline = deposited_count(reply)
                status.update(initial_deposited_particle_count=baseline,
                              initial_bucket_particle_count=int(reply['bucket_load_metrics']['count']))
                previous = stable_steps = 0
                if video is not None:
                    video.write(reply, {**status, 'status': 'running',
                                        'final_bucket_open': float(reply['joint_positions'][3]) >= -0.80})
                if policy is not None:
                    policy.reset()
                actions = []
                policy_start = time.monotonic()
                for step in range(args.steps):
                    # Validate every returned observation, including between replans.
                    input_state = validate_state(reply).tolist()
                    if args.replay_reference:
                        if step >= len(reference['actions']):
                            status['termination_reason'] = 'reference_finished'
                            break
                        actions = [reference['actions'][step]]
                    elif not actions:
                        batch = observation(reply)
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
                    validate_state(reply)
                    count = max(0, deposited_count(reply) - baseline)
                    stable_steps, success = update_success(count, previous, stable_steps, args.min_particles)
                    previous = count
                    bucket_q = float(reply['joint_positions'][3])
                    bucket_open = bucket_q >= -0.80
                    elapsed = time.monotonic() - policy_start
                    status.update(success=success, task_completed_success_and_bucket_open=success and bucket_open,
                                  policy_steps_executed=step + 1, final_newly_deposited_particle_count=count,
                                  final_bucket_position_rad=bucket_q, final_bucket_open=bucket_open,
                                  max_bucket_particles=max(status.get('max_bucket_particles', 0),
                                                           int(reply['bucket_load_metrics']['count'])))
                    if success and 'success_time_from_policy_start_seconds' not in status:
                        status['success_time_from_policy_start_seconds'] = elapsed
                        status['success_time_from_simulation_launch_seconds'] = time.monotonic() - started
                    trace.write(json.dumps({'step': step, 'predicted': predicted.tolist(), 'applied': applied.tolist(),
                                            'input_state': input_state, 'state': reply['observation_state_28d'],
                                            'joint_command_positions': reply['joint_command_positions'],
                                            'deposited_particles': count, 'stable_steps': stable_steps,
                                            'bucket_position': bucket_q, 'bridge_timing': reply['bridge_timing']}) + '\n')
                    trace.flush()
                    if video is not None:
                        video.write(reply, {**status, 'status': 'running'}, stable_steps)
                    if success and bucket_open:
                        status.update(early_terminated=True, termination_reason='success_and_bucket_open',
                                      task_completion_time_from_policy_start_seconds=elapsed,
                                      task_completion_time_from_simulation_launch_seconds=time.monotonic() - started)
                        break
                status.update(status='completed', rollout_time_from_policy_start_seconds=time.monotonic() - policy_start,
                              rollout_time_from_simulation_launch_seconds=time.monotonic() - started)
                status.setdefault('termination_reason', 'step_limit')
                (folder / 'rollout_validation.json').write_text(json.dumps({
                    'status': 'passed', 'pure_model_execution_validated': not args.replay_reference,
                    'execution_source': config['execution_source'],
                    'supervisor_enabled': False, 'action_velocity_limits': action_limits.tolist(),
                }, indent=2))
        except Exception as exc:
            status['error'] = f'{type(exc).__name__}: {exc}'
            failed = True
        finally:
            if video is not None:
                try:
                    video.finish(status)
                    status['video_frames'] = video.frame_count
                except Exception as exc:
                    status['video_error'] = f'{type(exc).__name__}: {exc}'
                    status.update(status='error', error=status['video_error'])
                    print(f"Episode {index} video error: {status['video_error']}", flush=True)
                    failed = True
            atomic_json(folder / 'episode_status.json', status)
            if args.fresh_simulator or failed:
                session.close()
        print(f"Episode {index}: {status['status']}, success={status['success']}, error={status['error']}", flush=True)
        if failed:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
