"""Event-driven repeated-reference task; goals are instructions, never expert actions."""
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from excavator_common.evaluation_metrics import ParticleTransferTracker
from excavator_common.evaluation_standard import BUCKET_OPEN_RAD, COMPLETION_OBSERVATIONS
from excavator_common.vla_observation_contract import point_delta_in_heading_frame

STANDARD = 'excavator_repeat_reference_v1'


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vector(value):
    if (not isinstance(value, list) or len(value) != 3 or any(isinstance(x, bool)
            or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value)):
        raise ValueError('Repeated task needs three finite goal coordinates')
    return list(value)


def load_repeat_spec(dataset, reference_path, cycles, cycle_steps, strict_minimum):
    dataset, reference_path = Path(dataset), Path(reference_path)
    reference = read(reference_path)
    eid = reference['provenance']['source_export_episode']
    metadata = dataset / f'provenance/episode_{eid:04d}/meta.json'
    index = dataset / f'provenance/episode_{eid:04d}/episode_index_record.json'
    backend = read(dataset / 'backend_provenance.json')
    entry = next(row for row in backend['episodes'] if row['episode_index'] == eid)
    if (backend['soil_backend'] != 'physx' or digest(metadata) != entry['evidence']['sha256']
            or reference['provenance']['source_parent'] != entry['parent_id']):
        raise ValueError('Repeated-reference parent/backend evidence differs')
    meta, record = read(metadata), read(index)
    if cycles != 5 or meta['scoops_completed'] != cycles or len(meta['repeat_cycles']) != cycles:
        raise ValueError('This version evaluates the recorded complete five-cycle task')
    if not meta['success'] or not record['success'] or len(record['scoop_results']) != cycles:
        raise ValueError('Repeated task requires a successful recorded five-cycle parent')
    gate = read(dataset / 'expert_replay_gate.json')
    criterion = gate['collection_criterion']
    minimum = criterion['min_payload_particles_per_cycle']
    scoop_minima = set()
    evidence = {}
    for item in criterion['source_evidence']:
        path = dataset / item['path']
        if digest(path) != item['sha256']:
            raise ValueError('Archived collection criterion checksum differs')
        config = read(path)
        if config['quality_min_dump_particles'] != minimum:
            raise ValueError('Collection dump thresholds differ')
        scoop_minima.add(config['quality_min_bucket_particles'])
        evidence[str(path)] = item['sha256']
    if len(scoop_minima) != 1 or type(minimum) is not int or minimum < 1:
        raise ValueError('Inconsistent collection scoop/dump criteria')
    goals = []
    for i, row in enumerate(record['scoop_results']):
        if row['scoop_index'] != i or not row['execution_success']:
            raise ValueError('Recorded scoop sequence is incomplete')
        goal = {'dig_target_xyz': vector(row['target_xyz']),
                'unload_landing_xyz': vector(row['unload_landing_xyz'])}
        if not np.allclose(goal['unload_landing_xyz'], reference['repeat_cycles'][i]['landing']['landing'], atol=1e-5):
            raise ValueError('Recorded task landing differs from reference metadata')
        goals.append(goal)
    if not np.allclose(goals[0]['dig_target_xyz'], reference['dig_target_xyz'], atol=1e-5):
        raise ValueError('First repeated-task dig target differs from initial scene')
    return {'version': STANDARD, 'cycles': cycles, 'fps': reference['fps'], 'cycle_step_limit': cycle_steps,
            'goal_source': 'recorded_reference_task_coordinates_v1_not_live_fill_replanning',
            'task': reference['task'], 'goals': goals,
            'min_payload_particles': minimum, 'min_scooped_particles': scoop_minima.pop(),
            'strict_min_payload_particles': strict_minimum,
            'return_xy_radius_m': 1.25, 'return_height_above_target_m': [.15, 2.5],
            'return_empty_bucket_particles': 5, 'return_consecutive_observations': 3,
            'source_evidence': {str(metadata): digest(metadata), str(index): digest(index), **evidence},
            'scope': 'Five model-controlled scoop/dump/return cycles in one preserved reference scene; '
                     'fixed recorded task goals, not live collection replanning or novel-scene generalization.'}


def source_ids(reply):
    ids = reply.get('evaluation_particle_ids')
    if not isinstance(ids, dict) or not all(k in ids for k in ('particle_count', 'bucket', 'truck')):
        raise ValueError('Repeated task requires sampled source-particle IDs')
    return ids


class RepeatedTask:
    def __init__(self, spec, initial_reply, *, started_wall=None):
        self.spec = spec
        self.cycles = []
        self.last_step = -1
        self.finished = False
        self.failure_reason = None
        self.start_cycle(initial_reply, 0, time.monotonic() if started_wall is None else started_wall)

    def start_cycle(self, reply, start_step, wall):
        ids = source_ids(reply)
        self.initial_bucket = set(ids['bucket'])
        self.tracker = ParticleTransferTracker()
        self.start_step, self.started_wall = start_step, wall
        self.phase = 'dig_and_dump'
        self.dump_consecutive = self.return_consecutive = 0
        self.dump_step = None
        self.transfer = self._transfer(ids)

    def _transfer(self, ids):
        return self.tracker.observe(ids['particle_count'],
            sorted(set(ids['bucket']) - self.initial_bucket),
            sorted(set(ids['truck']) - self.initial_bucket))

    def observe(self, reply, step, wall):
        if self.finished or self.phase == 'await_goal_update' or step != self.last_step + 1:
            raise ValueError('Repeated task observations must be consecutive and use the updated task goal')
        self.last_step = step
        self.transfer = self._transfer(source_ids(reply))
        state = reply['observation_state_28d']
        if len(state) != 28 or not np.isfinite(state).all():
            raise ValueError('Invalid repeated-task observation')
        qualifies = (self.transfer['payload_particles'] >= self.spec['min_payload_particles']
                     and self.transfer['scooped_particles'] >= self.spec['min_scooped_particles']
                     and state[3] >= BUCKET_OPEN_RAD)
        if self.phase == 'dig_and_dump':
            self.dump_consecutive = self.dump_consecutive + 1 if qualifies else 0
            if self.dump_consecutive >= COMPLETION_OBSERVATIONS:
                self.phase, self.dump_step = 'return_to_pile', step
        if self.phase == 'return_to_pile':
            xy = math.hypot(state[18], state[19])
            height = -float(state[20])
            low, high = self.spec['return_height_above_target_m']
            returned = (xy <= self.spec['return_xy_radius_m'] and low <= height <= high
                        and reply['bucket_load_metrics']['count'] <= self.spec['return_empty_bucket_particles'])
            self.return_consecutive = self.return_consecutive + 1 if returned else 0
            if self.return_consecutive >= self.spec['return_consecutive_observations']:
                success = qualifies or (self.transfer['payload_particles'] >= self.spec['min_payload_particles']
                    and self.transfer['scooped_particles'] >= self.spec['min_scooped_particles'])
                self.cycles.append(self._row(success, wall, '' if success else 'deposit_lost_before_return'))
                self.finished = not success or len(self.cycles) == self.spec['cycles']
                self.failure_reason = None if success else 'deposit_lost_before_return'
                self.phase = 'completed' if self.finished and success else 'await_goal_update'
                return 'finished' if self.finished else 'next_cycle'
        if step - self.start_step + 1 >= self.spec['cycle_step_limit']:
            self.failure_reason = 'return_timeout' if self.dump_step is not None else 'dig_dump_timeout'
            self.finished = True
            return 'finished'
        return None

    def advance(self, reply, wall):
        if self.phase != 'await_goal_update' or self.finished:
            raise ValueError('Cannot advance an incomplete or finished repeated task')
        self.start_cycle(reply, self.last_step + 1, wall)

    def _row(self, success, wall, reason):
        return {'cycle_index': len(self.cycles), 'status': 'completed' if success else 'failed',
                'success': bool(success), 'strict_success': bool(success and
                    self.transfer['payload_particles'] >= self.spec['strict_min_payload_particles']),
                'returned': self.return_consecutive >= self.spec['return_consecutive_observations'],
                'start_step': self.start_step, 'end_step': self.last_step, 'dump_step': self.dump_step,
                'steps': max(0, self.last_step - self.start_step + 1), 'failure_mode': reason or 'none',
                'cycle_time_seconds': (self.last_step - self.start_step + 1) / self.spec['fps'] if success else None,
                'runtime_seconds': max(0., wall - self.started_wall),
                'goal': self.spec['goals'][len(self.cycles)], **self.transfer}

    def result(self, *, final=False, wall=None):
        rows = list(self.cycles)
        if final and len(rows) < self.spec['cycles'] and self.phase != 'await_goal_update':
            rows.append(self._row(False, time.monotonic() if wall is None else wall,
                                 self.failure_reason or 'step_limit'))
        if final:
            rows += [{'cycle_index': i, 'status': 'not_attempted', 'success': False,
                      'strict_success': False, 'returned': False, 'failure_mode': 'previous_cycle_incomplete'}
                     for i in range(len(rows), self.spec['cycles'])]
        completed = sum(row['success'] for row in self.cycles)
        success = completed == self.spec['cycles']
        return {'repeat_schema_version': STANDARD, 'repeat_cycles_target': self.spec['cycles'],
                'repeat_min_payload_particles': self.spec['min_payload_particles'],
                'repeat_strict_min_payload_particles': self.spec['strict_min_payload_particles'],
                'repeat_cycles_completed': completed,
                'repeat_returns_completed': sum(row['returned'] for row in self.cycles),
                'repeat_collection_success': success,
                'repeat_strict_success': success and all(row['strict_success'] for row in self.cycles),
                'repeat_cycle_index': min(len(self.cycles), self.spec['cycles'] - 1),
                'repeat_phase': self.phase, 'repeat_cycles': rows,
                'repeat_cycle_payload_particles': self.transfer['payload_particles'],
                'repeat_failure_mode': 'none' if success else self.failure_reason or 'incomplete_repeated_task'}


def verify_goal_ack(before, after, goal, revision):
    if after.get('type') != 'task_goals_ack' or after.get('goal_revision') != revision:
        raise ValueError('Missing repeated-task goal acknowledgment')
    if after['bridge_timing']['simulated_seconds'] != 0 or after['bridge_timing']['physics_ticks'] != 0:
        raise ValueError('Task-goal update advanced physics')
    if not np.array_equal(before['joint_command_positions'], after['joint_command_positions']):
        raise ValueError('Task-goal update changed controller targets')
    for key in ('joint_positions', 'evaluation_particle_ids', 'cameras', 'evaluation_video_frame'):
        if before.get(key) != after.get(key):
            raise ValueError(f'Task-goal update changed physical observation: {key}')
    old, new = np.asarray(before['observation_state_28d']), np.asarray(after['observation_state_28d'])
    keep = [*range(18), *range(24, 28)]
    if not np.array_equal(old[keep], new[keep]):
        raise ValueError('Task-goal update changed state/history beyond goal geometry')
    if any(after['observation_context'][k] != v for k, v in goal.items()):
        raise ValueError('Acknowledged task coordinates differ from the requested goal')


def audit_repeat_trace(folder, spec, expected):
    folder = Path(folder)
    initial = read(folder / 'initial_observation.json')
    machine = RepeatedTask(spec, {'evaluation_particle_ids': initial['evaluation_particle_ids']}, started_wall=0.)
    rows = [json.loads(line) for line in (folder / 'steps.jsonl').read_text().splitlines()]
    heading_origin = initial['observation_context']['initial_heading_rad']
    history = read(folder.parent.parent / 'config.json').get('observation_history', {}).get('delta_indices', [0])
    for step, row in enumerate(rows):
        if row['step'] != step or row['repeat_cycle_index'] != len(machine.cycles):
            raise ValueError('Trace has a discontinuous repeated-task cycle/step sequence')
        if row.get('action_source') != 'model' or (len(history) > 1 and (
                row.get('input_history_steps') != [max(0, step + offset) for offset in history]
                or row.get('input_history_is_pad') != [step + offset < 0 for offset in history])):
            raise ValueError('Repeated trace has an expert action or reset/skipped policy history')
        goal = spec['goals'][len(machine.cycles)]
        heading = heading_origin + row['state'][0]
        landmarks = row['bucket_landmarks_world']
        wanted = [*point_delta_in_heading_frame(goal['dig_target_xyz'], landmarks['tip'], heading),
                  *point_delta_in_heading_frame(goal['unload_landing_xyz'], landmarks['load'], heading)]
        if not np.allclose(wanted, row['state'][18:24], atol=2e-5, rtol=0):
            raise ValueError('Trace task-goal geometry differs from the active repeated task')
        reply = {'evaluation_particle_ids': row['evaluation_particle_ids'],
                 'observation_state_28d': row['state'],
                 'bucket_load_metrics': {'count': row['repeat_bucket_particles']}}
        event = machine.observe(reply, step, step / spec['fps'])
        if event == 'next_cycle':
            machine.advance(reply, step / spec['fps'])
        elif event == 'finished' and step != len(rows) - 1:
            raise ValueError('Trace continued after repeated task termination')
    actual = machine.result(final=True, wall=len(rows) / spec['fps'])
    for key in ('repeat_cycles_completed', 'repeat_returns_completed', 'repeat_collection_success', 'repeat_strict_success'):
        if actual[key] != expected[key]:
            raise ValueError(f'Repeated task summary differs from saved trace: {key}')
    for a, b in zip(actual['repeat_cycles'], expected['repeat_cycles']):
        for key in ('status', 'returned', 'strict_success', 'payload_particles', 'scooped_particles', 'spillage_particles'):
            if a.get(key) != b.get(key):
                raise ValueError(f'Repeated cycle accounting differs from saved trace: {key}')
    updates = read(folder / 'goal_updates.json')
    transitions = [row for row in actual['repeat_cycles'] if row['success']][:spec['cycles'] - 1]
    if len(updates) != len(transitions):
        raise ValueError('Missing or extra repeated-task goal transition')
    for revision, (event, cycle) in enumerate(zip(updates, transitions), 1):
        if (event['after_step'] != cycle['end_step'] or event['command'] != {
                'type': 'set_task_goals', 'revision': revision, **spec['goals'][revision]}
                or event['physics_ticks'] != 0 or event['simulated_seconds'] != 0
                or event['camera_sha256'] != rows[cycle['end_step']]['camera_sha256']
                or event['joint_command_positions'] != rows[cycle['end_step']]['joint_command_positions']):
            raise ValueError('Goal transition changed physical state or used an invalid cycle boundary')
    return {'status': 'passed', 'steps': len(rows), 'cycles_completed': actual['repeat_cycles_completed'],
            'check': 'independent event/ID/return reconstruction and active task-goal geometry'}
