"""Physical cycle detection independent of delivery quality and precise task goals."""
import math

import numpy as np

from excavator_common.evaluation_standard import BUCKET_OPEN_RAD
from repeat_eval import RepeatedTask, source_ids


def point_in_polygon(point, polygon):
    x, y = point[:2]
    inside = False
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        ax, ay = a
        bx, by = b
        dx, dy = bx-ax, by-ay
        if abs(dx*(y-ay)-dy*(x-ax)) <= 1e-7 and min(ax,bx)-1e-7 <= x <= max(ax,bx)+1e-7 and min(ay,by)-1e-7 <= y <= max(ay,by)+1e-7:
            return True
        if (ay > y) != (by > y) and x < ax + (y-ay)*dx/dy:
            inside = not inside
    return inside


class PhysicalRepeatTask(RepeatedTask):
    """Count dig/carry/open/return attempts; score every deposit separately.

    A failed deposit still completes a physical attempt, advances the next task
    goal and cannot turn two subsequent attempts into a single 500-step cycle.
    Only the enclosing episode's fixed step budget limits an unfinished task.
    """
    def start_cycle(self, reply, start_step, wall):
        super().start_cycle(reply, start_step, wall)
        self.phase = 'seek_dig'
        self.dig_step = self.carry_step = None
        self.dig_consecutive = 0
        self.saw_closed_bucket = False
        self.peak_bucket_particles = 0

    def observe(self, reply, step, wall):
        if self.finished or self.phase == 'await_goal_update' or step != self.last_step + 1:
            raise ValueError('Physical repeated-task observations must be consecutive with updated goals')
        self.last_step = step
        ids = source_ids(reply)
        self.global_transfer = self._global_transfer(ids)
        self.transfer = self._transfer(ids)
        state = reply['observation_state_28d']
        tip = np.asarray(reply['bucket_landmarks_world']['tip'], dtype=float)
        if len(state) != 28 or not np.isfinite(state).all() or tip.shape != (3,) or not np.isfinite(tip).all():
            raise ValueError('Invalid physical repeated-task observation')
        in_pile = (math.dist(tip[:2], self.spec['pile_center_xy']) <= self.spec['pile_radius_m']
                   if 'pile_radius_m' in self.spec else point_in_polygon(tip, self.spec['pile_polygon_xy']))
        height = float(tip[2]) - self.spec['ground_top_z']
        self.peak_bucket_particles = max(self.peak_bucket_particles, len(set(ids['bucket']) - self.initial_bucket))
        required = self.spec['physical_event_observations']
        if self.phase == 'seek_dig':
            digging = in_pile and height <= self.spec['dig_max_height_m']
            self.dig_consecutive = self.dig_consecutive + 1 if digging else 0
            if self.dig_consecutive >= required:
                self.dig_step, self.phase = step, 'carry_to_unload'
        if self.phase == 'carry_to_unload':
            self.saw_closed_bucket |= state[3] <= self.spec['carry_bucket_closed_rad']
            pile_dist = math.dist(tip[:2], self.spec['pile_center_xy'])
            truck_dist = math.dist(tip[:2], self.spec['truck_center_xy'])
            transport = (not in_pile and truck_dist < pile_dist
                         and height >= self.spec['carry_min_height_m'] and self.saw_closed_bucket)
            if transport and self.carry_step is None:
                self.carry_step = step
            opening = self.carry_step is not None and transport and state[3] >= BUCKET_OPEN_RAD
            self.dump_consecutive = self.dump_consecutive + 1 if opening else 0
            if self.dump_consecutive >= required:
                self.dump_step, self.phase = step, 'return_to_pile'
        if self.phase == 'return_to_pile':
            low, high = self.spec['return_height_above_ground_m']
            returned = in_pile and low <= height <= high
            self.return_consecutive = self.return_consecutive + 1 if returned else 0
            if self.return_consecutive >= self.spec['return_consecutive_observations']:
                delivered = self.transfer['payload_particles'] >= self.spec['min_payload_particles']
                scooped = self.transfer['scooped_particles'] >= self.spec['min_scooped_particles']
                empty = reply['bucket_load_metrics']['count'] <= self.spec['return_empty_bucket_particles']
                reason = ('low_delivered_mass_kg' if not delivered else
                          'insufficient_observed_scoop' if not scooped else
                          'bucket_not_empty_after_return' if not empty else '')
                self.cycles.append(self._row(delivered and scooped and empty, wall, reason))
                self.finished = len(self.cycles) == self.spec['cycles']
                self.phase = 'completed' if self.finished else 'await_goal_update'
                if self.finished and not all(row['success'] for row in self.cycles):
                    self.failure_reason = 'one_or_more_cycle_deliveries_failed'
                return 'finished' if self.finished else 'next_cycle'
        return None

    def _row(self, success, wall, reason):
        row = super()._row(success, wall, reason)
        completed = self.dump_step is not None and self.return_consecutive >= self.spec['return_consecutive_observations']
        row.update(status='completed' if completed else 'failed', physical_cycle_completed=completed,
                   dig_step=self.dig_step, carry_departure_step=self.carry_step,
                   max_bucket_particles=self.peak_bucket_particles,
                   cycle_time_seconds=row['steps']/self.spec['fps'] if completed else None)
        return row

    def result(self, *, final=False, wall=None):
        if final and not self.finished:
            self.failure_reason = self.failure_reason or 'episode_step_limit'
        row = super().result(final=final, wall=wall)
        row.update(repeat_cycles_completed=len(self.cycles),
                   repeat_cycles_successful=sum(item['success'] for item in self.cycles),
                   repeat_cycle_count_method='physical_dig_carry_open_return_v3_independent_of_delivery_quality',
                   repeat_failed_completed_cycles=sum(not item['success'] for item in self.cycles))
        return row
