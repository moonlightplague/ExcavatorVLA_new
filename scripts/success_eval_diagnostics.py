"""Observation-only diagnostics; these never change model actions or scoring."""
from collections import deque
import math


JOINT_NAMES = ('swing', 'boom', 'arm', 'bucket')


class RolloutDiagnostics:
    def __init__(self, joint_limits, stall_window_steps=50, tolerance_rad=math.radians(3)):
        if (len(joint_limits) != 4 or any(len(pair) != 2 or
                any(math.isnan(value) for value in pair) or pair[0] >= pair[1]
                for pair in joint_limits)):
            raise ValueError('Expected four ordered joint limits')
        if (type(stall_window_steps) is not int or stall_window_steps < 2 or
                not math.isfinite(tolerance_rad) or tolerance_rad <= 0):
            raise ValueError('Invalid diagnostic window or joint tolerance')
        self.limits = joint_limits
        self.tolerance = tolerance_rad
        self.window = deque(maxlen=stall_window_steps)
        self.observations = 0
        self.violations = {}
        self.max_tracking_error = dict.fromkeys(JOINT_NAMES, 0.)
        self.first_stall_step = None
        self.stall_steps = 0
        self.stall_active = False

    def observe(self, step, state, commands, transfer):
        if (len(state) != 28 or len(commands) != 4 or
                not all(math.isfinite(value) for value in (*state, *commands))):
            raise ValueError('Diagnostics require a finite 28D state and four commands')
        self.observations += 1
        for joint, position, command, (lower, upper) in zip(
                JOINT_NAMES, state[:4], commands, self.limits):
            self.max_tracking_error[joint] = max(self.max_tracking_error[joint], abs(command-position))
            excess = max(lower-position, position-upper, 0.)
            if excess > self.tolerance:
                violation = self.violations.setdefault(joint, {
                    'steps': 0, 'first_step': step, 'max_excess_rad': 0.,
                    'lower_limit_rad': lower, 'upper_limit_rad': upper,
                })
                violation['steps'] += 1
                violation['max_excess_rad'] = max(violation['max_excess_rad'], excess)
        self.window.append((tuple(state[:3]), transfer.get('payload_particles', 0),
                            transfer.get('retained_particles', 0)))
        # A loaded bucket with no transport motion or new deposits over the
        # window is a useful symptom, not a diagnosis or a termination rule.
        self.stall_active = (len(self.window) == self.window.maxlen and
            all(sample[2] > 0 for sample in self.window) and
            max(sample[1] for sample in self.window) == min(sample[1] for sample in self.window) and
            all(max(sample[0][axis] for sample in self.window) -
                min(sample[0][axis] for sample in self.window) <= self.tolerance for axis in range(3)))
        if self.stall_active:
            if self.first_stall_step is None:
                self.first_stall_step = step
            self.stall_steps += 1

    def summary(self):
        return {
            'schema_version': 'excavator_rollout_diagnostics_v1',
            'observations': self.observations,
            'joint_tolerance_rad': self.tolerance,
            'measured_joint_limit_violations': {key: dict(value) for key, value in self.violations.items()},
            'max_tracking_error_rad': dict(self.max_tracking_error),
            'retained_payload_stall': {
                'detected': self.first_stall_step is not None, 'active': self.stall_active,
                'first_step': self.first_stall_step, 'steps': self.stall_steps,
                'window_steps': self.window.maxlen,
            },
        }

    def warnings(self):
        result = [f'{joint}_outside_joint_limits' for joint in self.violations]
        if self.stall_active:
            result.append('retained_payload_stall')
        return result
