"""Explicit, bounded post-dig recovery outside the unchanged ACT weights."""
from collections import deque
from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class StallSettings:
    window_steps: int = 40
    earliest_trigger_step: int = 250
    minimum_fill: float = .001
    maximum_fill: float | None = None
    maximum_dig_distance_m: float = 2.5
    minimum_unload_distance_m: float = 5.
    maximum_parent_action_rad_s: float = .04
    maximum_swing_action_rad_s: float = .04
    maximum_bucket_action_rad_s: float = .12
    maximum_parent_span_rad: float = .08
    maximum_swing_span_rad: float = .10
    maximum_bucket_span_rad: float = .15
    recovery_step_budget: int = 120
    curl_dip_m: float = 0.


class PostDigStallDetector:
    """Use policy observations and raw commands; no scores, phases, or particle IDs."""
    def __init__(self, settings=None):
        self.settings = settings or StallSettings()
        self.window = deque(maxlen=self.settings.window_steps)
        self.step = -1

    def observe(self, state, predicted):
        state = np.asarray(state, dtype=float)
        action = np.asarray(predicted, dtype=float)
        if state.shape != (28,) or action.shape != (4,) or not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError('Stall detector needs finite 28D state and 4D action')
        self.window.append((state.copy(), action.copy()))
        self.step += 1
        if len(self.window) < self.settings.window_steps:
            return False, dict(ready=False)
        states = np.stack([x[0] for x in self.window])
        actions = np.stack([x[1] for x in self.window])
        span = np.ptp(states[:, :4], axis=0)
        s = self.settings
        details = dict(ready=True, current_fill=float(state[26]), minimum_window_fill=float(states[:, 26].min()),
            dig_distance_m=float(np.linalg.norm(state[18:21])),
            unload_distance_m=float(np.linalg.norm(state[21:24])), joint_span_rad=span.tolist(),
            mean_parent_action_rad_s=float(np.linalg.norm(actions[:, 1:3], axis=1).mean()),
            mean_swing_action_rad_s=float(np.abs(actions[:, 0]).mean()),
            mean_bucket_action_rad_s=float(np.abs(actions[:, 3]).mean()))
        checks = dict(timeout_reached=self.step >= s.earliest_trigger_step,
            filled=details['minimum_window_fill'] >= s.minimum_fill,
            low_fill=s.maximum_fill is None or details['current_fill'] <= s.maximum_fill,
            near_dig=details['dig_distance_m'] <= s.maximum_dig_distance_m,
            far_from_unload=details['unload_distance_m'] >= s.minimum_unload_distance_m,
            small_parent_action=details['mean_parent_action_rad_s'] <= s.maximum_parent_action_rad_s,
            small_swing_action=details['mean_swing_action_rad_s'] <= s.maximum_swing_action_rad_s,
            small_bucket_action=details['mean_bucket_action_rad_s'] <= s.maximum_bucket_action_rad_s,
            stationary_parents=bool((span[1:3] <= s.maximum_parent_span_rad).all()),
            stationary_swing=bool(span[0] <= s.maximum_swing_span_rad),
            stationary_bucket=bool(span[3] <= s.maximum_bucket_span_rad))
        return all(checks.values()), dict(**details, checks=checks)


class SecureLiftRecovery:
    """Close, lift and return to ACT; never turn toward the truck or unload."""
    def __init__(self, model, scene, state, dt=.1, dip_m=0.):
        from act_recovery_expert import RecoveryExpert
        from excavator_reachability import AuthoredExcavatorChain, TruckBedColumn
        self.chain = AuthoredExcavatorChain(model)
        self.scene = scene
        self.bed = TruckBedColumn(scene, .15)
        self.bounds = np.asarray(scene['canonical_joint_limits_rad'], dtype=float)
        self.dt = dt
        self.minimum_bucket = self.bounds[3, 0]
        self.closed_pitch = math.radians(-135)
        self.collision_vertices = np.unique(np.asarray(model['bucket_collision_vertices'], dtype=float), axis=0)
        self.initial_q = np.asarray(state[:4], dtype=float)
        if ((self.initial_q < self.bounds[:, 0] - .05) | (self.initial_q > self.bounds[:, 1] + .05)).any():
            raise ValueError('Measured joints already beyond the recovery planning bounds')
        self.minimum_bucket_z = lambda q: RecoveryExpert.minimum_bucket_z(self, q)
        self.path = RecoveryExpert.progressive_curl(self, self.initial_q, dip_m=dip_m)
        self.lift = RecoveryExpert.lift_goal(self, self.path[-1])
        samples = self.path[-1] + (self.lift - self.path[-1]) * np.linspace(0, 1, 51)[:, None]
        if min(self.minimum_bucket_z(q) for q in samples) < .005:
            raise ValueError('Secure-to-lift interpolation crosses the ground')
        self.velocity = lambda goal, command, speeds: RecoveryExpert.velocity(self, goal, command, speeds)
        self.phase = 'unwind'
        self.waypoint = 0
        self.steps = 0
        self.empty_observations = 0
        self.finished = False
        self.initial_tip = self.chain.point(self.initial_q, 'tip')

    def action(self, state, command, landmarks):
        q, command = np.asarray(state[:4]), np.asarray(command)
        for name in ('tip', 'load', 'pour'):
            if np.linalg.norm(self.chain.point(q, name) - np.asarray(landmarks[name])) > .02:
                raise ValueError('Recovery FK differs from actual bucket landmarks')
        self.empty_observations = self.empty_observations + 1 if state[26] < .0003 else 0
        if self.empty_observations >= 3:
            self.finished = True
            return None, dict(phase='aborted_empty', handback=True, steps=self.steps)
        if self.phase == 'unwind' and abs(command[3] - q[3]) <= .09:
            self.phase = 'curl'
        if self.phase == 'curl' and np.max(np.abs(q - self.path[self.waypoint])) < .035:
            self.waypoint += 1
            if self.waypoint == len(self.path):
                self.phase = 'lift'
        if self.phase == 'lift' and np.max(np.abs(q - self.lift)) < .04:
            self.finished = True
            return None, dict(phase='completed_secure_lift', handback=True, steps=self.steps,
                tip_rise_m=float(self.chain.point(q, 'tip')[2] - self.initial_tip[2]),
                carry_pitch_deg=math.degrees(float(self.chain.carry_pitch(q))))
        if self.phase == 'unwind':
            goal = q.copy(); goal[3] = max(self.minimum_bucket, q[3] - .05)
            speeds = np.array([.2, .2, .2, 4.])
        elif self.phase == 'curl':
            goal = self.path[self.waypoint]
            speeds = np.array([.15, .18, .18, .25])
        else:
            goal = self.lift
            speeds = np.array([.28, .24, .24, .4])
        self.steps += 1
        return self.velocity(goal, command, speeds).astype(np.float32), dict(phase=self.phase,
            steps=self.steps, waypoint=self.waypoint, goal_q=goal.tolist(), handback=False)


class SecureAlignRecovery:
    """Bridge a detected stall through contained truck alignment; ACT unloads."""
    def __init__(self, model, scene, state, dt=.1, dip_m=0.):
        self.model, self.scene, self.dt = model, scene, dt
        self.secure_lift = SecureLiftRecovery(model, scene, state, dt, dip_m=dip_m)
        self.lift = self.secure_lift.lift
        self.transit = None
        self.steps = 0
        self.finished = False

    def action(self, state, command, landmarks):
        from act_recovery_expert import RecoveryExpert, RecoveryFailure
        if not self.secure_lift.finished:
            action, details = self.secure_lift.action(state, command, landmarks)
            if action is not None:
                self.steps += 1
                return action, dict(details, steps=self.steps, stage='secure_lift')
            if details['phase'] != 'completed_secure_lift':
                self.finished = True
                return None, details
        if self.transit is None:
            try:
                self.transit = RecoveryExpert(self.model, self.scene, state, self.dt)
                self.transit.phase = 'slew'
                self.transit.phase_step = 0
            except (ValueError, RuntimeError) as error:
                self.finished = True
                return None, dict(phase='alignment_plan_unresolved', handback=True, error=str(error), steps=self.steps)
        # The geometry controller only needs an empty/loaded decision. It uses
        # the policy fill observation, without access to scored particle IDs.
        observed_loaded = bool(state[26] >= .0003)
        observed_particles = dict(retained_particles=100 if observed_loaded else 0, payload_particles=0)
        try:
            action, details = self.transit.action(state, command, observed_particles, landmarks)
        except RecoveryFailure as error:
            self.finished = True
            return None, dict(phase='aborted_empty_or_stalled', handback=True, error=str(error), steps=self.steps)
        details.pop('measured_retained', None)
        if details['phase'] == 'open':
            # Discard the computed opening command. The final closed pose and
            # its opening arc have been checked; opening belongs to ACT.
            self.finished = True
            residual = float(np.linalg.norm(self.transit.bed.residual(
                self.transit.opening_points(np.asarray(state[:4]))), axis=1).max())
            return None, dict(phase='completed_secure_align', handback=True, steps=self.steps,
                closed_q=np.asarray(state[:4]).tolist(), contained_opening_arc_residual_m=residual)
        self.steps += 1
        return action, dict(details, phase=details['phase'], steps=self.steps, stage='transit_align', handback=False,
                            loaded_from_policy_fill=observed_loaded)


def mode_settings(mode='secure_lift'):
    if mode == 'secure_lift':
        return StallSettings()
    if mode == 'secure_align':
        return StallSettings(earliest_trigger_step=0, maximum_fill=.05, recovery_step_budget=180)
    if mode == 'secure_align_scoop':
        return StallSettings(earliest_trigger_step=0, maximum_fill=.05, recovery_step_budget=200, curl_dip_m=.35)
    raise ValueError('Unknown post-dig recovery mode')


def settings_dict(mode='secure_lift'):
    return asdict(mode_settings(mode))


def make_stall_guard_class(base_guard, model, mode='secure_lift'):
    class StallRecoveryGuard(base_guard):
        def __init__(self, scene, *args, **kwargs):
            super().__init__(scene, *args, **kwargs)
            self.scene = scene
            self.detector = PostDigStallDetector(mode_settings(mode))
            self.recovery = None
            self.attempted = False

        def update(self, state, action, command_positions, landmarks):
            triggered, detection = self.detector.observe(state, action)
            trace = dict(enabled=True, attempted=self.attempted, triggered=False, override=False, handback=False)
            if triggered and not self.attempted:
                self.attempted = True
                trace.update(triggered=True, detection=detection)
                try:
                    recovery_class = SecureLiftRecovery if mode == 'secure_lift' else SecureAlignRecovery
                    self.recovery = recovery_class(model, self.scene, state, self.dt,
                                                   dip_m=self.detector.settings.curl_dip_m)
                    trace['planned_lift_q'] = self.recovery.lift.tolist()
                except (ValueError, RuntimeError) as error:
                    trace['planning_error'] = str(error)
            chosen = np.asarray(action)
            if self.recovery is not None and not self.recovery.finished:
                if self.recovery.steps >= self.detector.settings.recovery_step_budget:
                    self.recovery.finished = True
                    trace.update(phase='aborted_budget', handback=True)
                else:
                    replacement, details = self.recovery.action(state, command_positions, landmarks)
                    trace.update(details)
                    if replacement is not None:
                        chosen = replacement
                        trace['override'] = True
                        trace['recovery_command'] = replacement.tolist()
            closure_phase = trace.get('phase') in ('unwind', 'curl') and trace.get('override')
            if closure_phase:
                # Its intermediate targets deliberately move from the current
                # pitch to the closed pitch. Activating the carry constraint
                # early would clamp to the final pitch and fight that path.
                applied = np.asarray(chosen, dtype=np.float32)
                guard = dict(enabled=True, active=False, released=self.released,
                    modified=False, infeasible=False, suspended_for_progressive_closure=True,
                    phase_constraint='Validated closure path with ground/command-limit checks')
            else:
                applied, guard = super().update(state, chosen, command_positions, landmarks)
            guard['stall_recovery'] = trace
            return applied, guard
    return StallRecoveryGuard
