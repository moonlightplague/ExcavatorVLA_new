"""Optional ACT carry-attitude constraint; this is an explicitly hybrid policy."""
from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class CarryGuardSettings:
    max_carry_pitch_deg: float = -130.
    loaded_fill_fraction: float = .01
    empty_fill_fraction: float = .002
    closed_bucket_rad: float = -1.3
    min_tip_clearance_m: float = .25
    swing_trigger_rad_s: float = .08
    bed_margin_m: float = .1
    release_observations: int = 3
    parent_projection: str = 'balanced'
    release_geometry: str = 'load_point'
    bucket_open_interlock: bool = False
    near_bed_relax_distance_m: float = 0.
    near_bed_max_pitch_deg: float = -130.
    opening_pose_near_edge_m: float = .3
    opening_pose_target_bucket_rad: float = -.8


class CarryAttitudeGuard:
    def __init__(self, scene, dt=.1, action_limits=(6., 2.5, 2.5, 6.), settings=None):
        self.settings = settings or CarryGuardSettings()
        self.dt = float(dt)
        self.action_limits = np.asarray(action_limits, dtype=np.float64)
        self.joint_limits = np.asarray(scene['canonical_joint_limits_rad'], dtype=np.float64)
        bed = scene['truck_counting_volume']
        self.local_from_world = np.asarray(bed['local_from_world'], dtype=np.float64)
        self.bed_min = np.asarray(bed['local_min'], dtype=np.float64)
        self.bed_max = np.asarray(bed['local_max'], dtype=np.float64)
        world_from_local = np.linalg.inv(self.local_from_world)
        self.bed_scales = np.linalg.norm(world_from_local[:3, :3], axis=1)
        self.target = np.asarray(scene['observation_context']['unload_landing_xyz'], dtype=np.float64)
        if (self.local_from_world.shape != (4, 4) or self.joint_limits.shape != (4, 2)
                or self.action_limits.shape != (4,) or not self.dt > 0
                or not np.isfinite(self.local_from_world).all() or np.any(self.bed_scales <= 0)):
            raise ValueError('Invalid carry guard geometry or action contract')
        self.active = False
        self.released = False
        self.release_count = 0
        self.empty_count = 0

    def release_geometry(self, state, landmarks):
        # These points are the same current rigid-body landmarks used to form
        # the policy's pour-axis and unload-offset observations, not expert data.
        points = np.asarray([landmarks['load'], landmarks['pour']], dtype=np.float64)
        if points.shape != (2, 3) or not np.isfinite(points).all():
            raise ValueError('Carry guard requires finite current bucket landmarks')
        # current-v4 records zero fixed-base heading; canonical swing is the
        # current upper heading. Recorded reference profiles need no optional
        # randomized-scene yaw fields for this observation consistency check.
        heading = float(state[0])
        c, s = math.cos(heading), math.sin(heading)
        delta = np.array([c * state[21] - s * state[22],
                          s * state[21] + c * state[22], state[23]])
        if not np.allclose(points[0], self.target - delta, rtol=0, atol=.02):
            raise ValueError('Bucket geometry and the current 28D observation disagree')
        local = (np.column_stack((points, np.ones(2))) @ self.local_from_world)[:, :3]
        margin = self.settings.bed_margin_m / self.bed_scales
        within = ((local[:, :2] >= self.bed_min[:2] + margin[:2]).all(1) &
                  (local[:, :2] <= self.bed_max[:2] - margin[:2]).all(1) &
                  (local[:, 2] >= self.bed_max[2] + margin[2]))
        if self.settings.release_geometry == 'load_point':
            # Future opening geometry is not current cargo arrival.
            return bool(within[0]), local.tolist()
        if self.settings.release_geometry == 'load_and_pour':
            # Keep the mouth inside the bed as well as the current load point;
            # a load point alone can be over the rim while the pour point is out.
            return bool(within.all()), local.tolist()
        if self.settings.release_geometry == 'opening_pose' and not within.any():
            # Opening moves both the load and pour landmarks about the bucket
            # pivot. Constants match the existing rigid-body landmarks in the
            # simulator; infer its current transform from measured geometry.
            excess = np.maximum(np.maximum(self.bed_min[:2] - local[0, :2],
                                           local[0, :2] - self.bed_max[:2]), 0.)
            edge_distance = np.linalg.norm(excess * self.bed_scales[:2])
            if edge_distance <= self.settings.opening_pose_near_edge_m:
                projected = opening_pour_point(
                    state, landmarks, target_bucket_rad=self.settings.opening_pose_target_bucket_rad)
                projected_local = (np.r_[projected, 1.] @ self.local_from_world)[:3]
                if ((projected_local[:2] >= self.bed_min[:2]).all() and
                        (projected_local[:2] <= self.bed_max[:2]).all() and
                        projected_local[2] >= self.bed_max[2] + margin[2]):
                    return True, local.tolist()
        return bool(within.any()), local.tolist()

    def update(self, state, action, command_positions, landmarks):
        state = np.asarray(state, dtype=np.float64)
        action = np.asarray(action, dtype=np.float64)
        command = np.asarray(command_positions, dtype=np.float64)
        if (state.shape != (28,) or action.shape != (4,) or command.shape != (4,)
                or not all(np.isfinite(value).all() for value in (state, action, command))):
            raise ValueError('Invalid carry guard state/action')
        if np.any(np.abs(action) > self.action_limits + 1e-6):
            raise ValueError('Clip policy actions to original limits before the guard')
        fill = float(state[26])
        over_bed, local_points = self.release_geometry(state, landmarks)
        rearmed = False
        if (self.settings.release_geometry in ('load_point', 'load_and_pour') and self.released
                and fill >= self.settings.empty_fill_fraction):
            load = np.asarray(local_points[0])
            inside_actual_column = ((load[:2] >= self.bed_min[:2]).all()
                and (load[:2] <= self.bed_max[:2]).all() and load[2] >= self.bed_max[2])
            # The un-inset box provides hysteresis. This landmark is a current
            # rigid-body reference, not a measured sand centre of mass.
            if not inside_actual_column:
                self.released = False
                self.active = True
                self.release_count = 0
                rearmed = True
        turning = abs(action[0]) >= self.settings.swing_trigger_rad_s or abs(state[4]) >= .08
        lifting = np.linalg.norm(action[1:3]) >= .12
        if (not self.released and fill >= self.settings.loaded_fill_fraction
                and state[3] <= self.settings.closed_bucket_rad
                and -state[20] >= self.settings.min_tip_clearance_m and (turning or lifting)):
            self.active = True
        self.release_count = self.release_count + 1 if over_bed and self.active else 0
        if self.release_count >= self.settings.release_observations:
            self.released = True
            self.active = False
        self.empty_count = self.empty_count + 1 if fill < self.settings.empty_fill_fraction else 0
        if self.empty_count >= 3:
            self.active = False
        phi_observed = math.atan2(float(state[17]), float(state[16]))
        total_q = float(state[1:4].sum())
        phi = total_q + math.atan2(math.sin(phi_observed - total_q), math.cos(phi_observed - total_q))
        lower = np.maximum(self.joint_limits[:, 0], command - self.dt * self.action_limits)
        upper = np.minimum(self.joint_limits[:, 1], command + self.dt * self.action_limits)
        target = np.clip(command + self.dt * action, self.joint_limits[:, 0], self.joint_limits[:, 1])
        bucket_open_blocked = False
        if self.active and not self.released and self.settings.bucket_open_interlock:
            # Keep the integrated target closed until release. If a previous
            # action chunk already accumulated an open target, unwind it at
            # the original bucket velocity limit.
            held_target = min(target[3], self.settings.closed_bucket_rad)
            held_target = max(held_target, command[3] - self.dt * self.action_limits[3])
            bucket_open_blocked = held_target < target[3] - 1e-9
            target[3] = held_target
        predicted_phi = phi + float((target[1:4] - state[1:4]).sum())
        limit_deg = float(self.settings.max_carry_pitch_deg)
        nearest_bed_gap = 0.
        if self.settings.near_bed_relax_distance_m > 0:
            points = np.asarray(local_points, dtype=np.float64)[:, :2]
            excess = np.maximum(np.maximum(self.bed_min[:2] - points,
                                           points - self.bed_max[:2]), 0.)
            nearest_bed_gap = float(np.min(np.linalg.norm(excess * self.bed_scales[:2], axis=1)))
            near_fraction = np.clip(1. - nearest_bed_gap / self.settings.near_bed_relax_distance_m, 0., 1.)
            limit_deg = ((1. - near_fraction) * limit_deg
                         + near_fraction * float(self.settings.near_bed_max_pitch_deg))
        details = dict(enabled=True, active=self.active, released=self.released, over_bed=over_bed,
            rearmed=rearmed,
            bucket_open_blocked=bucket_open_blocked,
            release_observations=self.release_count, fill_fraction=fill, pitch_rad=phi,
            predicted_pitch_rad=predicted_phi, limit_rad=math.radians(limit_deg),
            nearest_bed_gap_m=nearest_bed_gap, effective_limit_deg=limit_deg,
            local_load_and_pour=local_points, modified=False, infeasible=False)
        if not self.active or self.released or predicted_phi <= details['limit_rad']:
            if bucket_open_blocked:
                adjusted = action.copy()
                adjusted[3] = (target[3] - command[3]) / self.dt
                adjusted = np.clip(adjusted, -self.action_limits, self.action_limits)
                details['modified'] = bool(np.max(np.abs(adjusted-action)) > 1e-7)
                details['corrected_pitch_rad'] = predicted_phi
                return adjusted.astype(np.float32), details
            return action.astype(np.float32), details
        need = predicted_phi - details['limit_rad']
        # Use remaining bucket curl travel first. At its real joint limit,
        # constrain parent-joint motion too; zero bucket velocity cannot hold
        # world attitude while boom/arm angles continue changing.
        curl = min(need, max(0., target[3] - lower[3]))
        target[3] -= curl
        need -= curl
        if self.settings.parent_projection == 'arm_first':
            for joint in (2, 1):
                change = min(need, max(0., target[joint] - lower[joint]))
                target[joint] -= change
                need -= change
        movable = [1, 2]
        while need > 1e-12 and movable:
            share = need / len(movable)
            removed = 0.
            remaining = []
            for joint in movable:
                available = max(0., target[joint] - lower[joint])
                change = min(available, share)
                target[joint] -= change
                removed += change
                if available - change > 1e-12:
                    remaining.append(joint)
            need -= removed
            movable = remaining
        adjusted = action.copy()
        adjusted[1:4] = (target[1:4] - command[1:4]) / self.dt
        adjusted = np.clip(adjusted, -self.action_limits, self.action_limits)
        details.update(modified=bool(np.max(np.abs(adjusted-action)) > 1e-7),
                       infeasible=bool(need > 1e-6),
                       corrected_pitch_rad=phi + float((target[1:4] - state[1:4]).sum()))
        return adjusted.astype(np.float32), details


def opening_pour_point(state, landmarks, target_bucket_rad):
    tip, load, pour = [np.asarray(landmarks[key], dtype=np.float64) for key in ('tip', 'load', 'pour')]
    # Infer the actual rigid-body axes from all three authored landmarks.
    # Local Z is the bucket hinge axis, not world vertical.
    x_axis, z_axis = np.linalg.solve(np.array([[.4, -.26], [.5, .22]]),
                                     np.stack([tip-load, pour-load]))
    z_length = np.linalg.norm(z_axis)
    if z_length < 1e-6 or not np.isfinite(x_axis).all():
        raise ValueError('Degenerate bucket geometry')
    axis = z_axis / z_length
    pivot = load - .35 * x_axis - .08 * z_axis
    delta = max(0., float(target_bucket_rad) - float(state[3]))
    rotated_x = (math.cos(delta) * x_axis + math.sin(delta) * np.cross(axis, x_axis)
                 + axis * np.dot(axis, x_axis) * (1-math.cos(delta)))
    return pivot + .85 * rotated_x + .30 * z_axis


def settings_dict(parent_projection='balanced', release_geometry='load_point', max_carry_pitch_deg=-130.,
                  near_bed_relax_distance_m=0., opening_pose_near_edge_m=.3,
                  opening_pose_target_bucket_rad=-.8, bucket_open_interlock=False):
    return asdict(CarryGuardSettings(parent_projection=parent_projection, release_geometry=release_geometry,
        max_carry_pitch_deg=max_carry_pitch_deg, near_bed_relax_distance_m=near_bed_relax_distance_m,
        opening_pose_near_edge_m=opening_pose_near_edge_m,
        opening_pose_target_bucket_rad=opening_pose_target_bucket_rad,
        bucket_open_interlock=bucket_open_interlock))
