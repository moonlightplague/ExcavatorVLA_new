"""Conservative release-workspace bounds and positive endpoint IK witnesses.

IK failure is unresolved, never proof that a scene is physically impossible.
Endpoints do not establish collision-free paths or successful particle transfer.
"""
import math

import numpy as np


class AuthoredExcavatorChain:
    def __init__(self, model):
        if model.get('schema') != 'excavator_authored_chain_v1':
            raise ValueError('Expected authored, source-pinned joint geometry')
        self.base = np.asarray(model['base_world'], dtype=float)
        self.joints = model['joints']
        self.landmarks = {key: np.asarray(value, dtype=float) for key, value in model['landmarks'].items()}
        if (set(self.landmarks) != {'tip', 'load', 'pour'}
                or any(point.shape != (3,) or not np.isfinite(point).all() for point in self.landmarks.values())):
            raise ValueError('Expected the three finite deployed landmarks')
        if [joint['name'] for joint in self.joints] != ['swing', 'boom', 'arm', 'bucket']:
            raise ValueError('Expected canonical deployed joint order')
        self.parents = np.asarray([joint['parent_from_joint'] for joint in self.joints], dtype=float)
        children = np.asarray([joint['child_from_joint'] for joint in self.joints], dtype=float)
        if (self.base.shape != (4, 4) or self.parents.shape != (4, 4, 4)
                or children.shape != (4, 4, 4)
                or not all(np.isfinite(value).all() for value in (self.base, self.parents, children))):
            raise ValueError('Invalid joint frames')
        for frame in (self.base, *self.parents, *children):
            if (not np.allclose(frame[3], [0., 0., 0., 1.], atol=1e-7)
                    or not np.allclose(frame[:3, :3].T @ frame[:3, :3], np.eye(3), atol=1e-6)):
                raise ValueError('Workspace bounds require rigid, unscaled joint frames')
        self.child_inverse = np.linalg.inv(children)
        self.origin = self.base[:3, 3]
        self.swing_anchor = (self.base @ self.parents[0])[:3, 3]
        self.swing_axis = (self.base @ self.parents[0])[:3, 2]
        self.zero_heading = math.atan2((self.base @ self.parents[0])[1, 0], (self.base @ self.parents[0])[0, 0])
        if self.joints[0]['axis'] != 'Z' or not np.allclose(self.swing_axis, [0., 0., 1.], atol=1e-5):
            raise ValueError('The diagnostic requires the deployed vertical swing axis')

    def bucket_transform(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape[-1:] != (4,) or not np.isfinite(q).all():
            raise ValueError('Expected finite canonical joint positions')
        result = np.broadcast_to(self.base, (*q.shape[:-1], 4, 4)).copy()
        for index, joint in enumerate(self.joints):
            rotation = np.broadcast_to(np.eye(4), result.shape).copy()
            axis = {'X': 0, 'Y': 1, 'Z': 2}[joint['axis']]
            first, second = (axis + 1) % 3, (axis + 2) % 3
            cosine, sine = np.cos(q[..., index]), np.sin(q[..., index])
            rotation[..., first, first] = rotation[..., second, second] = cosine
            rotation[..., first, second], rotation[..., second, first] = -sine, sine
            result = result @ self.parents[index] @ rotation @ self.child_inverse[index]
        return result

    def point(self, q, name='release'):
        local = ((self.landmarks['load'] + self.landmarks['pour']) / 2
                 if name == 'release' else self.landmarks[name])
        return (self.bucket_transform(q) @ np.r_[local, 1.])[..., :3]

    def carry_pitch(self, q):
        transform = self.bucket_transform(q)
        vector = (transform[..., :3, :3] @ (self.landmarks['pour'] - self.landmarks['load']))
        heading = self.zero_heading + np.asarray(q)[..., 0]
        horizontal = np.cos(heading) * vector[..., 0] + np.sin(heading) * vector[..., 1]
        observed = np.arctan2(vector[..., 2], horizontal)
        total = np.asarray(q)[..., 1:4].sum(-1)
        return total + np.arctan2(np.sin(observed-total), np.cos(observed-total))

    def release_radius_upper_bound(self):
        # Triangle inequality deliberately ignores all joint limits. This is
        # an outer bound, so a numerical IK miss cannot shrink the workspace.
        radius = float(np.linalg.norm(self.child_inverse[0, :3, 3]))
        for index in range(1, 4):
            radius += float(np.linalg.norm(self.parents[index, :3, 3]))
            radius += float(np.linalg.norm(self.child_inverse[index, :3, 3]))
        return radius + max(float(np.linalg.norm(point)) for point in self.landmarks.values())

    def shoulder_envelope(self):
        shoulder = (self.base @ self.parents[0] @ self.child_inverse[0] @ self.parents[1])[:3, 3]
        arm_radius = sum(float(np.linalg.norm(self.child_inverse[index, :3, 3])) for index in range(1, 4))
        arm_radius += sum(float(np.linalg.norm(self.parents[index, :3, 3])) for index in (2, 3))
        arm_radius += max(float(np.linalg.norm(point)) for point in self.landmarks.values())
        return dict(pivot_circle_radius_m=float(np.linalg.norm((shoulder-self.swing_anchor)[:2])),
                    pivot_height_m=float(shoulder[2]), arm_radius_upper_bound_m=arm_radius)


class TruckBedColumn:
    def __init__(self, scene, margin_m=.1):
        bed = scene['truck_counting_volume']
        self.to_local = np.asarray(bed['local_from_world'], dtype=float)
        self.minimum = np.asarray(bed['local_min'], dtype=float)
        self.maximum = np.asarray(bed['local_max'], dtype=float)
        if (self.minimum.shape != (3,) or self.maximum.shape != (3,)
                or not np.isfinite([self.minimum, self.maximum]).all() or not 0 <= margin_m < math.inf):
            raise ValueError('Invalid finite bed bounds or margin')
        self.to_world = np.linalg.inv(self.to_local)
        # Row-vector USD convention: rows of local-to-world are local axes.
        axes = self.to_world[:3, :3]
        self.scales = np.linalg.norm(axes, axis=1)
        if (self.to_local.shape != (4, 4) or not np.isfinite(self.to_local).all()
                or not np.all(self.maximum > self.minimum)
                or not np.allclose((axes @ axes.T), np.diag(self.scales**2), atol=1e-5)):
            raise ValueError('Expected a finite, nonsheared truck counting box')
        self.margin = float(margin_m) / self.scales
        if np.any(self.maximum[:2] - self.minimum[:2] <= 2 * self.margin[:2]):
            raise ValueError('Truck inset consumes the entire bed')

    def local(self, points):
        points = np.asarray(points)
        return points @ self.to_local[:3, :3] + self.to_local[3, :3]

    def residual(self, points):
        local = self.local(points)
        low = self.minimum + self.margin
        high = self.maximum - self.margin
        xy = np.maximum(low[:2] - local[..., :2], 0.) + np.minimum(high[:2] - local[..., :2], 0.)
        z = np.maximum(self.maximum[2] + self.margin[2] - local[..., 2], 0.)
        return np.concatenate((xy, z[..., None]), axis=-1) * self.scales

    def distance_lower_bound(self, anchor):
        # Distance to the full bed column, without an inset. Safe to use for
        # exclusion under the explicit above-bed, gravity-release convention.
        local = self.local(anchor)
        nearest = local.copy()
        nearest[:2] = np.clip(local[:2], self.minimum[:2], self.maximum[:2])
        nearest[2] = max(local[2], self.maximum[2])
        return float(np.linalg.norm((nearest-local) * self.scales))

    def candidate_xy(self, target):
        points = [np.asarray(target)[:2]]
        for x in np.linspace(self.minimum[0]+self.margin[0], self.maximum[0]-self.margin[0], 5):
            for y in np.linspace(self.minimum[1]+self.margin[1], self.maximum[1]-self.margin[1], 3):
                points.append((np.array([x, y, self.maximum[2], 1.]) @ self.to_world)[:2])
        return points

    def top_corners_world(self):
        return np.asarray([np.array([x, y, self.maximum[2], 1.]) @ self.to_world
                           for x in (self.minimum[0], self.maximum[0])
                           for y in (self.minimum[1], self.maximum[1])])[:, :3]


def restored_truck_geometry(scene, model):
    """Restore old missing matrices from authored transforms and recorded pose.

    Do not infer an oriented counting box from its world AABB. No source scene
    files are changed. Existing matrices are checked against this reconstruction.
    """
    import copy
    restored = copy.deepcopy(scene)
    profile, template = scene['scene_profile'], model['truck_template']
    yaw = math.radians(profile.get('actual_truck_yaw_deg', profile['truck_yaw_deg']) - template['root_yaw_deg'])
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, s, 0., 0.], [-s, c, 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    untranslate, translate = np.eye(4), np.eye(4)
    untranslate[3, :3] = -np.asarray(template['root_xyz'])
    translate[3, :3] = profile['truck_translation_xyz']
    bed_to_world = np.asarray(template['bed_world_row_matrix']) @ untranslate @ rotation @ translate
    inverse = np.linalg.inv(bed_to_world)
    current = scene['truck_counting_volume'].get('local_from_world')
    if current is not None and not np.allclose(current, inverse, rtol=0, atol=2e-6):
        raise ValueError('Authored truck reconstruction disagrees with recorded exact transform')
    restored['truck_counting_volume']['local_from_world'] = inverse.tolist()
    return restored


def workspace_bound(chain, bed, target):
    radius = chain.release_radius_upper_bound()
    distance = bed.distance_lower_bound(chain.swing_anchor)
    # Target height is not a required release height; use XY alone for this
    # deliberately weak but safe target-column rejection test.
    target_xy_distance = float(np.linalg.norm(np.asarray(target)[:2] - chain.swing_anchor[:2]))
    envelope = chain.shoulder_envelope()
    corners = bed.top_corners_world()
    # AABB distance is weaker than oriented-box distance, which is deliberate
    # for an exclusion proof. The real box can only be farther away.
    minimum, maximum = corners[:, :2].min(0), corners[:, :2].max(0)
    excess = np.maximum(np.maximum(minimum-chain.swing_anchor[:2], chain.swing_anchor[:2]-maximum), 0.)
    bed_xy_lower = float(np.linalg.norm(excess))
    if bed.to_world[2, 2] <= 0:
        raise ValueError('Expected the truck bed local up-axis to point upward')
    height_lower = max(0., float(corners[:, 2].min()) - envelope['pivot_height_m'])
    def shoulder_distance(xy):
        return math.hypot(max(0., xy-envelope['pivot_circle_radius_m']), height_lower)
    bed_shoulder_lower = shoulder_distance(bed_xy_lower)
    target_shoulder_lower = shoulder_distance(target_xy_distance)
    arm_radius = envelope['arm_radius_upper_bound_m']
    return dict(scope='current release landmarks above the actual bed, not ballistic throwing',
        swing_anchor_xyz=chain.swing_anchor.tolist(), radius_upper_bound_m=radius,
        bed_column_distance_lower_bound_m=distance,
        target_xy_distance_m=target_xy_distance,
        shoulder_envelope=envelope, minimum_bed_top_z_m=float(corners[:, 2].min()),
        bed_shoulder_distance_lower_bound_m=bed_shoulder_lower,
        target_shoulder_distance_lower_bound_m=target_shoulder_lower,
        bed_column_certified_outside=bool(distance > radius + .02 or bed_shoulder_lower > arm_radius + .02),
        target_column_certified_outside=bool(target_xy_distance > radius + .02 or target_shoulder_lower > arm_radius + .02))


def find_endpoint(chain, scene, kind, *, target_only=False, max_starts=8):
    """Return a verified positive witness, or unresolved; never negative IK proof."""
    from scipy.optimize import least_squares

    if max_starts < 1:
        raise ValueError('A positive, bounded search budget is required')
    bed = TruckBedColumn(scene)
    target = np.asarray(scene['observation_context']['unload_landing_xyz'], dtype=float)
    bounds = np.asarray(scene['canonical_joint_limits_rad'], dtype=float).copy()
    if bounds.shape != (4, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError('Invalid captured joint limits')
    if kind == 'carry':
        bounds[3, 1] = min(bounds[3, 1], -1.3)
    elif kind == 'dump':
        bounds[3, 0] = max(bounds[3, 0], -.8)
    else:
        raise ValueError('Expected carry or dump endpoint')
    if bounds[3, 1] <= bounds[3, 0]:
        return dict(status='unresolved', reason='empty_endpoint_joint_range')

    def residual(q):
        points = chain.point(q)
        error = bed.residual(points)
        if target_only:
            error[..., :2] = points[..., :2] - target[:2]
        if kind == 'carry':
            # Metres/radians have equal numerical weight only in the search.
            # Every accepted witness is checked against both tolerances below.
            pitch = np.maximum(chain.carry_pitch(q) - math.radians(-130), 0.)
            error = np.concatenate((error, pitch[..., None]), axis=-1)
        return error

    axes = [np.linspace(*bounds[index], count) for index, count in ((1, 17), (2, 21), (3, 9))]
    grid = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    base = np.column_stack((np.zeros(len(grid)), grid))
    positions = chain.point(base)
    offset_bearing = np.arctan2(positions[:, 1]-chain.swing_anchor[1], positions[:, 0]-chain.swing_anchor[0])
    candidates = []
    for xy in ([target[:2]] if target_only else bed.candidate_xy(target)):
        q = base.copy()
        bearing = math.atan2(xy[1]-chain.swing_anchor[1], xy[0]-chain.swing_anchor[0])
        q[:, 0] = (bearing-offset_bearing+math.pi) % (2*math.pi) - math.pi
        q = q[(q[:, 0] >= bounds[0, 0]) & (q[:, 0] <= bounds[0, 1])]
        if len(q):
            score = np.linalg.norm(residual(q), axis=-1)
            candidates.extend((float(score[index]), q[index].copy()) for index in np.argsort(score)[:max_starts])
    candidates.sort(key=lambda item: item[0])
    best = None
    tried = []
    for score, q in candidates:
        if any(np.linalg.norm(q-old) < .2 for old in tried):
            continue
        tried.append(q.copy())
        if score > .001:
            result = least_squares(residual, q, bounds=(bounds[:, 0], bounds[:, 1]),
                                   max_nfev=100, ftol=1e-8, xtol=1e-8, gtol=1e-8)
            q = result.x
        errors = residual(q)
        best = float(np.linalg.norm(errors)) if best is None else min(best, float(np.linalg.norm(errors)))
        spatial = float(np.linalg.norm(errors[:3]))
        pitch_error = float(errors[3]) if kind == 'carry' else 0.
        if spatial <= .005 and pitch_error <= 1e-4:
            return dict(status='endpoint_found_path_unverified', kind=kind, target_only=target_only,
                q_rad=q.tolist(), release_xyz=chain.point(q).tolist(),
                spatial_constraint_error_m=spatial, carry_pitch_deg=float(np.degrees(chain.carry_pitch(q))),
                collision_checked=False, path_checked=False, transfer_checked=False)
        if len(tried) >= max_starts:
            break
    return dict(status='unresolved', kind=kind, target_only=target_only, starts=len(tried),
                best_search_residual=best, reason='finite_search_without_a_positive_witness')
