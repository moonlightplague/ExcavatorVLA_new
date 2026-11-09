"""Physical state capture for a private, paused PhysX unloading experiment.

This does not serialize PhysX contact caches or the renderer. Both treatment
arms must be restored, and repeated identical-arm controls measure the residual
simulation variation. It is never part of a production episode or its score.
"""
import hashlib

import numpy as np


def array_hash(array):
    value = np.ascontiguousarray(array)
    return hashlib.sha256(str(value.shape).encode() + value.dtype.str.encode() + value.tobytes()).hexdigest()


def capture(robot, particles):
    position, orientation = robot.get_world_pose()
    result = dict(position=np.asarray(position).copy(), orientation=np.asarray(orientation).copy(),
        joint_positions=np.asarray(robot.get_joint_positions()).copy(),
        joint_velocities=np.asarray(robot.get_joint_velocities()).copy(),
        linear_velocity=np.asarray(robot.get_linear_velocity()).copy(),
        angular_velocity=np.asarray(robot.get_angular_velocity()).copy(),
        particle_positions=np.asarray(particles.GetPointsAttr().Get(), dtype=np.float32).copy(),
        particle_velocities=np.asarray(particles.GetVelocitiesAttr().Get(), dtype=np.float32).copy())
    for name, value in result.items():
        if not np.isfinite(value).all():
            raise ValueError(f'Nonfinite snapshot field: {name}')
    if result['particle_positions'].shape != result['particle_velocities'].shape:
        raise ValueError('Particle velocities must be written back for every particle')
    return result


def restore(robot, particles, snapshot, command, indices, action_class, sync, update):
    from pxr import Vt
    robot.set_world_pose(position=snapshot['position'], orientation=snapshot['orientation'])
    robot.set_joint_positions(snapshot['joint_positions'])
    robot.set_joint_velocities(snapshot['joint_velocities'])
    robot.set_linear_velocity(snapshot['linear_velocity'])
    robot.set_angular_velocity(snapshot['angular_velocity'])
    robot.apply_action(action_class(joint_positions=np.asarray(command, dtype=np.float32), joint_indices=indices))
    particles.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(snapshot['particle_positions']))
    particles.GetVelocitiesAttr().Set(Vt.Vec3fArray.FromNumpy(snapshot['particle_velocities']))
    sync()
    # Kit commits the attribute updates while the timeline remains paused.
    for _ in range(3):
        update()
    actual = capture(robot, particles)
    errors = {name: float(np.max(np.abs(actual[name] - expected)))
              for name, expected in snapshot.items()}
    # Quaternions may change sign without changing orientation.
    errors['orientation'] = float(min(np.max(np.abs(actual['orientation']-snapshot['orientation'])),
                                      np.max(np.abs(actual['orientation']+snapshot['orientation']))))
    tolerance = dict(position=1e-5, orientation=1e-5, joint_positions=1e-5,
                     joint_velocities=1e-5, linear_velocity=1e-5, angular_velocity=1e-5,
                     particle_positions=0., particle_velocities=0.)
    if any(errors[name] > tolerance[name] for name in errors):
        raise ValueError(f'Snapshot restoration mismatch: {errors}')
    return dict(max_abs_errors=errors, tolerance=tolerance,
                particle_positions_sha256=array_hash(actual['particle_positions']),
                particle_velocities_sha256=array_hash(actual['particle_velocities']),
                joint_positions_sha256=array_hash(actual['joint_positions']))


def patch_simulator(code):
    """Add isolated snapshot commands and unload-cohort accounting to a copy."""
    if '_branch_snapshots = {}' in code:
        raise ValueError('Simulator already contains the unloading snapshot patch')
    declaration = '''    # Private unloading experiment: bridge remains paused for snapshot I/O.
    import copy as _branch_copy
    from act_unload_snapshot import capture as _capture_branch, restore as _restore_branch, array_hash as _branch_hash
    _branch_snapshots = {}
    _unload_cohort = None
'''
    handlers = '''            if cmd.get('type') in ('unload_snapshot_save', 'unload_snapshot_restore'):
                try:
                    if active_contract is None or sand_api.get('soil_backend', 'physx') != 'physx':
                        raise ValueError('Snapshot requires an active PhysX deployment contract')
                    timeline.pause()
                    timeline.commit()
                    _particle_path = f"{sand_root_path}/{sand_module.PARTICLE_POINTS_PATH_SUFFIX}"
                    _particles = UsdGeom.Points(stage.GetPrimAtPath(_particle_path))
                    _key = str(cmd['snapshot_id'])
                    if cmd['type'] == 'unload_snapshot_save':
                        if _key in _branch_snapshots:
                            raise ValueError('Snapshot IDs are immutable')
                        # Normal render steps need not publish velocity arrays.
                        # Explicitly fetch current PhysX positions AND velocities
                        # before taking the physical snapshot.
                        sync_physics_to_usd()
                        _bucket_ids = []
                        estimate_bucket_load_particles(source_tracking_required=True, particle_indices=_bucket_ids)
                        _unload_cohort = set(_bucket_ids)
                        _physical = _capture_branch(robot, _particles)
                        _history = _branch_copy.deepcopy(dict(
                            q_target=q_target, previous_q=deployment_previous_q, previous_load=deployment_previous_load,
                            previous_fill=deployment_previous_fill_fraction, previous_rate=deployment_previous_fill_rate,
                            elapsed=deployment_elapsed_seconds, scheduler=tick_scheduler,
                            phase=active_phase_estimator, tracker=bucket_source_tracker,
                            cohort=_unload_cohort, reply=reply))
                        _branch_snapshots[_key] = dict(physical=_physical, history=_history)
                        _physical_path = cmd.get('physical_path')
                        if _physical_path:
                            np.savez_compressed(_physical_path, **_physical, cohort_ids=np.array(sorted(_unload_cohort)))
                        response_queue.put(dict(type='unload_snapshot_ack', ok=True, snapshot_id=_key,
                            particle_positions_sha256=_branch_hash(_physical['particle_positions']),
                            particle_velocities_sha256=_branch_hash(_physical['particle_velocities']),
                            maximum_particle_speed=float(np.max(np.linalg.norm(_physical['particle_velocities'], axis=1))),
                            particle_count=len(_physical['particle_positions']), cohort_count=len(_unload_cohort),
                            physics_time=world.current_time, deployment_elapsed_seconds=deployment_elapsed_seconds))
                    else:
                        _saved = _branch_snapshots[_key]
                        _history = _branch_copy.deepcopy(_saved['history'])
                        _audit = _restore_branch(robot, _particles, _saved['physical'], _history['q_target'],
                            canonical_joint_indices, ArticulationAction, sync_physics_to_usd, simulation_app.update)
                        _bucket_prim = stage.GetPrimAtPath('/World/URDF_real3/bucket_link')
                        _bucket_transform = UsdGeom.Xformable(_bucket_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                        _landmark_errors = {}
                        for _name, _point in [('tip', (.75, 0., -.18)), ('load', (.35, 0., .08)), ('pour', (.85, 0., .30))]:
                            _actual_point = np.asarray(_bucket_transform.Transform(Gf.Vec3d(*_point)))
                            _expected_point = np.asarray(_history['reply']['bucket_landmarks_world'][_name])
                            _landmark_errors[_name] = float(np.max(np.abs(_actual_point - _expected_point)))
                        if max(_landmark_errors.values()) > 1e-4:
                            raise ValueError(f'Restored bucket transform differs: {_landmark_errors}')
                        _audit['bucket_landmark_max_abs_errors_m'] = _landmark_errors
                        q_target = _history['q_target']
                        deployment_previous_q = _history['previous_q']
                        deployment_previous_load = _history['previous_load']
                        deployment_previous_fill_fraction = _history['previous_fill']
                        deployment_previous_fill_rate = _history['previous_rate']
                        deployment_elapsed_seconds = _history['elapsed']
                        tick_scheduler = _history['scheduler']
                        active_phase_estimator = _history['phase']
                        bucket_source_tracker.clear()
                        bucket_source_tracker.update(_history['tracker'])
                        _unload_cohort = _history['cohort']
                        # Give both arms the same real camera observation captured
                        # at this snapshot. Later observations are live renderings.
                        reply = _history['reply']
                        response_queue.put(dict(type='unload_restore_ack', ok=True, snapshot_id=_key,
                            audit=_audit, observation=reply, deployment_elapsed_seconds=deployment_elapsed_seconds))
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    response_queue.put(dict(type='error', ok=False, error=f'UnloadSnapshotError: {type(exc).__name__}: {exc}'))
                continue

'''
    cohort = '''            if _unload_cohort is not None:
                _cb = _unload_cohort.intersection(bucket_particle_indices or [])
                _ct = _unload_cohort.intersection(truck_particle_indices or [])
                reply['unload_cohort_metrics'] = dict(total=len(_unload_cohort), bucket=len(_cb),
                    truck=len(_ct), outside=len(_unload_cohort - _cb - _ct))
'''
    patches = [
        ('    while simulation_app.is_running():\n        lock_visible_viewport_to_overview()',
         declaration + '\n    while simulation_app.is_running():\n        lock_visible_viewport_to_overview()'),
        ('            if cmd.get("type") == "reset_episode":', handlers + '            if cmd.get("type") == "reset_episode":'),
        ('            response_queue.put(reply)\n\n        else:', cohort + '\n            response_queue.put(reply)\n\n        else:')]
    for original, replacement in patches:
        if code.count(original) != 1:
            raise ValueError('Unexpected simulator source for snapshot patch')
        code = code.replace(original, replacement)
    return code
