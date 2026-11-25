"""Explicit ten-cycle extension of an archived five-cycle PhysX reference.

Used only by the private evaluator prepared for this experiment. The archived
publication and the existing five-cycle evaluator are left untouched.
"""
import copy
import math
from pathlib import Path

import numpy as np

STANDARD = 'excavator_repeat_physx_ten_reference_v1'


def load_physx_ten_spec(dataset, reference_path, strict_minimum):
    from repeat_eval import load_repeat_spec, read, digest

    dataset, reference_path = Path(dataset), Path(reference_path)
    legacy = load_repeat_spec(dataset, reference_path, 5, 500, strict_minimum)
    reference = read(reference_path)
    eid = reference['provenance']['source_export_episode']
    meta_path = dataset / f'provenance/episode_{eid:04d}/meta.json'
    index_path = meta_path.with_name('episode_index_record.json')
    meta, record = read(meta_path), read(index_path)
    mass = float(meta['repeat_metrics']['particle_mass_kg'])
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError('Archived PhysX particle mass is invalid')
    scene = record['scene_randomization']['applied']['scene_context']
    polygon = np.asarray(scene['sand_polygon_xy'], dtype=float)
    center = np.asarray(reference['profile']['sand_xy'], dtype=float)
    if (polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2
            or not np.isfinite(polygon).all() or center.shape != (2,)
            or not np.isfinite(center).all()
            or not np.allclose(scene['pile_center'][:2], center, atol=1e-5)):
        raise ValueError('Archived PhysX pile geometry is invalid')
    result = copy.deepcopy(legacy)
    result.update(
        version=STANDARD, cycles=10, cycle_step_limit=0,
        task=legacy['task'].replace('Dig 5 consecutive scoops', 'Dig 10 consecutive scoops'),
        goals=[copy.deepcopy(g) for _ in range(2) for g in legacy['goals']],
        goal_source='archived_five_recorded_goals_repeated_twice_without_scene_reset',
        source_goal_indices=list(range(5))*2,
        metric_point_mass_kg=mass,
        minimum_per_cycle_kg=mass*legacy['min_payload_particles'],
        # The historical collection criterion had no final-mass retention gate.
        minimum_retention_ratio=0.,
        pile_polygon_xy=polygon.tolist(), pile_center_xy=center.tolist(),
        truck_center_xy=list(reference['profile']['unload_landing_xyz'][:2]),
        ground_top_z=float(meta['repeat_baseline']['ground_top_z']),
        dig_max_height_m=2.5, carry_min_height_m=1.2, carry_bucket_closed_rad=-1.,
        return_height_above_ground_m=[.15, 2.5], physical_event_observations=3,
        source_identity='initial_pile_region',
        return_region_method='legacy_recorded_goal_relative_xy_and_height_window',
        success_scope='Archived scoop/deposit thresholds plus ten physical returns; '
                      '50 kg per cycle is reported separately as a diagnostic.',
        scope='Old PhysX reference-scene diagnostic: five recorded goals repeated twice, '
              'ten continuous model-controlled cycles, no material/robot/history reset; '
              'not an original recorded ten-cycle or held-out Newton benchmark.')
    result['source_evidence'][str(reference_path)] = digest(reference_path)
    return result


def task_class():
    # Import lazily: repeat_eval imports this factory while dispatching a spec.
    from repeat_mass_physical import PhysicalRepeatTask

    class PhysXTenTask(PhysicalRepeatTask):
        def reference_return_region(self, reply, tip):
            goal = self.spec['goals'][len(self.cycles)]['dig_target_xyz']
            low, high = self.spec['return_height_above_target_m']
            return (math.dist(tip[:2], goal[:2]) <= self.spec['return_xy_radius_m']
                    and low <= float(tip[2])-goal[2] <= high)

        def _global_transfer(self, ids):
            mass = ids.get('metric_point_mass_kg', 0.)
            count = ids.get('particle_count', 0)
            identity = (count, ids.get('source_count'),
                        tuple(ids.get('initial_excluded_ids', [])))
            if (ids.get('soil_backend') != 'physx' or not math.isfinite(mass)
                    or not math.isclose(mass, self.spec['metric_point_mass_kg'], rel_tol=1e-6)
                    or type(count) is not int or count <= 0
                    or ids.get('source_identity') != 'initial_pile_region'
                    or type(ids.get('source_count')) is not int
                    or not 0 < ids['source_count'] <= count
                    or getattr(self, '_physx_identity', identity) != identity):
                raise ValueError('Live PhysX material mass/backend/source identity differs')
            self._physx_identity = identity
            return self.global_tracker.observe(count, ids['bucket'], ids['truck'])

        def result(self, *, final=False, wall=None):
            result = super().result(final=final, wall=wall)
            mass = self.spec['metric_point_mass_kg']
            for row in result['repeat_cycles']:
                if 'payload_particles' in row:
                    row['delivered_mass_kg'] = row['payload_particles']*mass
            complete = [r for r in result['repeat_cycles'] if r.get('physical_cycle_completed')]
            result['diagnostic_cycles_ge_50kg'] = sum(r['delivered_mass_kg'] >= 50 for r in complete)
            result['original_quality_thresholds_preserved'] = True
            return result

    return PhysXTenTask
