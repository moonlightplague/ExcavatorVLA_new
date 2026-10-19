"""Exercise the actual USD bound/transform APIs used by Isaac, without a GPU app."""
import ast
import os
from pathlib import Path
import subprocess
import sys
import unittest

BASE = Path(__file__).resolve().parents[1]


class InstalledUsdGeometryTests(unittest.TestCase):
    def test_translated_scaled_rotated_bed_uses_mesh_coordinates(self):
        libraries = sorted((BASE / 'isaacsim/extscache').glob('omni.usd.libs-*'))
        if not libraries:
            self.skipTest('Isaac USD bindings are not installed')
        library = libraries[0]
        environment = dict(os.environ)
        environment['PYTHONPATH'] = str(library)
        environment['LD_LIBRARY_PATH'] = str(library / 'bin') + ':' + environment.get('LD_LIBRARY_PATH', '')
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--usd-child'],
                                env=environment, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def exercise_usd():
    import numpy as np
    from pxr import Gf, Usd, UsdGeom
    import time
    from types import SimpleNamespace

    sys.path.insert(0, str(BASE / 'ExcavatorVLA-smolvla'))
    from excavator_common.evaluation_standard import TRUCK_COUNT_METHOD

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, '/truck')
    root.AddTranslateOp().Set(Gf.Vec3d(10, 20, 0))
    root.AddRotateZOp().Set(37.)
    bed = UsdGeom.Cube.Define(stage, '/truck/bed')
    bed.AddTranslateOp().Set(Gf.Vec3d(2, -3, 4))
    bed.AddScaleOp().Set(Gf.Vec3f(1.2, 1.7, .8))

    namespace = dict(np=np, Gf=Gf, Usd=Usd, UsdGeom=UsdGeom, stage=stage,
                     TRUCK_BED_COLLISION_PATH='/truck/bed', random_scene_profile={'seed': 12},
                     TRUCK_COUNT_METHOD=TRUCK_COUNT_METHOD, time=time)
    tree = ast.parse((BASE / 'ExcavatorVLA-smolvla/run_simulation.py').read_text())
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                 and node.name in ('authored_truck_bed_counting_volume', 'estimate_truck_bed_load_particles')]
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<real-usd-geometry>', 'exec'), namespace)
    volume = namespace['authored_truck_bed_counting_volume']()
    assert volume['available'], volume
    np.testing.assert_allclose(volume['local_min'], [-1 + .05/1.2, -1 + .05/1.7, -1 - .03/.8], atol=1e-6)
    np.testing.assert_allclose(volume['local_max'], [1 - .05/1.2, 1 - .05/1.7, 1 + .05/.8], atol=1e-6)

    transform = UsdGeom.Xformable(bed).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    center = transform.Transform(Gf.Vec3d(0, 0, 0))
    outside_corner = [volume['world_min'][0] + .1, volume['world_min'][1] + .1, center[2]]
    points = np.array([center, transform.Transform(Gf.Vec3d(.5, 0, 0)), outside_corner,
                       transform.Transform(Gf.Vec3d(0, .5, 0))])
    bucket = UsdGeom.Xform.Define(stage, '/World/URDF_real3/bucket_link')
    bucket.AddTranslateOp().Set(center)
    api = {'particle_positions_fn': lambda: points}
    namespace.update(truck_bed_counting_volume=volume, sand_api=api,
                     builtins=SimpleNamespace(_SAND_SITE=api),
                     bucket_source_tracker={'available': True, 'mask': np.array([True, True, True, False])},
                     bucket_volume_topology={'available': True},
                     points_in_closed_bucket_mesh=lambda local: np.all(np.abs(local) <= .1, axis=1))
    indices = []
    metric = namespace['estimate_truck_bed_load_particles'](indices)
    assert metric['available'], metric
    # Center is still in the bucket, a world-AABB corner is outside the oriented
    # bed, and the fourth particle is not from the initial pile. Only ID 1 counts.
    assert metric['candidate_count'] == 3, metric
    assert metric['count'] == 1 and indices == [1], (metric, indices)
    print('Real USD translated/scaled/rotated bounds and particle counting passed')


if __name__ == '__main__':
    if '--usd-child' in sys.argv:
        exercise_usd()
    else:
        unittest.main()
