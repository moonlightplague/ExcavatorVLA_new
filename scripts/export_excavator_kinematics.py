"""Read authored USD joint frames without starting physics or an expert policy."""
import argparse
import hashlib
import json
from pathlib import Path


def export_model(scene_path):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    scene_path = Path(scene_path).resolve()
    stage = Usd.Stage.Open(str(scene_path), load=Usd.Stage.LoadNone)
    root = '/World/URDF_real3'
    stage.Load(root, Usd.LoadWithoutDescendants)
    stage.Load('/World/truck', Usd.LoadWithoutDescendants)
    cache = UsdGeom.XformCache()

    def matrix(value):
        # USD uses row vectors; the portable model uses column vectors.
        return [[float(value[j][i]) for j in range(4)] for i in range(4)]

    def joint_frame(position, rotation):
        value = Gf.Matrix4d().SetRotate(Gf.Quatd(rotation))
        value.SetTranslateOnly(Gf.Vec3d(position))
        return matrix(value)

    joints = []
    for name in ('swing', 'boom', 'arm', 'bucket'):
        joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(f'{root}/joints/{name}'))
        if not joint:
            raise ValueError(f'Missing authored joint: {name}')
        joints.append(dict(name=name, axis=joint.GetAxisAttr().Get(),
            body0=str(joint.GetBody0Rel().GetTargets()[0]),
            body1=str(joint.GetBody1Rel().GetTargets()[0]),
            parent_from_joint=joint_frame(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get()),
            child_from_joint=joint_frame(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get())))
    for first, second in zip(joints, joints[1:]):
        if first['body1'] != second['body0']:
            raise ValueError('Expected the deployed serial joint chain')
    truck = stage.GetPrimAtPath('/World/truck')
    bed = stage.GetPrimAtPath('/World/truck/DumpBedCollision/dump_bed_collision')
    ops = {op.GetOpName(): op.Get() for op in UsdGeom.Xformable(truck).GetOrderedXformOps()}
    rotation = ops['xformOp:rotateXYZ']
    if abs(rotation[0]) + abs(rotation[1]) > 1e-6:
        raise ValueError('Expected the deployment truck root to use yaw only')
    truck_template = dict(root_xyz=list(map(float, ops['xformOp:translate'])),
        root_yaw_deg=float(rotation[2]),
        bed_world_row_matrix=[list(map(float, row)) for row in cache.GetLocalToWorldTransform(bed)])
    layers = {str(Path(layer.realPath).resolve()): hashlib.sha256(Path(layer.realPath).read_bytes()).hexdigest()
              for layer in stage.GetUsedLayers() if layer.realPath}
    return dict(schema='excavator_authored_chain_v1', scene_path=str(scene_path),
        source_layers_sha256=layers, column_vector_convention=True,
        base_world=matrix(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(joints[0]['body0']))),
        joints=joints, truck_template=truck_template,
        landmarks=dict(tip=[.75, 0., -.18], load=[.35, 0., .08], pour=[.85, 0., .30]),
        scope='Authored joint geometry only; no physics, motion planning, or expert execution')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    model = export_model(args.scene)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(model, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(args.output), base_world_xyz=[row[3] for row in model['base_world'][:3]],
                          source_layers=len(model['source_layers_sha256']))))


if __name__ == '__main__':
    main()
