"""Read the authored bucket collision vertices for conservative ground checks."""
import argparse
import hashlib
import json
from pathlib import Path


def extract(scene):
    from pxr import Usd,UsdGeom,UsdPhysics,Gf
    stage=Usd.Stage.Open(str(scene),load=Usd.Stage.LoadNone)
    stage.Load('/World/URDF_real3')
    body=stage.GetPrimAtPath('/World/URDF_real3/bucket_link')
    cache=UsdGeom.XformCache();inverse=cache.GetLocalToWorldTransform(body).GetInverse()
    meshes=[];vertices=[]
    for prim in Usd.PrimRange(body):
        if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):continue
        mesh=UsdGeom.Mesh(prim);transform=cache.GetLocalToWorldTransform(prim)*inverse
        points=mesh.GetPointsAttr().Get()
        if points:
            meshes.append(str(prim.GetPath()))
            vertices.extend([list(map(float,transform.Transform(Gf.Vec3d(*map(float,p))))) for p in points])
    if not vertices:raise ValueError('No authored bucket collision mesh vertices found')
    return dict(schema='bucket_collision_local_vertices_v1',scene=str(scene),scene_sha256=hashlib.sha256(scene.read_bytes()).hexdigest(),
                body=str(body.GetPath()),meshes=meshes,vertices=vertices)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=extract(args.scene.resolve());args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
    print(json.dumps(dict(meshes=result['meshes'],vertices=len(result['vertices']))))
