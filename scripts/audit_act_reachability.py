"""Read-only geometry audit of saved ACT scenes; never revises old task scores."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from excavator_reachability import (AuthoredExcavatorChain, TruckBedColumn, find_endpoint,
                                    restored_truck_geometry, workspace_bound)

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def verify_asset_sources(scene_path, model):
    snapshot = next((parent/'source_snapshot.json' for parent in scene_path.parents
                     if (parent/'source_snapshot.json').exists()), None)
    if snapshot is None:
        raise ValueError(f'Missing frozen source evidence: {scene_path}')
    files = json.loads(snapshot.read_text())['files']
    for path, expected in model['source_layers_sha256'].items():
        relative = str(Path(path).relative_to(ROOT))
        if (files.get(relative) != expected or sha(path) != expected
                or sha(snapshot.parent/'frozen_sources'/relative) != expected):
            raise ValueError(f'Authored asset differs from the recorded simulation: {relative}')
    return str(snapshot)


def validate_saved_geometry(chain, rows, scene):
    errors = []
    for row in rows:
        state = np.asarray(row['input_state'])
        landmarks = row.get('input_bucket_landmarks_world')
        if landmarks:
            for name in ('tip', 'load', 'pour'):
                errors.append(float(np.linalg.norm(chain.point(state[:4], name)-landmarks[name])))
        else:
            heading = chain.zero_heading + state[0]
            c, s = np.cos(heading), np.sin(heading)
            delta = np.array([c*state[21]-s*state[22], s*state[21]+c*state[22], state[23]])
            load = np.asarray(scene['observation_context']['unload_landing_xyz'])-delta
            errors.append(float(np.linalg.norm(chain.point(state[:4], 'load')-load)))
    if not errors or max(errors) > .02:
        raise ValueError('Authored FK fails the independent recorded-geometry check')
    return dict(points_checked=len(errors), max_error_m=max(errors), mean_error_m=float(np.mean(errors)),
                tolerance_m=.02)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--scenes', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model = json.loads(args.model.read_text())
    chain = AuthoredExcavatorChain(model)
    protocol = dict(scope='retrospective development geometry diagnosis, not a new success-rate evaluation',
        scene_contexts=[str(path.resolve()) for path in args.scenes],
        model_path=str(args.model.resolve()), model_sha256=sha(args.model),
        bed_inset_m=.1, release_above_rim_m=.1, spatial_tolerance_m=.005,
        loaded_bucket_upper_rad=-1.3, carry_pitch_upper_deg=-130., open_bucket_lower_rad=-.8,
        endpoint_grid=[17, 21, 9], max_refinement_starts=8, refinement_max_nfev=100,
        exclusion='Only a conservative outer-workspace bound can certify outside release geometry; '
                  'IK misses remain unresolved. No collision/path/particle-transfer certification.',
        scoring='Old results and denominators unchanged; no post-hoc filtered success rate.',
        new_simulations=0, optimizer_training_updates=0)
    write(args.output/'protocol.json', protocol)
    frozen = args.output/'frozen_sources'
    frozen.mkdir()
    source_files = {}
    for name in ('audit_act_reachability.py', 'excavator_reachability.py', 'export_excavator_kinematics.py'):
        source = ROOT/'scripts'/name
        source_files[name] = sha(source)
        shutil.copy2(source, frozen/name)
    protected = {str(path): sha(path) for scene in args.scenes
                 for path in (scene, scene.parent/'steps.jsonl', scene.parent/'episode_status.json')}
    write(args.output/'source_manifest.json', dict(scripts_sha256=source_files, inputs_sha256=protected))
    results = []
    for path in args.scenes:
        original = json.loads(path.read_text())
        source_snapshot = verify_asset_sources(path.resolve(), model)
        scene = restored_truck_geometry(original, model)
        rows = [json.loads(line) for line in (path.parent/'steps.jsonl').read_text().splitlines()]
        geometry = validate_saved_geometry(chain, rows, scene)
        bed = TruckBedColumn(scene)
        target = scene['observation_context']['unload_landing_xyz']
        bound = workspace_bound(chain, bed, target)
        endpoints = {}
        for region, certified in (('bed', bound['bed_column_certified_outside']),
                                  ('target', bound['target_column_certified_outside'])):
            endpoints[region] = ({kind: dict(status='outside_release_workspace_bound') for kind in ('dump', 'carry')}
                                 if certified else {kind: find_endpoint(chain, scene, kind, target_only=region=='target')
                                                    for kind in ('dump', 'carry')})
        result = dict(scene_seed=scene['scene_seed'], scene_context=str(path.resolve()),
            source_snapshot=source_snapshot, observation_target_xyz=target,
            reconstructed_missing_truck_matrix=original['truck_counting_volume'].get('local_from_world') is None,
            recorded_geometry_validation=geometry, bounds=bound, endpoints=endpoints)
        results.append(result)
        print(json.dumps(dict(seed=scene['scene_seed'], target_distance=bound['target_xy_distance_m'],
            target_outside=bound['target_column_certified_outside'], bed_outside=bound['bed_column_certified_outside'],
            bed_endpoints={key: value['status'] for key, value in endpoints['bed'].items()})), flush=True)
    for path, expected in protected.items():
        if sha(path) != expected:
            raise ValueError(f'Source result changed during the audit: {path}')
    if sha(args.model) != protocol['model_sha256']:
        raise ValueError('Authored geometry changed during the audit')
    for name, expected in source_files.items():
        if sha(ROOT/'scripts'/name) != expected or sha(frozen/name) != expected:
            raise ValueError('Audit implementation changed during execution')
    write(args.output/'results.json', dict(protocol=protocol, scenes=results, source_inputs_unchanged=True))
    lines = ['# ACT 卸料目标与车斗可达性诊断', '',
        '只读复核已有场景，不训练、不启动新仿真、不改变旧任务成绩或统计分母。', '',
        '几何端点只说明关节限位内存在一个位置；未检查碰撞、运输路径或物料运动，不能据此认定完整任务可行。数值求解失败记为待判。', '',
        '| 场景 seed | 目标距回转中心/m | 指定目标超出几何上界 | 整个车斗超出几何上界 | 车内卸料端点 | 车内保料端点 |',
        '| --- | ---: | --- | --- | --- | --- |']
    def label(endpoint):
        return {'endpoint_found_path_unverified': '找到，路径未验', 'unresolved': '待判',
                'outside_release_workspace_bound': '超出卸料工作空间'}[endpoint['status']]
    for row in results:
        b, e = row['bounds'], row['endpoints']['bed']
        lines.append(f"| {row['scene_seed']} | {b['target_xy_distance_m']:.3f} | "
                     f"{'是' if b['target_column_certified_outside'] else '未证实'} | "
                     f"{'是' if b['bed_column_certified_outside'] else '未证实'} | {label(e['dump'])} | {label(e['carry'])} |")
    lines.extend(['', '上界采用真实基座/关节框架和车斗计数几何，并故意放宽关节限制：只有最宽松几何仍够不到才给出排除证据。',
        '证据范围是当前卸料点位于车斗上方的重力卸料约定，不证明任意抛射动作都无法入车。', '',
        '下一步先处理指定目标不可达但车斗近侧可用的目标准备问题。正式新对照应在策略执行前固定场景和目标，通过采集流程的可达性、碰撞及路径预检；原 ACT 与候选共用同一清单。',
        '场景预检未通过、仅约束姿态未找到、策略执行失败分别记录。旧结果保留，不按这次分析事后重算成功率。', '',
        '逐场变换、几何误差、保守上界、关节解及来源哈希见 `results.json`、`protocol.json` 和 `source_manifest.json`。'])
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
