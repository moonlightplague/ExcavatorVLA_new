"""Export physically completed expert suffixes, retaining zero-weight model prefixes."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def encode_lossless(frames,path,fps=10):
    import imageio_ffmpeg
    import numpy as np
    frames=np.asarray(frames,dtype=np.uint8)
    if frames.ndim!=4 or frames.shape[-1]!=3:raise ValueError('Expected RGB camera frames')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    writer=imageio_ffmpeg.write_frames(str(path),size=(frames.shape[2],frames.shape[1]),fps=fps,
        codec='libx264rgb',pix_fmt_in='rgb24',pix_fmt_out='rgb24',ffmpeg_log_level='error',
        output_params=['-crf','0','-preset','fast'])
    writer.send(None)
    try:
        for frame in frames:writer.send(np.ascontiguousarray(frame))
    finally:writer.close()


def verify_video(path,expected):
    import av
    import numpy as np
    count=0
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            if count>=len(expected):raise ValueError('Video has extra frames')
            np.testing.assert_array_equal(frame.to_ndarray(format='rgb24'),expected[count])
            if frame.time is None or abs(frame.time-count/10)>1e-5:raise ValueError('Video timestamp mismatch')
            count+=1
    if count!=len(expected):raise ValueError('Video lost frames')
    return count


def validate_collection(folder):
    import numpy as np
    from act_transport_diagnostics import analyze_transport
    status=read(folder/'episode_status.json')
    if not status.get('accepted_for_training') or not status.get('success') or status['status']!='completed':
        raise ValueError('Only completed full-task expert recoveries may be exported')
    if status['execution_source']!='model_prefix_then_online_geometry_expert':raise ValueError('Wrong execution provenance')
    rows=[json.loads(s) for s in (folder/'steps.jsonl').read_text().splitlines()]
    supervision=read(folder/'supervision.json');meta=read(folder/'trajectory/meta.json')
    arrays=np.load(folder/'trajectory/arrays.npz',allow_pickle=False)
    if not len(rows) or len(rows)!=meta['frames'] or len(rows)!=len(supervision['accepted_weights']):raise ValueError('Incomplete capture')
    if [r['step'] for r in rows]!=list(range(len(rows))):raise ValueError('Nonconsecutive steps')
    sources=[r['action_source'] for r in rows]
    expected=[int(s=='online_geometry_expert') for s in sources]
    if supervision['accepted_weights']!=expected or not any(expected):raise ValueError('Wrong expert supervision mask')
    start=expected.index(1)
    if any(expected[:start]) or not all(expected[start:]):raise ValueError('Expert suffix must be contiguous')
    if start!=status['prefix_steps']:raise ValueError('Wrong takeover position')
    before=np.asarray([r['input_state'] for r in rows],dtype=np.float32)
    after=np.asarray([r['state'] for r in rows],dtype=np.float32)
    actions=np.asarray([r['applied'] for r in rows],dtype=np.float32)
    for key,value in [('observation.state',before),('observation.state_next',after),('action.applied',actions)]:
        np.testing.assert_array_equal(arrays[key],value)
    if not np.isfinite(before).all() or not np.isfinite(actions).all():raise ValueError('Nonfinite capture')
    if (np.abs(actions)>np.array([6.,2.5,2.5,6.])+1e-6).any():raise ValueError('Action limit violation')
    np.testing.assert_allclose(before[1:,12:16],actions[:-1],rtol=0,atol=1e-5)
    np.testing.assert_allclose(after[:-1],before[1:],rtol=0,atol=1e-6)
    consecutive=0;full=False
    limits=np.asarray(read(folder/'scene_context.json')['canonical_joint_limits_rad'])
    for i,row in enumerate(rows):
        if abs(row['bridge_timing']['simulated_seconds']-.1)>1e-5:raise ValueError('Wrong physical action cadence')
        if i:
            previous=np.asarray(rows[i-1]['joint_command_positions'])
            command=np.clip(previous+.1*actions[i],limits[:,0],limits[:,1])
            np.testing.assert_allclose(command,row['joint_command_positions'],rtol=0,atol=2e-5)
        consecutive=consecutive+1 if row['deposited_particles']>=100 and row['bucket_position']>=-.8 else 0
        if i>=start:full=full or consecutive>=3
    if not full:raise ValueError('Full task completion does not reproduce from the recorded observations')
    transport=analyze_transport(folder)
    if not transport['available'] or transport.get('pretruck_empty_step') is not None:
        raise ValueError('No valid loaded transport or pre-truck empty event')
    result=dict(status='passed',frames=len(rows),expert_start=start,expert_frames=len(rows)-start,
        prefix_positive_weights=0,action_history_aligned=True,command_integrator_aligned=True,
        full_task_recomputed=True,transport=transport,
        files_sha256={name:sha(folder/name) for name in ['steps.jsonl','episode_status.json','supervision.json',
            'trajectory/arrays.npz','trajectory/frames.npz','trajectory/meta.json']})
    return result,rows,arrays,supervision,meta


def export(collection,output):
    import numpy as np
    import pandas as pd
    sys.path.insert(0,str(ROOT/'ExcavatorVLA-vlm'))
    from excavator_vlm.data.contract import audit_dataset,CAMERAS,STATE_NAMES,ACTION_NAMES
    from excavator_vlm.rl.build_finetune_dataset import build_info,compute_stats
    collections=collection if isinstance(collection,list) else [collection]
    protocols={folder:read(folder/'protocol.json') for folder in collections}
    for folder in collections:
        if read(folder/'status.json')['status']!='completed':raise ValueError('Finish collection before exporting its data')
    protocol=protocols[collections[0]]
    archive=Path(protocol['archive'])
    if any(p['baseline_id']!=protocol['baseline_id'] or p['model_sha256']!=protocol['model_sha256'] for p in protocols.values()):
        raise ValueError('Collections use different baseline or normalization sources')
    candidates=[folder/f'scene{case["scene"]}' for folder,p in protocols.items() for case in p['cases']]
    candidates=[p for p in candidates if (p/'episode_status.json').exists() and read(p/'episode_status.json').get('accepted_for_training')]
    if not candidates:raise ValueError('No successful expert recoveries; no dataset is created')
    checked=[]
    for folder in candidates:
        checked.append((folder,validate_collection(folder)))
    output.mkdir(parents=True,exist_ok=False)
    for name in ('meta','provenance'):(output/name).mkdir()
    tasks=[];all_rows=[];episodes=[];eligible=[];sources=[];offset=0
    for ep,(folder,(validation,trace,arrays,supervision,meta)) in enumerate(checked):
        scene=read(folder/'scene_context.json');task=scene['observation_context']['task_text']
        if task not in tasks:tasks.append(task)
        length=len(trace);chunk=f'chunk-{ep:03d}';filename='file-000'
        rows=[]
        for frame,row in enumerate(trace):
            expert=row['action_source']=='online_geometry_expert'
            item=dict(index=offset+frame,episode_index=ep,frame_index=frame,timestamp=frame/10,
                task_index=tasks.index(task),**{'observation.state':arrays['observation.state'][frame].tolist(),
                'action':arrays['action.applied'][frame].tolist(),'action_loss_weight':float(expert),
                'action_is_expert':expert,'action.source_id':int(expert),
                'observation.stage_current_id':int(row['stage_id']) if expert else -1,
                'observation.stage_known':expert,'observation.recovery_active':expert})
            rows.append(item)
            if expert:eligible.append(dict(episode_index=ep,frame_index=frame))
        path=output/'data'/chunk;path.mkdir(parents=True)
        pd.DataFrame(rows).to_parquet(path/(filename+'.parquet'),index=False)
        episode=dict(episode_index=ep,tasks=[task],length=length,dataset_from_index=offset,dataset_to_index=offset+length,
            **{'meta/episodes/chunk_index':ep,'meta/episodes/file_index':0,'data/chunk_index':ep,'data/file_index':0})
        raw_frames=np.load(folder/'trajectory/frames.npz',allow_pickle=False)
        verified=0
        for camera,key in enumerate(CAMERAS):
            frames=raw_frames[str(camera)]
            if frames.shape!=(length,256,256,3):raise ValueError('Invalid captured image dimensions')
            expected=meta['frame_sha256'][str(camera)]
            if len(expected)!=length:raise ValueError('Missing frame hashes')
            for i,frame in enumerate(frames):
                if hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()!=expected[i]:raise ValueError('Captured frame hash mismatch')
                if expected[i]!=trace[i]['pre_action_camera_sha256'][str(camera)]:raise ValueError('Trace/frame identity mismatch')
            video=output/'videos'/key/chunk/(filename+'.mp4')
            encode_lossless(frames,video);verified+=verify_video(video,frames)
            episode.update({f'videos/{key}/chunk_index':ep,f'videos/{key}/file_index':0,
                f'videos/{key}/from_timestamp':0.,f'videos/{key}/to_timestamp':(length-1)/10})
        validation['lossless_video_frames_verified']=verified
        write(folder/'recovery_validation.json',validation)
        evidence=dict(source_episode=str(folder),scene_seed=scene['scene_seed'],
            simulation_settings=scene['simulation_settings'],expert_version=meta['provenance']['expert_version'],
            expert_start=validation['expert_start'],expert_frames=validation['expert_frames'],
            acceptance='Full original task completed; original ACT prefix has no expert supervision',source_validation=validation)
        evidence_path=output/'provenance'/f'episode_{ep:04d}.json';write(evidence_path,evidence)
        sources.append(dict(episode_index=ep,soil_backend='physx',evidence=dict(path=str(evidence_path.relative_to(output)),
            sha256=sha(evidence_path),field='simulation_settings.soil_backend')))
        meta_path=output/'meta/episodes'/chunk;meta_path.mkdir(parents=True)
        pd.DataFrame([episode]).to_parquet(meta_path/(filename+'.parquet'),index=False)
        all_rows.extend(rows);episodes.append(evidence);offset+=length
        print(json.dumps(dict(exported_episode=ep,scene=scene['scene_seed'],frames=length,expert_frames=validation['expert_frames'])),flush=True)
    info=build_info(len(episodes),len(all_rows),len(tasks),10)
    info.update(state_schema_version='excavator_state_v4_28d_expert_recovery_v1',
        action_policy_version='online_geometry_expert_recovery_v1',
        normalization_policy='Frozen original PHW40k train-only statistics; recorded separately from new dataset counts',
        training_observation_manifest='eligible_training_observations.json',
        training_eligibility='Only current observations from validated expert suffixes; model prefix never sampled')
    for key,dtype in [('action_is_expert','bool'),('action.source_id','int64'),('observation.stage_current_id','int64'),
                      ('observation.stage_known','bool'),('observation.recovery_active','bool')]:
        info['features'][key]=dict(dtype=dtype,shape=[1],names=[key])
    stats=compute_stats(all_rows);original_stats=read(archive/'normalization_stats.json')
    for key in ('observation.state','action',*CAMERAS):stats[key]=original_stats[key]
    write(output/'meta/info.json',info);write(output/'meta/stats.json',stats)
    pd.DataFrame({'task_index':list(range(len(tasks)))},index=pd.Index(tasks)).to_parquet(output/'meta/tasks.parquet',index=True)
    write(output/'backend_provenance.json',dict(schema_version='excavator_backend_provenance_v1',soil_backend='physx',episodes=sources))
    write(output/'eligible_training_observations.json',dict(reason='Current observation must belong to the validated expert suffix',
        observations=eligible,model_prefix_eligible=False))
    write(output/'recovery_provenance.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        collections=[str(folder) for folder in collections],baseline_id=protocol['baseline_id'],
        expert_versions=sorted({e['expert_version'] for e in episodes}),
        episodes=len(episodes),frames=len(all_rows),expert_frames=len(eligible),prefix_frames=len(all_rows)-len(eligible),
        normalization_source=str(archive/'normalization_stats.json'),normalization_source_sha256=sha(archive/'normalization_stats.json'),
        train_only=True,original_validation_and_test_unchanged=True,video_encoding='Lossless H264 RGB; every decoded frame verified',
        source_semantics=dict(model_prefix=0,online_geometry_expert=1,unknown_prefix_stage=-1)))
    result=audit_dataset(output)
    write(output/'dataset_audit.json',result[0])
    return dict(status='exported_and_audited',output=str(output),episodes=len(episodes),
                frames=len(all_rows),expert_frames=len(eligible),dataset_fingerprint=result[0]['fingerprint'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection',type=Path,nargs='+',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();sys.path.insert(0,str(ROOT/'scripts'))
    print(json.dumps(export([p.resolve() for p in args.collection],args.output.resolve()),ensure_ascii=False))
