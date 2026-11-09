"""Replay an unlabelled ACT prefix, then collect online expert recovery actions."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(os.environ.get('ACT_PROJECT_ROOT',str(Path(__file__).resolve().parents[1]))).resolve()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def takeover_valid(scene,state,particles):
    if particles['retained_particles']<100:return False
    fill=float(state[26]);bucket=float(state[3])
    if scene==1:return .002<=fill<=.04 and -1.95<bucket<-.8
    if scene==9:return .002<=fill<=.04 and bucket<=-1.8
    if scene==20:return .005<=fill<=.10 and bucket>-.5
    raise ValueError('Undeclared recovery scene')


def accepted_supervision(sources,success):
    return [int(bool(success) and source=='online_geometry_expert') for source in sources]


def worker(out,scene_index):
    protocol=read(out/'protocol.json');case=next(c for c in protocol['cases'] if c['scene']==scene_index)
    archive=Path(protocol['archive'])
    inventory=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],check=True,capture_output=True,text=True).stdout.splitlines()
    os.environ.update(CUDA_VISIBLE_DEVICES=inventory[case['gpu']],HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',
        HF_HOME=str(ROOT/'models/hf-cache'),TORCH_HOME=str(ROOT/'models/torch-cache'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    sys.path[:0]=[str(out/'frozen_scripts'),str(archive/'runtime_dependencies'),str(archive/'frozen_sources/scripts'),
                  str(archive/'frozen_sources/ExcavatorVLA-vlm')]
    import numpy as np
    from eval_smolvla_success import (SimulatorSession,exchange,validate_state,decoded_frames,frame_hashes,
        build_client_contract,OBSERVATION_SCHEMA_28D_V4_PLUS_EFFORT,EpisodeMetrics,CompletionTracker,EpisodeVideo)
    from excavator_common.simulation_settings import resolve_simulation_settings,apply_simulation_settings
    from excavator_common.deployment_contract import sha256_files
    from excavator_common.evaluation_standard import EVALUATION_STANDARD,TRUCK_COUNT_METHOD
    from excavator_vlm.rl.rollout_store import TrajectoryStore
    from act_recovery_expert import RecoveryExpert,RecoveryFailure
    from act_carry_guard import CarryAttitudeGuard,CarryGuardSettings

    folder=out/f'scene{scene_index}';folder.mkdir(exist_ok=False)
    prefix=read(Path(case['source_episode'])/'initial_observation.json')
    context=read(Path(case['source_episode'])/'scene_context.json')
    original=[json.loads(s) for s in (Path(case['source_episode'])/'steps.jsonl').read_text().splitlines()]
    if sha(Path(case['source_episode'])/'steps.jsonl')!=case['source_steps_sha256']:raise ValueError('Original prefix changed')
    profile=folder/'scene_profile.json';write(profile,context['scene_profile'])
    command=[str(archive/'frozen_sources/run_isaac_python.sh'),str(archive/'frozen_sources/ExcavatorVLA-smolvla/run_simulation.py'),
        '--sand-amount',str(context['scene_profile']['sand_amount_multiplier']),'--active-cuda-gpu','0','--cpu-threads','8',
        '--robot-initial-joints-deg',*map(str,np.degrees(prefix['state'][:4])),
        '--headless','--idle-ui-hz','.1','--render-every-physics-step','--scene-profile',str(profile),
        '--state27-dig-target-world',*map(str,context['observation_context']['dig_target_xyz']),
        '--state27-unload-target-world',*map(str,context['observation_context']['unload_landing_xyz']),
        '--task-text',prefix['task']]
    settings=resolve_simulation_settings('physx',False);apply_simulation_settings(settings)
    session=SimulatorSession(command,archive/'frozen_sources/ExcavatorVLA-smolvla',folder/'simulator.log',True,exchange,600)
    source_checkpoint=archive/'checkpoints/040000/pretrained_model'
    normalization=sha256_files(sorted(source_checkpoint.glob('*processor*')))
    maximum=case['recorded_state_step']+protocol['expert_steps']
    video=EpisodeVideo(folder/'evaluation.mp4',scene_index,case['seed'],10,maximum,100)
    store=TrajectoryStore(folder,10,case['seed'],provenance=dict(execution_source='model_prefix_then_online_geometry_expert',
        baseline_id=protocol['baseline_id'],prefix_source=case['source_episode'],expert_version=RecoveryExpert.version,
        original_checkpoint_model_sha256=protocol['model_sha256'],supervision='zero for prefix; expert accepted only if original full task completes'))
    status=dict(status='starting',success=False,task_completed_success_and_bucket_open=False,error=None,
        scene_seed=case['seed'],scene_index=scene_index,execution_source='model_prefix_then_online_geometry_expert',
        policy_type='expert_recovery',evaluation_profile='expert_recovery_collection',evaluation_standard=EVALUATION_STANDARD,
        truck_count_method=TRUCK_COUNT_METHOD,min_particles=100,policy_steps_executed=0,
        prefix_steps=case['recorded_state_step'],expert_steps=0,accepted_for_training=False)
    metrics=EpisodeMetrics();completion=CompletionTracker(100)
    sources=[];stages=[];expert=None;started=time.monotonic();rollout_started=None
    def interrupted(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        status.update(session.prepare(case['seed']))
        actual_scene=exchange(session.sock,{'type':'scene_context'})
        write(folder/'scene_context.json',actual_scene)
        if actual_scene['scene_seed']!=case['seed'] or actual_scene['simulation_settings']!=settings:
            raise ValueError('Wrong physical scene or physics')
        contract=build_client_contract(10,normalization,OBSERVATION_SCHEMA_28D_V4_PLUS_EFFORT,actual_scene['observation_context'])
        ack=exchange(session.sock,contract)
        if ack.get('type')!='handshake_ack' or ack.get('timing_mode')!='deterministic_lockstep':raise ValueError('Invalid bridge handshake')
        write(folder/'contract.json',dict(request=contract,reply=ack))
        reply=exchange(session.sock,{'joint_velocities':[0.,0.,0.,0.]});state=validate_state(reply)
        if np.max(np.abs(state[16:26]-np.asarray(prefix['state'])[16:26]))>.05:raise ValueError('Reference geometry mismatch')
        frames=decoded_frames(reply);metrics.observe(reply)
        write(folder/'initial_observation.json',dict(state=state.tolist(),task=reply['task_text'],camera_sha256=frame_hashes(frames)))
        guard=CarryAttitudeGuard(actual_scene,settings=CarryGuardSettings(parent_projection='arm_first',release_geometry='opening_pose'))
        rollout_started=time.monotonic()
        with (folder/'steps.jsonl').open('x') as trace:
            for step in range(maximum):
                before=validate_state(reply);digests=frame_hashes(frames)
                if step<case['recorded_state_step']:
                    predicted=np.asarray(original[step]['applied'],dtype=np.float32)
                    action_source='model_prefix_replay';details=None;stage=0
                else:
                    if expert is None:
                        valid=takeover_valid(scene_index,before,metrics.transfer)
                        report=dict(valid=valid,step=step,state=before.tolist(),particles=metrics.transfer,
                            joint_command_positions=reply['joint_command_positions'],
                            source_state_max_abs_difference=float(np.max(np.abs(before-np.asarray(case['recorded_state'])))))
                        write(folder/'takeover_check.json',report)
                        if not valid:
                            status['termination_reason']='failure_class_not_reproduced';break
                        expert=RecoveryExpert(read(out/'robot_model.json'),actual_scene,before)
                        write(folder/'expert_plan.json',dict(secure=expert.secure.tolist(),lift=expert.lift.tolist(),
                            slew=expert.slew.tolist(),dump=expert.dump.tolist(),opening=expert.plan,
                            progressive_curl=[point.tolist() for point in expert.curl_path]))
                    predicted,details=expert.action(before,reply['joint_command_positions'],metrics.transfer,reply['bucket_landmarks_world'])
                    action_source='online_geometry_expert';stage=details['stage_id']
                applied=np.clip(predicted,[-6.,-2.5,-2.5,-6.],[6.,2.5,2.5,6.]).astype(np.float32)
                guard_trace=None
                if expert is not None and expert.phase not in ('unwind','progressive_curl'):
                    applied,guard_trace=guard.update(before,applied,reply['joint_command_positions'],reply['bucket_landmarks_world'])
                input_landmarks=reply['bucket_landmarks_world']
                t=time.monotonic();post=exchange(session.sock,{'joint_velocities':applied.tolist()})
                latency=(time.monotonic()-t)*1000
                after=validate_state(post);metrics.observe(post,action_step=True)
                metrics.bridge_ms.append(latency);metrics.step_ms.append(latency)
                success=completion.observe(metrics.transfer['payload_particles'],float(after[3]))
                status.update(status='running',success=bool(success and expert is not None),
                    task_completed_success_and_bucket_open=bool(success and expert is not None),policy_steps_executed=step+1,
                    expert_steps=0 if expert is None else expert.total_steps,final_newly_deposited_particle_count=metrics.transfer['payload_particles'],
                    final_bucket_position_rad=float(after[3]),final_bucket_open=bool(after[3]>=-.8))
                store.add(step=step,frames=frames,frame_sha256=digests,state_pre=before,state_post=after,
                    predicted=predicted,applied=applied,particles=metrics.transfer,bucket_position=float(after[3]),replanned=expert is not None)
                sources.append(action_source);stages.append(stage)
                row=dict(step=step,predicted=predicted.tolist(),applied=applied.tolist(),action_source=action_source,
                    expert=details,stage_id=stage,carry_guard=guard_trace,input_bucket_landmarks_world=input_landmarks,
                    input_state=before.tolist(),state=after.tolist(),joint_command_positions=post['joint_command_positions'],
                    deposited_particles=metrics.transfer['payload_particles'],stable_steps=completion.consecutive,bucket_position=float(after[3]),
                    bridge_timing=post['bridge_timing'],particle_transfer_metrics=post['particle_transfer_metrics'],
                    truck_bed_load_metrics=post['truck_bed_load_metrics'],pre_action_camera_sha256=digests,
                    completion_observations=completion.consecutive)
                trace.write(json.dumps(row)+'\n');trace.flush()
                video.write(post,status,completion.consecutive)
                reply=post;frames=decoded_frames(reply)
                if step%20==0:
                    write(folder/'progress.json',dict(step=step,phase='prefix' if expert is None else expert.phase,
                        retained=metrics.transfer['retained_particles'],deposited=metrics.transfer['payload_particles'],
                        state=after[:4].tolist(),expert=details,wall_seconds=time.monotonic()-started))
                if success and expert is not None:
                    status.update(termination_reason='expert_full_task_completed',task_completion_time_from_policy_start_seconds=time.monotonic()-rollout_started)
                    break
        status['status']='completed'
        status.setdefault('termination_reason','expert_step_limit')
    except RecoveryFailure as error:
        status.update(status='completed',success=False,task_completed_success_and_bucket_open=False,
            termination_reason='expert_recovery_failed',expert_failure=str(error))
    except BaseException as error:
        status.update(status='error',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        elapsed=time.monotonic()-rollout_started if rollout_started else None
        metrics.finalize(status,elapsed)
        status['accepted_for_training']=bool(status['status']=='completed' and status['success'] and expert is not None)
        write(folder/'episode_status.json',status)
        if expert is not None:write(folder/'expert_transitions.json',expert.transitions)
        store.close(status)
        write(folder/'supervision.json',dict(action_sources=sources,stage_ids=stages,
            accepted_weights=accepted_supervision(sources,status['accepted_for_training']),
            prefix_is_expert=False,acceptance='Original full task completed after online expert takeover'))
        try:video.finish(status)
        finally:session.close()
    print(json.dumps(dict(scene=scene_index,success=status['success'],accepted=status['accepted_for_training'],
        retained=status.get('retained_particles'),deposited=status.get('payload_particles'),termination=status['termination_reason'])),flush=True)


def prepare(out,scenes):
    out.mkdir(parents=True,exist_ok=False)
    manifest=read(ROOT/'docs/baselines/act_phw40k_carry275_20261105.json')
    protocol=read(ROOT/'outputs/experiments/act_recovery_data_prepare_20261105_01/protocol.json')
    protocol.pop('not_done',None)
    protocol.update(status='prepared',created_utc=datetime.now(timezone.utc).isoformat(),
        archive=str(ROOT/manifest['archive']),baseline_id=manifest['baseline_id'],model_sha256=manifest['model_sha256'],
        expert_steps=450,acceptance='Only original full-task completion after expert takeover yields positive expert action weights',
        expert_version=__import__('act_recovery_expert').RecoveryExpert.version,physics_source='Unchanged archived baseline simulator and assets',
        supervision='Model prefix weights zero; failed expert attempts weights zero; only successful expert suffix is positive',
        scope='Three fixed expert recovery collection cases; not ACT evaluation, no old scores changed')
    protocol['cases']=[case for case in protocol['cases'] if case['scene'] in scenes]
    for index,case in enumerate(protocol['cases']):case['gpu']=index%2
    (out/'frozen_scripts').mkdir()
    for name in ('run_act_recovery_collection.py','act_recovery_expert.py','excavator_reachability.py','act_carry_guard.py'):
        shutil.copy2(ROOT/'scripts'/name,out/'frozen_scripts'/name)
    shutil.copy2(ROOT/'outputs/analysis/act_reachability_20261103_01/robot_and_truck_model.json',out/'robot_model.json')
    collision=ROOT/'outputs/analysis/act_bucket_collision_20261105_01.json'
    if collision.exists():
        model=read(out/'robot_model.json');model['bucket_collision_vertices']=read(collision)['vertices']
        model['bucket_collision_source_sha256']=sha(collision);write(out/'robot_model.json',model)
        protocol['progressive_curl']='For initially open bucket only: unwind integrated target, then move toward current dig target while curling with authored bucket/ground checks. Full-task acceptance unchanged.'
    write(out/'protocol.json',protocol)
    write(out/'source_sha256.json',{str(p):sha(p) for p in out.rglob('*') if p.is_file()})
    write(out/'status.json',dict(status='prepared',completed={}))
    return protocol


def controller(out,scenes):
    protocol=prepare(out,scenes);pending=list(protocol['cases']);running=[];completed={};handles=[]
    def stop(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        while pending or running:
            busy={case['gpu'] for case,child in running if child.poll() is None}
            for case in list(pending):
                if case['gpu'] in busy:continue
                log=(out/f'scene{case["scene"]}.log').open('x');handles.append(log)
                command=[sys.executable,'-u',str(out/'frozen_scripts/run_act_recovery_collection.py'),
                    '--output',str(out),'--worker-scene',str(case['scene'])]
                child=subprocess.Popen(command,cwd=out,env=dict(os.environ,ACT_PROJECT_ROOT=str(ROOT)),
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                running.append((case,child));pending.remove(case);busy.add(case['gpu'])
            for case,child in list(running):
                if child.poll() is None:continue
                status=out/f'scene{case["scene"]}/episode_status.json'
                completed[str(case['scene'])]=read(status) if status.exists() else dict(status='error',error='Worker exited without status')
                running.remove((case,child))
                if child.returncode:raise RuntimeError(f'Recovery worker scene{case["scene"]} exited {child.returncode}')
            write(out/'status.json',dict(status='running',completed=completed,
                running=[dict(scene=c['scene'],pid=p.pid,gpu=c['gpu']) for c,p in running]))
            if pending or running:time.sleep(5)
        for path,expected in read(out/'source_sha256.json').items():
            if sha(path)!=expected:raise ValueError(f'Frozen collector input changed: {path}')
        write(out/'results.json',completed)
        write(out/'status.json',dict(status='completed',completed=completed,
            accepted_scenes=[k for k,v in completed.items() if v.get('accepted_for_training')]))
    except BaseException as error:
        write(out/'status.json',dict(status='failed',error=str(error),completed=completed));raise
    finally:
        for case,child in running:
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        for case,child in running:
            if child.poll() is None:
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        for log in handles:log.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker-scene',type=int,choices=[1,9,20],help=argparse.SUPPRESS)
    parser.add_argument('--scenes',default='1,9,20')
    args=parser.parse_args()
    if args.worker_scene:worker(args.output.resolve(),args.worker_scene)
    else:controller(args.output.resolve(),[int(x) for x in args.scenes.split(',')])
