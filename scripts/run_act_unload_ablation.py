"""Matched-state on/off experiment for the frozen ACT unloading pose hold."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(os.environ.get('ACT_PROJECT_ROOT', str(Path(__file__).resolve().parents[1]))).resolve()
SOURCE = ROOT / 'outputs/experiments/act_unload_pose_hold_20261106_01/validation8'
ARCHIVE = ROOT / 'outputs/baselines/act_phw40k_carry275_20261105'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare(out, scenes, repeats, comparison='hold_on_off', source=SOURCE):
    from act_unload_snapshot import patch_simulator
    source=Path(source).resolve()
    out.mkdir(parents=True, exist_ok=False)
    frozen = out / 'frozen_sources'
    frozen.mkdir()
    for path in (source / 'frozen_sources').iterdir():
        if path.name in ('scripts', 'ExcavatorVLA-smolvla'):
            continue
        (frozen/path.name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
    shutil.copytree(source/'frozen_sources/scripts', frozen/'scripts',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    runtime = frozen / 'ExcavatorVLA-smolvla'
    runtime.mkdir()
    for path in (ARCHIVE/'frozen_sources/ExcavatorVLA-smolvla').iterdir():
        if path.name == 'run_simulation.py':
            continue
        (runtime/path.name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
    simulator_source = ARCHIVE/'frozen_sources/ExcavatorVLA-smolvla/run_simulation.py'
    simulator = patch_simulator(simulator_source.read_text())
    # Import only this experiment's snapshot helper, without changing dependencies.
    simulator = simulator.replace('SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))',
        "sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))\n"
        'SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))')
    (runtime/'run_simulation.py').write_text(simulator)
    guard = frozen/'scripts/act_stall_recovery.py'
    code = guard.read_text()
    original = 'if (not self.unload_pose_hold_active and loaded and opening_intent and over_bed'
    if code.count(original) != 1:
        raise ValueError('Unexpected frozen pose-hold implementation')
    code = code.replace(original, 'if (self.unload_pose_hold_enabled and not self.unload_pose_hold_active and loaded and opening_intent and over_bed')
    original = '            self.unload_pose_hold_active = False\n'
    if code.count(original) < 1:
        raise ValueError('Missing frozen pose-hold initialization')
    code = code.replace(original, '            self.unload_pose_hold_enabled = True\n' + original, 1)
    guard.write_text(code)
    if comparison in ('latch_handoff','handoff_wait','controlled_open','geometry_feedback'):
        if comparison=='geometry_feedback':
            from act_unload_geometry import patch_controller as patch
            helper='act_unload_geometry.py'
        elif comparison=='controlled_open':
            from act_unload_controlled_open import patch_controller as patch
            helper='act_unload_controlled_open.py'
        else:
            from act_unload_pose_latch import patch_controller,patch_wait_controller
            patch=patch_wait_controller if comparison=='handoff_wait' else patch_controller
            helper='act_unload_pose_latch.py'
        (frozen/'scripts/act_stall_recovery_latched.py').write_text(patch(code))
        shutil.copy2(ROOT/'scripts'/helper,frozen/'scripts'/helper)
    for name in ('act_unload_snapshot.py', 'run_act_unload_ablation.py'):
        shutil.copy2(ROOT/'scripts'/name, frozen/'scripts'/name)
    shutil.copy2(source/'robot_model.json', out/'robot_model.json')
    if (source/'references').is_dir():shutil.copytree(source/'references', out/'references')
    else:
        (out/'references').mkdir()
        template=read(ROOT/'outputs/experiments/act_phw_optimization_20261103_01/reference_setup.json')
        for scene in scenes:
            folder=source/f'evaluation/episodes/episode_{scene:04d}'
            context=read(folder/'scene_context.json');initial=read(folder/'initial_observation.json')
            ref=dict(template,profile=context['scene_profile'],task=initial['task'],
                initial_joints_deg=[math.degrees(v) for v in initial['state'][:4]],expected_initial_state=initial['state'],
                dig_target_xyz=context['observation_context']['dig_target_xyz'],
                sand_amount=context['scene_profile']['sand_amount_multiplier'],actions=[])
            write(out/'references'/f'scene{scene}.json',ref)
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        baseline_id='act_phw40k_carry275_20261105', archive=str(ARCHIVE),
        checkpoint=str(ARCHIVE/'checkpoints/040000/pretrained_model'),
        candidate_source=str(source), scenes=scenes, repeats=repeats, branch_steps=100,
        total_episode_limit=500, fps=10, replan_interval=3,
        branches='hold_off vs identical frozen controller with hold_on; first repetition also repeats hold_off',
        snapshot_selection='First pre-action state where the frozen hold would start; autonomous common prefix; no outcome-based retries or exclusions',
        matching='Within-process restore of all particle positions/velocities, base pose/velocities, joints/velocities, targets, observation history, scheduler, phase, source-ID tracker; clone client guard, pending actions and RNG',
        initial_images='Reuse the same actual snapshot camera observation for both first actions; subsequent images are live',
        restore_limitations='PhysX contact caches and renderer history are not serialized; both arms are restored, order is balanced, and identical-off controls measure remaining variability. Absolute simulator clock is not rewound; causal deployment elapsed time is restored.',
        repetitions='Fresh autonomous prefix and snapshot for every scene/repetition; repeated states within each scene are clustered development evidence',
        not_reached='Keep the case in coverage report if no eligible unloading state within 500 steps; never replace it with a more favorable scene',
        scoring='Original >=100 initial-pile particles in truck and bucket>=-0.8 rad for three observations within remaining 500-step budget; run fixed branch duration even after first success',
        primary_metric='Fraction of snapshot bucket particle cohort in truck at fixed branch end',
        secondary_metrics=['cohort outside bucket/truck fraction', 'cohort retained in bucket fraction',
            'parent-joint drift from snapshot', 'bucket hinge XY drift', 'opening-pour-point bed gap',
            'original task completion during branch', 'hold-active steps'],
        cohort='Fixed sampled particle IDs physically inside bucket at snapshot; disjoint truck/bucket/outside accounting, no new particle IDs added',
        order='off,on,off_repeat for odd scene-index first repetition; reversed on,off_repeat,off for even index; off/on order alternates on later repetitions',
        advance_gate=dict(minimum_matched_pairs=8, minimum_reached_scenes=3,
            minimum_mean_truck_fraction_gain=.10, minimum_positive_scene_count=3,
            maximum_mean_outside_fraction_increase=.02, no_task_success_count_regression=True,
            maximum_identical_off_truck_fraction_difference=.05),
        scope='Viewed development scenes; unloading ablation only, no new overall success rate or baseline replacement',
        no_training=True, weights_unchanged=True, baseline_replaced=False)
    protocol['comparison']=comparison
    protocol['scene_seeds']={str(s):int(read(out/'references'/f'scene{s}.json')['profile']['seed']) for s in scenes}
    if comparison in ('latch_handoff','handoff_wait','controlled_open','geometry_feedback'):
        protocol.update(branches='off=current combined controller; on=handoff wait and fixed parent goal; first repetition also repeats current controller',
            snapshot_selection='First verified completed_secure_align handback or first legacy hold start, whichever comes first; current-controller autonomous common prefix',
            latch_settings=dict(wait_for_act_opening_steps=20,opening_hold_steps=50,bucket_action='unchanged ACT',
                parent_goal='measured pose captured once at entry',entry='verified secure_align handoff or original hold trigger'),
            problem_scenes=[10,13,28],success_controls=[9,24,30],
            advance_gate=dict(minimum_matched_pairs=8,minimum_problem_pairs=4,minimum_control_pairs=3,
                minimum_problem_truck_gain=.10,minimum_problem_task_success_gain=2,
                maximum_mean_outside_fraction_increase=.02,maximum_identical_off_truck_fraction_difference=.05))
        write(out/'alignment_metric_definition.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
            geometry_windows=['first 50 steps from common handoff, truncated after 3 empty observations', 'full branch'],
            comparison='current combined controller vs handoff pose latch',primary_metrics_and_gate_unchanged=True))
        if comparison=='handoff_wait':
            protocol['branches']='off=current combined controller; on=pre-opening handoff wait with original unloading hold; first repetition also repeats current controller'
            protocol['latch_settings']=dict(wait_for_act_opening_steps=20,opening_hold='legacy implementation preserved verbatim',
                parent_goal='fixed only during pre-opening handoff wait',bucket_action='unchanged ACT')
            protocol['advance_gate'].update(minimum_matched_pairs=4,minimum_problem_pairs=2,minimum_control_pairs=2,
                minimum_problem_task_success_gain=1)
            write(out/'alignment_metric_definition.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
                geometry_windows=['first 50 steps from common handoff, truncated after 3 empty observations', 'full branch'],
                comparison='current combined controller vs pre-opening handoff wait; legacy opening hold unchanged',
                primary_metrics_and_gate_unchanged=True))
        if comparison=='controlled_open':
            protocol.pop('latch_settings')
            protocol.update(branches='off=selected secure_align+unload_pose_hold; on=contained controlled opening fallback inside unload_pose_hold; first repetition repeats current',
                success_controls=[8,9,33],
                controlled_open=dict(maximum_bucket_speed_rad_s=.8,target_bucket_rad=-.4,
                    entry='verified contained secure_align handoff, residual<=.015m, loaded, over bed, original hold inactive',
                    parent_target='original measured parent pose',
                    release='original 50-step counter or three empty observations',
                    regular_act_triggered_hold='unchanged',handoff_wait_enabled=False))
            protocol['advance_gate'].update(minimum_matched_pairs=4,minimum_problem_pairs=2,minimum_control_pairs=2,
                minimum_problem_task_success_gain=1,minimum_branch_steps_for_coverage=50,
                maximum_control_truck_fraction_loss=.05)
            write(out/'alignment_metric_definition.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
                geometry_windows=['first 50 steps from common handoff, truncated after 3 empty observations','full branch'],
                comparison='selected 57.5% controller vs controlled opening inside unloading hold',
                note='No handoff-wait component; bucket action is explicitly controlled in the fallback only'))
        if comparison=='geometry_feedback':
            protocol.pop('latch_settings')
            protocol.update(branches='off=selected 57.5% controller; on=aperture alignment, hold and bounded opening; first repetition repeats current',
                success_controls=[8,9,33],
                geometry_feedback=dict(aperture='Full collision width across verified open-bucket profile chord',
                    controller_version='aperture_geometry_v2_measured_servo',
                    parent_control='Integrate measured position error with gain 2/s; bound command targets to the local correction box',
                    plan_inset_m=.15,measured_opening_margin_m=.10,
                    maximum_parent_delta_rad=[.12,.12,.16],maximum_parent_speed_rad_s=[.12,.12,.16],
                    target_bucket_rad=-.4,maximum_bucket_speed_rad_s=.8,
                    maximum_alignment_steps=20,maximum_total_steps=50,minimum_remaining_episode_steps=50,
                    measured_alignment_observations=2,regular_act_triggered_hold='unchanged',
                    handoff_wait_enabled=False,model_observation_landmarks='unchanged'))
            protocol['advance_gate'].update(minimum_matched_pairs=4,minimum_problem_pairs=2,minimum_control_pairs=2,
                minimum_problem_task_success_gain=1,minimum_branch_steps_for_coverage=50,
                maximum_control_truck_fraction_loss=.05,minimum_affected_control_pairs=2)
            write(out/'alignment_metric_definition.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
                geometry_windows=['first 50 steps from common handoff, truncated after 3 empty observations','full branch'],
                comparison='selected 57.5% controller vs measured aperture alignment then opening',
                note='Primary metrics remain fixed-cohort transfer and original task completion; generic window drift is descriptive only'))
    write(out/'protocol.json', protocol)
    paths=[*frozen.glob('scripts/*.py'),runtime/'run_simulation.py',out/'protocol.json',out/'robot_model.json',
           *out.glob('references/*.json'), *Path(protocol['checkpoint']).glob('*'),
           *out.glob('alignment_metric_definition.json')]
    write(out/'inputs_sha256.json', {str(p):sha(p) for p in paths if p.is_file()})
    write(out/'status.json',dict(status='prepared', expected_cases=len(scenes)*repeats))
    return protocol


def geometry_metrics(state, landmarks, scene, initial_state, initial_landmarks):
    import numpy as np
    from act_carry_guard import opening_pour_point
    def pivot(points):
        tip,load,pour=[np.asarray(points[k]) for k in ('tip','load','pour')]
        x,z=np.linalg.solve(np.array([[.4,-.26],[.5,.22]]),np.stack([tip-load,pour-load]))
        return load-.35*x-.08*z
    delta=np.asarray(state[:3])-np.asarray(initial_state[:3])
    delta[0]=np.arctan2(np.sin(delta[0]),np.cos(delta[0]))
    bed=scene['truck_counting_volume']
    transform=np.asarray(bed['local_from_world'])
    point=opening_pour_point(state,landmarks,-.8)
    local=(np.r_[point,1.]@transform)[:3]
    scales=np.linalg.norm(np.linalg.inv(transform)[:3,:3],axis=1)
    excess=np.maximum(np.maximum(np.asarray(bed['local_min'])[:2]-local[:2],
                                  local[:2]-np.asarray(bed['local_max'])[:2]),0.)
    return dict(parent_abs_drift_rad=np.abs(delta).tolist(),
        hinge_xy_drift_m=float(np.linalg.norm((pivot(landmarks)-pivot(initial_landmarks))[:2])),
        opening_pour_bed_gap_m=float(np.linalg.norm(excess*scales[:2])))


def worker(out, scene_index, repetition, gpu, probe=False):
    protocol=read(out/'protocol.json')
    inventory=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],check=True,capture_output=True,text=True).stdout.splitlines()
    os.environ.update(CUDA_VISIBLE_DEVICES=inventory[gpu],HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',
        HF_HOME=str(ROOT/'models/hf-cache'),TORCH_HOME=str(ROOT/'models/torch-cache'),OMP_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8',
        PYTHONDONTWRITEBYTECODE='1',ACT_POSTDIG_RECOVERY='0')
    frozen=out/'frozen_sources'
    sys.path[:0]=[str(frozen/'scripts'),str(ARCHIVE/'runtime_dependencies'),str(frozen/'ExcavatorVLA-vlm')]
    import numpy as np
    import torch
    from eval_smolvla_success import (SimulatorSession,exchange,validate_state,decoded_frames,frame_hashes,
        build_client_contract,OBSERVATION_SCHEMA_28D_V4_PLUS_EFFORT,CompletionTracker,EpisodeVideo,
        policy_batch,PreTrainedConfig,get_policy_class,make_pre_post_processors)
    from excavator_common.simulation_settings import resolve_simulation_settings,apply_simulation_settings
    from excavator_common.deployment_contract import sha256_files
    from act_carry_guard import CarryAttitudeGuard,CarryGuardSettings
    from act_stall_recovery import make_stall_guard_class
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    seed=protocol.get('scene_seeds',{}).get(str(scene_index),2000000+scene_index-1)
    torch.manual_seed(seed+10000*repetition)
    np.random.seed(seed+10000*repetition)
    checkpoint=Path(protocol['checkpoint'])
    config=PreTrainedConfig.from_pretrained(str(checkpoint))
    config.device='cuda';config.pretrained_backbone_weights=None
    policy=get_policy_class('act').from_pretrained(str(checkpoint),config=config,strict=True,local_files_only=True).to('cuda').eval()
    dataset_info=read(ROOT/'data/export_phw/meta/info.json')
    action_limits=np.asarray(dataset_info['action_limits_rad_s'],dtype=np.float32)
    if dataset_info['fps']!=10:raise ValueError('Unexpected data action cadence')
    preprocess,postprocess=make_pre_post_processors(policy.config,pretrained_path=str(checkpoint),
        preprocessor_overrides={'device_processor':{'device':'cuda'}})
    case_name='probe' if probe else f'scene{scene_index}_repeat{repetition}'
    folder=out/case_name;folder.mkdir()
    ref=read(out/'references'/f'scene{scene_index}.json')
    write(folder/'scene_profile.json',ref['profile'])
    command=[str(frozen/'run_isaac_python.sh'),str(frozen/'ExcavatorVLA-smolvla/run_simulation.py'),
        '--sand-amount',str(ref['sand_amount']),'--active-cuda-gpu','0','--cpu-threads','8',
        '--robot-initial-joints-deg',*map(str,ref['initial_joints_deg']),
        '--headless','--idle-ui-hz','.1','--render-every-physics-step',
        '--scene-profile',str(folder/'scene_profile.json'),
        '--state27-dig-target-world',*map(str,ref['dig_target_xyz']),
        '--state27-unload-target-world',*map(str,ref['profile']['unload_landing_xyz']),
        '--task-text',ref['task']]
    apply_simulation_settings(resolve_simulation_settings('physx',False))
    session=SimulatorSession(command,frozen/'ExcavatorVLA-smolvla',folder/'simulator.log',True,exchange,600)
    status=dict(status='starting',scene=scene_index,repetition=repetition,gpu=gpu)
    write(folder/'status.json',status)
    def interrupted(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def infer(observation):
        frames=decoded_frames(observation)
        with torch.inference_mode():
            chunk=policy.predict_action_chunk(preprocess(policy_batch(frames,validate_state(observation),observation['task_text'])))
            return [postprocess(chunk[:,i]).cpu().numpy()[0] for i in range(3)]
    def row(step,before,after,predicted,applied,trace,scene,initial):
        state=validate_state(after)
        return dict(step=step,predicted=predicted.tolist(),applied=applied.tolist(),input_state=validate_state(before).tolist(),
            input_bucket_landmarks_world=before['bucket_landmarks_world'],
            input_joint_command_positions=before['joint_command_positions'],
            bucket_landmarks_world=after['bucket_landmarks_world'],joint_command_positions=after['joint_command_positions'],
            state=state.tolist(),carry_guard=trace,particle_transfer_metrics=after['particle_transfer_metrics'],
            unload_cohort_metrics=after.get('unload_cohort_metrics'),
            geometry=geometry_metrics(state,after['bucket_landmarks_world'],scene,
                                      initial['observation_state_28d'],initial['bucket_landmarks_world']))
    try:
        status.update(session.prepare(seed))
        scene=exchange(session.sock,{'type':'scene_context'})
        write(folder/'scene_context.json',scene)
        normalization=sha256_files(sorted(checkpoint.glob('*processor*')))
        contract=build_client_contract(10,normalization,OBSERVATION_SCHEMA_28D_V4_PLUS_EFFORT,scene['observation_context'])
        ack=exchange(session.sock,contract)
        if ack.get('type')!='handshake_ack':raise ValueError('Invalid handshake')
        reply=exchange(session.sock,{'joint_velocities':[0.,0.,0.,0.]})
        initial_state=validate_state(reply)
        if scene['scene_seed']!=seed or scene['simulation_settings']!=resolve_simulation_settings('physx',False):
            raise ValueError('Scene seed or simulation settings differ from the protocol')
        # A supplied profile is loaded on attempt zero; sample_try records the
        # historical planner attempt and is not part of physical geometry.
        geometry_profile=lambda profile:{k:v for k,v in profile.items() if k!='sample_try'}
        if geometry_profile(scene['scene_profile'])!=geometry_profile(ref['profile']):
            raise ValueError('Actual scene geometry differs from the frozen reference')
        if np.max(np.abs(initial_state[16:26]-np.asarray(ref['expected_initial_state'])[16:26]))>.05:
            raise ValueError('Initial observation geometry differs from the frozen reference')
        if np.max(np.abs(initial_state[:4]-np.radians(ref['initial_joints_deg'])))>5e-4:
            raise ValueError('Initial articulation pose differs from the frozen reference')
        if reply['task_text']!=ref['task']:
            raise ValueError('Task instruction differs from the frozen reference')
        write(folder/'initial_observation.json',dict(state=validate_state(reply).tolist(),camera_sha256=frame_hashes(decoded_frames(reply))))
        if probe:
            saved=exchange(session.sock,{'type':'unload_snapshot_save','snapshot_id':'probe'})
            audits=[]
            for _ in range(2):
                for _movement in range(10):
                    exchange(session.sock,{'joint_velocities':[.1,.05,0.,0.]})
                restored=exchange(session.sock,{'type':'unload_snapshot_restore','snapshot_id':'probe'})
                audits.append(restored['audit'])
                reply=exchange(session.sock,{'joint_velocities':[0.,0.,0.,0.]})
                validate_state(reply)
            write(folder/'result.json',dict(status='completed',snapshot=saved,audits=audits))
            write(folder/'status.json',dict(status='completed'))
            return
        guard_class=make_stall_guard_class(CarryAttitudeGuard,read(out/'robot_model.json'),'secure_align')
        guard=guard_class(scene,action_limits=action_limits,settings=CarryGuardSettings(parent_projection='arm_first',release_geometry='opening_pose'))
        latch_comparison=protocol.get('comparison') in ('latch_handoff','handoff_wait','controlled_open','geometry_feedback')
        guard.unload_pose_hold_enabled=latch_comparison
        policy.reset();actions=[];eligible=False
        video=EpisodeVideo(folder/'prefix.mp4',scene_index,seed,10,500,100)
        prefix_completion=CompletionTracker(100)
        try:
            with (folder/'prefix_steps.jsonl').open('x') as trace:
                for step in range(protocol['total_episode_limit']):
                    if not actions:actions=infer(reply)
                    predicted=actions.pop(0)
                    predicted=np.clip(predicted,-action_limits,action_limits)
                    preview=copy.deepcopy(guard);preview.unload_pose_hold_enabled=True
                    _,preview_trace=preview.update(validate_state(reply),predicted,reply['joint_command_positions'],reply['bucket_landmarks_world'])
                    hold=preview_trace.get('stall_recovery',{}).get('unload_pose_hold',{})
                    event=preview_trace.get('stall_recovery',{})
                    if hold.get('started') or (latch_comparison and event.get('handback') and event.get('phase')=='completed_secure_align'):
                        eligible=True;break
                    before=reply
                    applied,guard_trace=guard.update(validate_state(reply),predicted,reply['joint_command_positions'],reply['bucket_landmarks_world'])
                    if guard_trace.get('stall_recovery',{}).get('handback'):
                        actions=[];policy.reset()
                    reply=exchange(session.sock,{'joint_velocities':applied.tolist()})
                    validate_state(reply)
                    trace.write(json.dumps(dict(step=step,predicted=predicted.tolist(),applied=applied.tolist(),
                        state=validate_state(reply).tolist(),carry_guard=guard_trace,
                        particle_transfer_metrics=reply['particle_transfer_metrics']))+'\n');trace.flush()
                    count=reply['particle_transfer_metrics']['payload_particles']
                    if prefix_completion.observe(count,float(reply['joint_positions'][3])):
                        status['prefix_completed_before_eligible']=True
                        break
                    video.write(reply,dict(policy_steps_executed=step+1,final_newly_deposited_particle_count=count,
                        final_bucket_open=float(reply['joint_positions'][3])>=-.8,status='common_prefix',success=False))
                    if step%25==0:
                        write(folder/'status.json',dict(status, status='prefix',step=step+1))
        finally:
            video.finish(dict(status='common_prefix',success=False,policy_steps_executed=step))
        if not eligible:
            result=dict(status='not_reached',scene=scene_index,repetition=repetition,
                prefix_steps=step+1,prefix_completed_before_eligible=status.get('prefix_completed_before_eligible',False),
                terminal_particles=reply['particle_transfer_metrics'])
            write(folder/'result.json',result);write(folder/'status.json',result)
            return
        snapshot=exchange(session.sock,{'type':'unload_snapshot_save','snapshot_id':case_name,
                                      'physical_path':str(folder/'physical_snapshot.npz')})
        if snapshot['cohort_count']<=0:raise ValueError('Eligible hold has no snapshot bucket cohort')
        initial=copy.deepcopy(reply)
        write(folder/'snapshot.json',dict(snapshot,prefix_steps=step,state=validate_state(reply).tolist(),
            camera_sha256=frame_hashes(decoded_frames(reply)),joint_command_positions=reply['joint_command_positions'],
            landmarks=reply['bucket_landmarks_world'],particles=reply['particle_transfer_metrics'],
            first_predicted_action=predicted.tolist(),queued_actions=[a.tolist() for a in actions],
            guard_state_repr=repr(guard.__dict__),physical_file_sha256=sha(folder/'physical_snapshot.npz')))
        initial_guard=copy.deepcopy(guard)
        initial_actions=[predicted.copy(),*[a.copy() for a in actions]]
        torch_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state();numpy_rng=np.random.get_state()
        case_order=protocol['scenes'].index(scene_index)
        order=(['off','on','off_repeat'] if case_order%2==0 else ['on','off_repeat','off']) if repetition==1 else (
            ['off','on'] if (case_order+repetition)%2 else ['on','off'])
        horizon=min(protocol['branch_steps'],protocol['total_episode_limit']-step)
        results={}
        for branch in order:
            restored=exchange(session.sock,{'type':'unload_snapshot_restore','snapshot_id':case_name})
            reply=restored['observation']
            if reply!=initial:raise ValueError('Restored initial model observation differs')
            if restored['audit']['particle_positions_sha256']!=snapshot['particle_positions_sha256']:
                raise ValueError('Restored particle position hash differs')
            if restored['audit']['particle_velocities_sha256']!=snapshot['particle_velocities_sha256']:
                raise ValueError('Restored particle velocity hash differs')
            branch_folder=folder/branch;branch_folder.mkdir()
            write(branch_folder/'restore_audit.json',dict(audit=restored['audit'],initial_observation_equal=True,
                first_camera_sha256=frame_hashes(decoded_frames(reply))))
            torch.set_rng_state(torch_rng);torch.cuda.set_rng_state(cuda_rng);np.random.set_state(numpy_rng)
            policy.reset()
            if latch_comparison and branch=='on':
                from act_stall_recovery_latched import make_stall_guard_class as make_latched_guard
                candidate_class=make_latched_guard(CarryAttitudeGuard,read(out/'robot_model.json'),'secure_align')
                guard=candidate_class(scene,action_limits=action_limits,settings=copy.deepcopy(initial_guard.settings))
                guard.__dict__.update(copy.deepcopy(initial_guard.__dict__))
            else:
                guard=copy.deepcopy(initial_guard)
                guard.unload_pose_hold_enabled=True if latch_comparison else branch=='on'
            actions=[a.copy() for a in initial_actions]
            completion=copy.deepcopy(prefix_completion)
            video=EpisodeVideo(branch_folder/'evaluation.mp4',scene_index,seed,10,horizon,100)
            rows=[];first_success=None
            write(folder/'status.json',dict(status='branch',scene=scene_index,repetition=repetition,branch=branch,horizon=horizon))
            try:
                with (branch_folder/'steps.jsonl').open('x') as trace:
                    for index in range(horizon):
                        if not actions:actions=infer(reply)
                        predicted=actions.pop(0)
                        predicted=np.clip(predicted,-action_limits,action_limits)
                        applied,guard_trace=guard.update(validate_state(reply),predicted,reply['joint_command_positions'],reply['bucket_landmarks_world'])
                        if guard_trace.get('stall_recovery',{}).get('handback'):
                            actions=[];policy.reset()
                        before=reply
                        reply=exchange(session.sock,{'joint_velocities':applied.tolist()})
                        entry=row(step+index,before,reply,predicted,applied,guard_trace,scene,initial)
                        cohort=entry['unload_cohort_metrics']
                        if sum(cohort[k] for k in ('bucket','truck','outside'))!=snapshot['cohort_count']:
                            raise ValueError('Snapshot particle cohort does not conserve IDs')
                        success=completion.observe(reply['particle_transfer_metrics']['payload_particles'],float(reply['joint_positions'][3]))
                        if success and first_success is None:first_success=step+index
                        rows.append(entry);trace.write(json.dumps(entry)+'\n');trace.flush()
                        video.write(reply,dict(policy_steps_executed=index+1,status=branch,
                            final_newly_deposited_particle_count=reply['particle_transfer_metrics']['payload_particles'],
                            final_bucket_open=float(reply['joint_positions'][3])>=-.8,
                            success=first_success is not None),completion.consecutive)
            finally:
                video.finish(dict(status=branch,success=first_success is not None,
                    task_completed_success_and_bucket_open=first_success is not None,policy_steps_executed=len(rows),
                    final_newly_deposited_particle_count=reply['particle_transfer_metrics']['payload_particles'],
                    final_bucket_open=float(reply['joint_positions'][3])>=-.8))
            terminal=copy.deepcopy(reply)
            # Settling is separate from scoring, with exactly the same 3 s diagnostic.
            for _ in range(30):reply=exchange(session.sock,{'joint_velocities':[0.,0.,0.,0.]})
            cohort=terminal['unload_cohort_metrics'];denominator=cohort['total']
            result=dict(branch=branch,scene=scene_index,repetition=repetition,prefix_steps=step,
                branch_steps=horizon,cohort_count=denominator,
                truck_fraction=cohort['truck']/denominator,outside_fraction=cohort['outside']/denominator,
                retained_fraction=cohort['bucket']/denominator,cohort_terminal=cohort,
                settled_cohort=reply['unload_cohort_metrics'],
                original_task_completed=first_success is not None,first_success_step=first_success,
                terminal_particles=terminal['particle_transfer_metrics'],
                max_parent_abs_drift_rad=np.max([r['geometry']['parent_abs_drift_rad'] for r in rows],axis=0).tolist(),
                max_hinge_xy_drift_m=max(r['geometry']['hinge_xy_drift_m'] for r in rows),
                max_opening_pour_bed_gap_m=max(r['geometry']['opening_pour_bed_gap_m'] for r in rows),
                hold_active_steps=sum(r['carry_guard'].get('stall_recovery',{}).get('unload_pose_hold',{}).get('active',False) for r in rows),
                handoff_wait_active_steps=sum(r['carry_guard'].get('stall_recovery',{}).get('unload_handoff_wait',{}).get('active',False) for r in rows),
                controlled_open_active_steps=sum(r['carry_guard'].get('stall_recovery',{}).get('unload_pose_hold',{}).get('controlled_open',{}).get('active',False) for r in rows),
                geometry_feedback_active_steps=sum(r['carry_guard'].get('stall_recovery',{}).get('unload_pose_hold',{}).get('geometry_feedback',{}).get('active',False) for r in rows),
                restore_audit_passed=True)
            results[branch]=result
            write(branch_folder/'result.json',result)
        result=dict(status='completed',scene=scene_index,repetition=repetition,prefix_steps=step,
                    cohort_count=snapshot['cohort_count'],order=order,branches=results)
        write(folder/'result.json',result);write(folder/'status.json',dict(status='completed',scene=scene_index,repetition=repetition))
    except BaseException as error:
        write(folder/'status.json',dict(status='failed',error=str(error),scene=scene_index,repetition=repetition))
        raise
    finally:
        session.close()


def summarize(out):
    protocol=read(out/'protocol.json')
    records={p.parent.name:read(p) for p in out.glob('scene*_repeat*/result.json')}
    paired=[r for r in records.values() if r['status']=='completed']
    aa=[abs(r['branches']['off']['truck_fraction']-r['branches']['off_repeat']['truck_fraction'])
        for r in paired if 'off_repeat' in r['branches']]
    differences=[r['branches']['on']['truck_fraction']-r['branches']['off']['truck_fraction'] for r in paired]
    outside=[r['branches']['on']['outside_fraction']-r['branches']['off']['outside_fraction'] for r in paired]
    by_scene={str(s):[r['branches']['on']['truck_fraction']-r['branches']['off']['truck_fraction']
                     for r in paired if r['scene']==s] for s in protocol['scenes']}
    scene_means={s:sum(values)/len(values) for s,values in by_scene.items() if values}
    mean=lambda values:sum(values)/len(values) if values else None
    successes={b:sum(r['branches'][b]['original_task_completed'] for r in paired) for b in ('off','on')}
    gate=protocol['advance_gate']
    latch_comparison=protocol.get('comparison') in ('latch_handoff','handoff_wait','controlled_open','geometry_feedback')
    if latch_comparison:
        problem=[r for r in paired if r['scene'] in protocol['problem_scenes']]
        control=[r for r in paired if r['scene'] in protocol['success_controls']]
        group_success=lambda group,branch:sum(r['branches'][branch]['original_task_completed'] for r in group)
        problem_gain=[r['branches']['on']['truck_fraction']-r['branches']['off']['truck_fraction'] for r in problem]
        checks=dict(coverage=len(paired)>=gate['minimum_matched_pairs'] and len(problem)>=gate['minimum_problem_pairs'] and len(control)>=gate['minimum_control_pairs'],
            problem_truck_gain=bool(problem_gain) and mean(problem_gain)>=gate['minimum_problem_truck_gain'],
            problem_task_success=group_success(problem,'on')-group_success(problem,'off')>=gate['minimum_problem_task_success_gain'],
            success_controls=group_success(control,'on')>=group_success(control,'off'),
            spillage=bool(outside) and mean(outside)<=gate['maximum_mean_outside_fraction_increase'],
            repeatability=bool(aa) and max(aa)<=gate['maximum_identical_off_truck_fraction_difference'])
        if protocol['comparison'] in ('controlled_open','geometry_feedback'):
            enough=lambda r:min(r['branches'][b]['branch_steps'] for b in ('off','on'))>=gate['minimum_branch_steps_for_coverage']
            checks['coverage']=checks['coverage'] and sum(enough(r) for r in problem)>=gate['minimum_problem_pairs'] and sum(enough(r) for r in control)>=gate['minimum_control_pairs']
            checks['control_material']=bool(control) and all(r['branches']['on']['truck_fraction']-r['branches']['off']['truck_fraction']>=-gate['maximum_control_truck_fraction_loss'] for r in control)
            if protocol['comparison']=='geometry_feedback':
                checks['affected_controls']=sum(r['branches']['on'].get('geometry_feedback_active_steps',0)>0 and enough(r) for r in control)>=gate['minimum_affected_control_pairs']
    else:
        checks=dict(coverage=len(paired)>=gate['minimum_matched_pairs'] and len(scene_means)>=gate['minimum_reached_scenes'],
            truck_gain=bool(differences) and mean(differences)>=gate['minimum_mean_truck_fraction_gain'],
            scene_consistency=sum(v>0 for v in scene_means.values())>=gate['minimum_positive_scene_count'],
            spillage=bool(outside) and mean(outside)<=gate['maximum_mean_outside_fraction_increase'],
            task_success=successes['on']>=successes['off'],
            repeatability=bool(aa) and max(aa)<=gate['maximum_identical_off_truck_fraction_difference'])
    summary=dict(cases=len(records),matched_pairs=len(paired),not_reached=sum(r['status']=='not_reached' for r in records.values()),
        mean_truck_fraction_difference=mean(differences),mean_outside_fraction_difference=mean(outside),
        scene_mean_truck_fraction_difference=scene_means,original_task_successes=successes,
        identical_off_max_truck_fraction_difference=max(aa) if aa else None,
        checks=checks,advance=all(checks.values()),baseline_replaced=False,results=records)
    if latch_comparison:
        summary.update(problem_pairs=len(problem),control_pairs=len(control),
            mean_problem_truck_fraction_difference=mean(problem_gain),
            problem_successes={b:group_success(problem,b) for b in ('off','on')},
            control_successes={b:group_success(control,b) for b in ('off','on')})
    write(out/'results.json',summary)
    lines=['# ACT卸料姿态保持：相同状态开关对照','',
        '仅切换冻结候选的卸料姿态保持。两组均从同一物理快照恢复；不更新模型，不替换历史基准。',
        '快照核对沙粒位置/速度、机器人位姿/速度与命令目标；首帧使用共同快照的真实相机观测，后续为实时图像。',
        'PhysX接触缓存与渲染历史未序列化；以同状态重复关闭组衡量剩余仿真差异。旧场景仅作开发消融。','',
        '| 场景 | 重复 | 快照斗内粒子 | 关闭入车比例 | 开启入车比例 | 关闭/开启任务成功 | 关闭/开启最大斗铰点XY漂移 m |',
        '| --- | --- | ---: | ---: | ---: | --- | --- |']
    for r in sorted(records.values(),key=lambda r:(r['scene'],r['repetition'])):
        if r['status']!='completed':
            lines.append(f"| {r['scene']} | {r['repetition']} | 未到达可配对卸料状态 | — | — | — | — |")
            continue
        a,b=r['branches']['off'],r['branches']['on']
        lines.append(f"| {r['scene']} | {r['repetition']} | {r['cohort_count']} | {a['truck_fraction']:.3f} | {b['truck_fraction']:.3f} | {a['original_task_completed']}/{b['original_task_completed']} | {a['max_hinge_xy_drift_m']:.3f}/{b['max_hinge_xy_drift_m']:.3f} |")
    lines += ['',f"已配对 {len(paired)}，未到达 {summary['not_reached']}。",f"平均入车比例差（开启−关闭）：{mean(differences)}。",
        f"平均斗外且车外比例差：{mean(outside)}。",f"同状态重复关闭组的最大入车比例差：{summary['identical_off_max_truck_fraction_difference']}。",
        f"预先固定的推进检查：{checks}；推进：{summary['advance']}。",'',
        '固定快照斗内粒子ID作为分母；不是全程挖掘成功率。枝末3秒零速度观察单独保存，不改原任务完成评分。',
        '未到达卸料状态的案例保留在覆盖统计中，不更换场景、不按结果追加重复或修改保持参数。']
    if latch_comparison:
        lines=[line.replace('相同状态开关对照','相同交接状态：当前版与锁定姿态版')
            .replace('仅切换冻结候选的卸料姿态保持。','比较57.5%当前组合与交接立即等待开斗＋固定姿态目标候选。')
            .replace('关闭','当前版').replace('开启','候选版') for line in lines]
        if protocol['comparison']=='handoff_wait':
            lines=[line.replace('当前版与锁定姿态版','当前版与开斗前交接等待版')
                .replace('交接立即等待开斗＋固定姿态目标候选','仅在开斗前等待、保留原卸料保持逻辑的候选') for line in lines]
        if protocol['comparison']=='controlled_open':
            lines=[line.replace('当前版与锁定姿态版','57.5%版与卸料保持主动开斗兜底')
                .replace('交接立即等待开斗＋固定姿态目标候选','原卸料保持内的限速主动开斗兜底候选') for line in lines]
        if protocol['comparison']=='geometry_feedback':
            lines=[line.replace('当前版与锁定姿态版','57.5%版与斗口校正后卸料版')
                .replace('交接立即等待开斗＋固定姿态目标候选','真实斗口位置校正、保持和限速开斗的候选') for line in lines]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return summary


def run(out):
    protocol=read(out/'protocol.json')
    jobs=[(s,r) for r in range(1,protocol['repeats']+1) for s in protocol['scenes']]
    active={}
    def interrupted(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        while jobs or active:
            for gpu in (0,1):
                if gpu in active or not jobs:continue
                scene,repetition=jobs.pop(0);name=f'scene{scene}_repeat{repetition}'
                log=(out/(name+'.log')).open('x')
                command=[sys.executable,'-u',str(out/'frozen_sources/scripts/run_act_unload_ablation.py'),
                    '--output',str(out),'--worker','--scene',str(scene),'--repetition',str(repetition),'--gpu',str(gpu)]
                child=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,ACT_PROJECT_ROOT=str(ROOT)),
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[gpu]=dict(child=child,log=log,name=name)
            for gpu,job in list(active.items()):
                code=job['child'].poll()
                if code is None:continue
                if code:raise RuntimeError(f"{job['name']} exited {code}; see log")
                job['log'].close();del active[gpu]
                summary=summarize(out)
                print(json.dumps(dict(finished=job['name'],matched=summary['matched_pairs'],not_reached=summary['not_reached'])),flush=True)
            write(out/'status.json',dict(status='running',completed_cases=len(list(out.glob('scene*_repeat*/result.json'))),
                expected_cases=len(protocol['scenes'])*protocol['repeats'],
                running=[dict(name=j['name'],pid=j['child'].pid,gpu=g) for g,j in active.items()]))
            if jobs or active:time.sleep(5)
        changed=[name for name,digest in read(out/'inputs_sha256.json').items() if sha(name)!=digest]
        if changed:raise ValueError(f'Protected inputs changed: {changed}')
        summary=summarize(out)
        write(out/'status.json',dict(status='completed',completed_cases=summary['cases'],matched_pairs=summary['matched_pairs'],
            not_reached=summary['not_reached'],advance=summary['advance'],inputs_unchanged=True,baseline_replaced=False,running=[]))
    except BaseException as error:
        write(out/'status.json',dict(status='failed',error=str(error)))
        raise
    finally:
        for job in active.values():
            if job['child'].poll() is None:
                os.killpg(job['child'].pid,signal.SIGTERM)
                try:job['child'].wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(job['child'].pid,signal.SIGKILL)
            job['log'].close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scenes',type=int,nargs='+',default=[7,16,29,33])
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--run-prepared',action='store_true')
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--probe',action='store_true')
    parser.add_argument('--scene',type=int,default=29)
    parser.add_argument('--repetition',type=int,default=1)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--comparison',choices=['hold_on_off','latch_handoff','handoff_wait','controlled_open','geometry_feedback'],default='hold_on_off')
    parser.add_argument('--source',type=Path,default=SOURCE)
    args=parser.parse_args();out=args.output.resolve()
    if args.worker:
        worker(out,args.scene,args.repetition,args.gpu,args.probe)
    else:
        if not args.run_prepared:prepare(out,args.scenes,args.repeats,args.comparison,args.source)
        if not args.prepare_only:run(out)


if __name__=='__main__':main()
