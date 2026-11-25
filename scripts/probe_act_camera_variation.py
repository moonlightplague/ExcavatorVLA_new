"""Build isolated capture diagnostics; do not mutate the deployed simulator."""
import argparse,ast,hashlib,json,os,shutil,subprocess,sys,time,textwrap
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from eval_smolvla_repeat40 import read,atomic_json,now,digest,evaluation_environment
RUNTIME=ROOT/'outputs/experiments/act_training_alternatives_runtime_20261119_01/python'
BASELINE=ROOT/'outputs/experiments/act_velocity_holdout8_20261119_01'

PROBE_COMMAND=r'''
            if cmd.get('type') == 'diagnostic_capture':
                try:
                    from pathlib import Path as ProbePath
                    import hashlib as probe_hashlib
                    import zlib as probe_zlib
                    import base64 as probe_base64
                    from PIL import Image as ProbeImage
                    probe_dir = ProbePath(cmd['output']).resolve()
                    probe_dir.mkdir(parents=True, exist_ok=False)
                    probe_settings = carb.settings.get_settings()
                    keys = ['/rtx/rendermode', '/rtx/post/aa/op', '/rtx/post/dlss/execMode',
                        '/rtx/post/tonemap/autoExposure/enabled', '/rtx/post/histogram/enabled', '/app/asyncRendering', '/rtx/rtpt/cached/enabled', '/rtx/rtpt/lightcache/cached/enabled']
                    original_settings = {key: probe_settings.get(key) for key in keys}
                    original_aa = probe_settings.get('/rtx/post/aa/op')
                    def probe_physical():
                        points = np.asarray(sand_api['particle_positions_fn'](), dtype=np.float32).copy()
                        joints = np.asarray(robot.get_joint_positions(), dtype=np.float32).copy()
                        return {'time': float(world.current_time), 'joints': joints.tolist(),
                                'particles_sha256': probe_hashlib.sha256(points.tobytes()).hexdigest()}, points
                    physical_before, initial_points = probe_physical()
                    np.save(probe_dir/'initial_particles.npy', initial_points)
                    np.savez(probe_dir/'initial_camera_transforms.npz', **{
                        name: np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=np.float64)
                        for name,path in active_camera_paths.items()})
                    probe_stages = []
                    for mode in ['single_hdr', 'authored_restored', 'single_hdr_again']:
                        control_dome = stage.GetPrimAtPath('/World/ControlRig/DomeLight')
                        if not control_dome.IsValid(): raise RuntimeError('Expected authored control dome is absent')
                        UsdGeom.Imageable(control_dome).CreateVisibilityAttr().Set('inherited' if mode == 'authored_restored' else 'invisible')
                        if mode == 'default_restored':
                            if original_aa is None: raise RuntimeError('Original AA setting is absent; cannot restore')
                            probe_settings.set('/rtx/post/aa/op', original_aa)
                        applied = apply_stable_dataset_render_settings(reason='isolated_camera_probe') if mode == 'collection_settings' else {}
                        folder = probe_dir/mode; folder.mkdir()
                        buffers = {}; steps = []; roundtrip = True
                        for capture_index in range(24):
                            images = capture_rgb_from_persistent_viewports(capture_views, simulation_app, capture_viewport_to_buffer, np, wait_frames=CAPTURE_WAIT_FRAMES)
                            for name,value in images.items():
                                value = np.ascontiguousarray(value)
                                buffers.setdefault(name,[]).append(value)
                                encoded = probe_base64.b64encode(probe_zlib.compress(value.tobytes(),level=1))
                                decoded = np.frombuffer(probe_zlib.decompress(probe_base64.b64decode(encoded)),dtype=value.dtype).reshape(value.shape)
                                roundtrip &= np.array_equal(value, decoded)
                                if capture_index in [0,1,7,15,23]: ProbeImage.fromarray(value).save(folder/f'camera_{name}_{capture_index:02d}.png')
                            physical, unused = probe_physical()
                            steps.append(dict(capture_index=capture_index,render_updates=LAST_CAPTURE_TIMING_MS.get('camera_render_updates'),physical=physical))
                        cameras = {}
                        for name,values in buffers.items():
                            frames = np.stack(values)
                            np.save(folder/f'camera_{name}.npy',frames)
                            diff = np.abs(frames[1:].astype(np.int16)-frames[:-1].astype(np.int16))
                            tail = diff[-8:]
                            cameras[name] = {'unique_hashes':len(set(probe_hashlib.sha256(frame.tobytes()).hexdigest() for frame in frames)),
                                'consecutive_mae':float(diff.mean()),'last8_consecutive_mae':float(tail.mean()),
                                'last8_changed_pixel_fraction':float(np.any(tail>0,axis=-1).mean()),
                                'last8_max_abs_difference':int(tail.max()),'shape':list(frames.shape)}
                        final_physical, final_points = probe_physical()
                        fixed = all(step['physical']==physical_before for step in steps)
                        stage_report={'mode':mode,'settings':{key:probe_settings.get(key) for key in keys},'applied':applied,
                            'physical_fixed':fixed,'transport_roundtrip_exact':bool(roundtrip),'cameras':cameras,'captures':steps}
                        (folder/'report.json').write_text(json.dumps(stage_report,indent=2)+'\n')
                        probe_stages.append(stage_report)
                        if not fixed: raise RuntimeError('Diagnostic render updates changed physical state')
                    report={'type':'diagnostic_capture_ack','ok':True,'physical_before':physical_before,
                        'settings_before':original_settings,'authored_light_inventory':diagnostic_original_lights,'stages':probe_stages,'output':str(probe_dir)}
                    (probe_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
                    response_queue.put(report)
                except Exception as exc:
                    response_queue.put({'type':'diagnostic_capture_ack','ok':False,'error':repr(exc)})
                continue
'''


def build(output):
    simulator=ROOT/'ExcavatorVLA-smolvla/run_simulation.py';evaluator=ROOT/'scripts/eval_smolvla_success.py'
    sim=simulator.read_text();ev=evaluator.read_text()
    collection=ROOT/'ExcavatorVLA-smolvla/run_vla_train_scene.py';source=collection.read_text();tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='apply_stable_dataset_render_settings')
    function=ast.get_source_segment(source,node)
    anchor='def main(args):\n'
    if sim.count(anchor)!=1:raise ValueError('Simulator entry anchor changed')
    sim=sim.replace(anchor,function+'\n\n'+anchor,1)
    sim=sim.replace('SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))','SCRIPT_DIR = '+repr(str(simulator.parent)),1)
    anchor="            if cmd.get('type') == 'set_task_goals':\n"
    if sim.count(anchor)!=1:raise ValueError('Probe command anchor changed')
    sim=sim.replace(anchor,PROBE_COMMAND+'\n'+anchor,1)
    startup='    print(\n        "[INFO] Simulation running in bridge-lockstep mode: "'
    if sim.count(startup)!=1:raise ValueError('Bridge readiness anchor changed')
    lighting="""    diagnostic_original_lights = []
    for light_prim in stage.Traverse():
        if 'light' in light_prim.GetTypeName().lower():
            diagnostic_original_lights.append({'path':str(light_prim.GetPath()),'type':light_prim.GetTypeName(),
                'attributes':{attribute.GetName():str(attribute.Get()) for attribute in light_prim.GetAttributes()
                    if any(key in attribute.GetName().lower() for key in ['texture','intensity','exposure','visibility'])}})
    control_dome = stage.GetPrimAtPath('/World/ControlRig/DomeLight')
    if not control_dome.IsValid():raise RuntimeError('Authored control dome missing')
    stage.SetEditTarget(stage.GetSessionLayer())
    UsdGeom.Imageable(control_dome).MakeInvisible()

"""
    sim=sim.replace(startup,lighting+startup,1)
    sim_path=output/'simulator_probe.py';sim_path.write_text(sim)
    ev=ev.replace('BASE = Path(__file__).resolve().parents[1]','BASE = Path('+repr(str(ROOT))+')\nsys.path.insert(0,str(BASE/"scripts"))',1)
    ev=ev.replace("str(REPO / 'run_simulation.py')",repr(str(sim_path)))
    anchor="                (folder / 'initial_observation.json').write_text(json.dumps(initial_report, indent=2) + '\\n')\n"
    if ev.count(anchor)!=1:raise ValueError('Initial observation anchor changed')
    injection='''                from PIL import Image as DiagnosticImage
                for name, image in frames.items():
                    DiagnosticImage.fromarray(image).save(folder / ('model_input_' + name + '.png'))
                sock.settimeout(600)
                probe = exchange(sock, {'type':'diagnostic_capture','output':str(folder/'frozen_capture')})
                atomic_json(folder/'camera_diagnostic.json',probe)
                if not probe.get('ok'): raise RuntimeError('CameraDiagnosticError: '+str(probe))
                video = None
                status.update(status='camera_diagnostic_only',success=False)
                return
'''
    ev=ev.replace(anchor,anchor+injection,1)
    ev_path=output/'evaluator_probe.py';ev_path.write_text(ev)
    for p in [sim_path,ev_path]:compile(p.read_text(),str(p),'exec')
    atomic_json(output/'source_manifest.json',dict(original={str(p.relative_to(ROOT)):digest(p) for p in [simulator,evaluator,collection]},
        diagnostic={p.name:digest(p) for p in [sim_path,ev_path]},shared_source_files_modified=False,
        stages=['single_hdr','authored_restored','single_hdr_again'],light_intervention='Only isolated diagnostic copy hides /World/ControlRig/DomeLight; restores authored light during middle phase',captures_per_stage=24,physics_paused=True))
    return ev_path


def main(args):
    args.output.mkdir(parents=True,exist_ok=False);probe=build(args.output)
    atomic_json(args.output/'status.json',dict(status='running_camera_probes',runs=4))
    for scene in [50,112]:
        historical=BASELINE/f'scene_{scene:04d}'
        gpu=read(historical/'evaluation/episodes/episode_0001/episode_status.json')['gpu_id']
        for repeat in [1,2]:
            folder=args.output/f'scene_{scene:04d}_repeat_{repeat:02d}';folder.mkdir()
            command=read(historical/'launch.json')['command'];command[1]=str(probe) if command[1]!='-B' else command[1]
            # Retain every original protocol/scene argument; only entrypoint and output change.
            old_entry=str(ROOT/'scripts/eval_smolvla_success.py')
            if command.count(old_entry)!=1:raise ValueError('Evaluator command changed')
            command[command.index(old_entry)]=str(probe)
            for key,value in {'--output':str(folder/'evaluation'),'--steps':'1','--gpus':gpu}.items():command[command.index(key)+1]=value
            env=evaluation_environment(folder);env['CUDA_VISIBLE_DEVICES']=gpu
            atomic_json(folder/'launch.json',dict(command=command,started_utc=now()))
            with (folder/'launch.log').open('w') as log:
                proc=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                atomic_json(folder/'process.json',dict(pid=proc.pid));code=proc.wait()
            atomic_json(folder/'launch_result.json',dict(exit_code=code,finished_utc=now()))
            path=folder/'evaluation/episodes/episode_0001/camera_diagnostic.json'
            # Native task scorer exits 1 for intentional diagnostic-only termination.
            if code not in (0,1) or not path.exists() or not read(path).get('ok'):
                atomic_json(args.output/'status.json',dict(status='probe_failed',scene=scene,repeat=repeat,exit_code=code));return
            atomic_json(args.output/'status.json',dict(status='running_camera_probes',last_completed_scene=scene,last_completed_repeat=repeat))
    for path,sha in read(args.output/'source_manifest.json')['original'].items():
        if digest(ROOT/path)!=sha:raise ValueError('Shared source file changed')
    atomic_json(args.output/'status.json',dict(status='completed',runs=4,shared_sources_unchanged=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=lambda p:Path(p).resolve(),required=True);main(parser.parse_args())
