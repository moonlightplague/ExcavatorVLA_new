"""Gate a versioned, isolated HDR evaluation profile on frozen-camera evidence."""
import argparse,json,os,shutil,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUNTIME=ROOT/'outputs/experiments/act_training_alternatives_runtime_20261119_01/python'
PROBE=ROOT/'outputs/experiments/act_camera_diagnostic_20261120_03'


def read(p):return json.loads(p.read_text())
def write(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)


def build(out):
    import hashlib
    simulator=ROOT/'ExcavatorVLA-smolvla/run_simulation.py';evaluator=ROOT/'scripts/eval_smolvla_success.py'
    sim=simulator.read_text();ev=evaluator.read_text()
    sim=sim.replace('SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))','SCRIPT_DIR = '+repr(str(simulator.parent)),1)
    anchor='    print(\n        "[INFO] Simulation running in bridge-lockstep mode: "'
    if sim.count(anchor)!=1:raise ValueError('Bridge readiness anchor changed')
    lights='''    # Versioned visual input profile only: leave physics and task actions unchanged.
    control_dome = stage.GetPrimAtPath('/World/ControlRig/DomeLight')
    hdr_dome = stage.GetPrimAtPath('/Environment/DomeLight')
    if not control_dome.IsValid() or not hdr_dome.IsValid():
        raise RuntimeError('Expected authored lighting prims are missing')
    hdr_asset = hdr_dome.GetAttribute('inputs:texture:file').Get()
    if not hdr_asset or not getattr(hdr_asset, 'path', ''):
        raise RuntimeError('Expected recorded-scene HDR texture is missing')
    stage.SetEditTarget(stage.GetSessionLayer())
    UsdGeom.Imageable(control_dome).MakeInvisible()
    print('[EVALUATION LIGHTING] single_authored_hdr_v1; control dome hidden in session layer; HDR and WorkLight preserved',flush=True)

'''
    sim=sim.replace(anchor,lights+anchor,1)
    sim_path=out/'simulator_single_hdr.py';sim_path.write_text(sim)
    ev=ev.replace('BASE = Path(__file__).resolve().parents[1]','BASE = Path('+repr(str(ROOT))+')\nsys.path.insert(0,str(BASE/"scripts"))',1)
    ev=ev.replace("str(REPO / 'run_simulation.py')",repr(str(sim_path)))
    anchor='    if args.act_carry_guard:\n'
    if ev.count(anchor)!=1:raise ValueError('Evaluation profile anchor changed')
    ev=ev.replace(anchor,"    config['render_lighting_profile'] = 'single_authored_hdr_v1'\n"+anchor,1)
    anchor="                (folder / 'initial_observation.json').write_text(json.dumps(initial_report, indent=2) + '\\n')\n"
    if ev.count(anchor)!=1:raise ValueError('Initial observation anchor changed')
    ev=ev.replace(anchor,anchor+'''                from PIL import Image as InitialDiagnosticImage
                for camera_name, image in frames.items():
                    InitialDiagnosticImage.fromarray(image).save(folder / ('model_input_' + camera_name + '.png'))
''',1)
    protocol_anchor="        protocol['expert_prefix_steps'] = args.expert_prefix_steps\n"
    if ev.count(protocol_anchor)!=1:raise ValueError('Protocol metadata anchor changed')
    ev=ev.replace(protocol_anchor,"        protocol['render_lighting_profile'] = 'single_authored_hdr_v1'\n        protocol['lighting_runtime_sha256'] = {'evaluator':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'simulator':hashlib.sha256(Path("+repr(str(sim_path))+").read_bytes()).hexdigest()}\n"+protocol_anchor,1)
    ev_path=out/'evaluator_single_hdr.py';ev_path.write_text(ev)
    for p in [sim_path,ev_path]:compile(p.read_text(),str(p),'exec')
    write(out/'source_manifest.json',dict(profile='single_authored_hdr_v1',shared_sources_modified=False,
        original={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [simulator,evaluator]},
        isolated={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [sim_path,ev_path]},
        intervention='Session-layer visibility override for untextured ControlRig dome only; WorkLight and recorded HDR retained; success and action logic unchanged'))
    return ev_path


def main(args):
    args.output.mkdir(parents=True,exist_ok=False);shutil.copy2(Path(__file__),args.output/'run.py')
    write(args.output/'status.json',dict(status='waiting_for_camera_validation',source=str(PROBE)))
    while not (PROBE/'analysis_completion.json').exists():
        if read(PROBE/'status.json')['status']=='probe_failed':raise RuntimeError('Camera validation failed')
        time.sleep(10)
    data=read(PROBE/'analysis.json');checks=[]
    for scene in ['50','112']:
        cross=data['cross_launch'][scene]
        good=cross['initial_state_max_difference']==0 and cross.get('particle_max_abs_difference')==0 and cross['camera_transform_max_abs_difference']==0
        max_error=max(v['last_frame_mae'] for v in cross['pixels']['single_hdr'].values());good &= max_error<2.
        checks.append(dict(scene=scene,physical_and_camera_match=bool(good),single_hdr_cross_run_pixel_mae_max=max_error))
    # Verify the actual saved initial model inputs have the intended recorded-sky domain.
    from PIL import Image
    import numpy as np
    blue=[]
    for scene in [50,112]:
        for repeat in [1,2]:
            image=np.asarray(Image.open(PROBE/f'scene_{scene:04d}_repeat_{repeat:02d}/evaluation/episodes/episode_0001/model_input_0.png'))
            color=image[:45].mean((0,1));blue.append(float(color[2]-color[0]))
    passed=all(c['physical_and_camera_match'] for c in checks) and min(blue)>20
    write(args.output/'camera_gate.json',dict(passed=bool(passed),checks=checks,initial_sky_blue_minus_red=blue,
        scope='Allows experimental baseline remeasurement under fixed HDR. Does not certify full rendering or task determinism.'))
    if not passed:
        write(args.output/'status.json',dict(status='camera_gate_rejected',training=False));return
    private=build(args.output)
    write(args.output/'status.json',dict(status='baseline_hdr_repeatability_running',training=False,runs=4,steps=2000))
    command=[str(RUNTIME),'-u','-B',str(ROOT/'scripts/evaluate_act_baseline_repeatability.py'),'--output',str(args.output/'repeats'),
             '--evaluator',str(private),'--lighting-profile','single_authored_hdr_v1']
    with (args.output/'repeatability.log').open('w') as log:
        subprocess.run(command,cwd=ROOT,env={**os.environ,'ACT_PROJECT_ROOT':str(ROOT),'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'},stdout=log,stderr=subprocess.STDOUT,check=True)
    write(args.output/'status.json',dict(status='completed',training=False,profile='single_authored_hdr_v1',results=str(args.output/'repeats')))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=lambda p:Path(p).resolve(),required=True);args=parser.parse_args()
    try:main(args)
    except Exception as error:
        if args.output.exists():write(args.output/'error.json',dict(error=repr(error)))
        raise
