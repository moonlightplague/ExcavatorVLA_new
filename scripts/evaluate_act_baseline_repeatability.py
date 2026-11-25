"""Repeat frozen baseline prefixes on the original GPU, scene and policy seed."""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(os.environ.get('ACT_PROJECT_ROOT',Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(ROOT/'scripts'))
from eval_smolvla_repeat40 import atomic_json,read,now,digest,evaluation_environment
ORIGINAL=ROOT/'outputs/experiments/act_velocity_holdout8_20261119_01'
CHECKPOINT=ROOT/'outputs/experiments/act_training_alternatives_20261119_01/velocity/train/checkpoints/008000/pretrained_model'
RUNTIME=ROOT/'outputs/experiments/act_training_alternatives_runtime_20261119_01/python'


def result(folder):return folder/'evaluation/episodes/episode_0001/episode_status.json'


def first_action(folder):
    path=folder/'evaluation/episodes/episode_0001/steps.jsonl'
    if not path.exists():return None
    with path.open() as file:
        line=file.readline()
        if not line:return None
        row=json.loads(line)
        return dict(state=row['input_state'],predicted=row['predicted'],camera_sha256=row.get('pre_action_camera_sha256'))


def prefix_counts(row):
    cycles=[c for c in row.get('repeat_cycles',[]) if isinstance(c.get('end_step'),int) and c['end_step']<2000 and c.get('returned')]
    return dict(completed_cycles=len(cycles),successful_cycles=sum(bool(c.get('success')) for c in cycles))


def comparison(out,jobs,lighting_profile="authored_original"):
    scenes={};active=[];errors=[];finished=0
    for episode in [112,50]:
        baseline_folder=ORIGINAL/f'scene_{episode:04d}';baseline=read(result(baseline_folder));reference=first_action(baseline_folder)
        samples=[dict(source='historical_baseline',lighting_profile='authored_original',**prefix_counts(baseline))]
        for job in jobs:
            if job['scene']!=episode:continue
            folder=Path(job['output']);path=result(folder)
            if path.exists() and read(path).get('status')=='completed':
                row=read(path)
                if row.get('error'):
                    errors.append(dict(scene=episode,repeat=job['repeat'],error=row['error']));continue
                finished+=1;first=first_action(folder)
                entry=dict(source='repeat_'+str(job['repeat']),lighting_profile=lighting_profile,**prefix_counts(row),steps=row.get('policy_steps_executed'),phase=row.get('repeat_phase'),gpu=row.get('gpu_id'))
                if first and reference:
                    entry.update(first_state_max_abs_difference=max(abs(a-b) for a,b in zip(first['state'],reference['state'],strict=True)),
                        first_predicted_action_max_abs_difference=max(abs(a-b) for a,b in zip(first['predicted'],reference['predicted'],strict=True)),
                        first_camera_hashes_equal=first['camera_sha256']==reference['camera_sha256'])
                samples.append(entry)
            else:
                progress=folder/'evaluation/workers/worker_00/progress.json'
                if progress.exists():active.append(dict(scene=episode,repeat=job['repeat'],**read(progress)))
            error=folder/'infrastructure_error.json'
            if error.exists():errors.append(dict(scene=episode,repeat=job['repeat'],**read(error)))
        comparable=[s for s in samples if s['lighting_profile']==lighting_profile]
        scenes[str(episode)]=dict(samples=samples,range_profile=lighting_profile,comparable_runs=len(comparable),
            completed_cycle_range=[min(s['completed_cycles'] for s in comparable),max(s['completed_cycles'] for s in comparable)] if comparable else None,
            successful_cycle_range=[min(s['successful_cycles'] for s in comparable),max(s['successful_cycles'] for s in comparable)] if comparable else None)
    report=dict(updated_utc=now(),requested_runs=4,completed_runs=finished,scenes=scenes,active=active,infrastructure_errors=errors,
        scope='Same-seed same-GPU repeatability of 2000-step prefixes, not ten-cycle success rates; historical runs included as reference only')
    atomic_json(out/'summary.json',report);return report


def work(out,jobs,protected,evaluator=None):
    for job in jobs:
        folder=Path(job['output']);folder.mkdir()
        try:
            for path,sha in protected.items():
                if digest(ROOT/path)!=sha:raise ValueError('Protected source file changed: '+path)
            original=ORIGINAL/f"scene_{job['scene']:04d}";cmd=read(original/'launch.json')['command']
            if evaluator is not None:
                entry=str(ROOT/'scripts/eval_smolvla_success.py')
                if cmd.count(entry)!=1:raise ValueError('Expected original evaluator entrypoint')
                cmd[cmd.index(entry)]=str(evaluator)
            changes={'--output':str(folder/'evaluation'),'--steps':'2000','--gpus':job['gpu_uuid']}
            for key,value in changes.items():
                if cmd.count(key)!=1:raise ValueError('Ambiguous evaluator argument: '+key)
                cmd[cmd.index(key)+1]=value
            env=evaluation_environment(folder);env['CUDA_VISIBLE_DEVICES']=job['gpu_uuid']
            atomic_json(folder/'launch.json',dict(command=cmd,started_utc=now(),gpu_uuid=job['gpu_uuid'],historical_command=str(original/'launch.json')))
            with (folder/'launch.log').open('w') as log:
                process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                atomic_json(folder/'process.json',dict(pid=process.pid));code=process.wait()
            atomic_json(folder/'launch_result.json',dict(exit_code=code,finished_utc=now()))
            if code:raise RuntimeError('Evaluator exit '+str(code))
            row=read(result(folder))
            if row.get('gpu_id')!=job['gpu_uuid']:raise ValueError('Actual GPU differs from historical baseline')
            if row.get('error'):raise RuntimeError(row['error'])
            for path,sha in protected.items():
                if digest(ROOT/path)!=sha:raise ValueError('Protected source file changed during run: '+path)
        except Exception as error:
            atomic_json(folder/'infrastructure_error.json',dict(error=repr(error),finished_utc=now()))


def main(args):
    protected={str(p.relative_to(ROOT)):digest(p) for p in CHECKPOINT.iterdir() if p.is_file()}
    expected=read(ROOT/'outputs/experiments/act_training_alternatives_20261119_01/external_files.json')
    for path,sha in expected.items():
        if digest(ROOT/path)!=sha:raise ValueError('Evaluator/controller changed before repeats')
    protected.update(expected)
    if args.evaluator:
        protected[str(args.evaluator.relative_to(ROOT))]=digest(args.evaluator)
        isolated=args.evaluator.parent/'source_manifest.json'
        if isolated.exists():
            for name,sha in read(isolated).get('isolated',{}).items():
                protected[str((args.evaluator.parent/name).relative_to(ROOT))]=sha
    jobs=[]
    for repeat in [1,2]:
        for episode in [112,50]:
            status=read(result(ORIGINAL/f'scene_{episode:04d}'))
            jobs.append(dict(scene=episode,repeat=repeat,gpu_uuid=status['gpu_id'],output=str(args.output/f'scene_{episode:04d}_repeat_{repeat:02d}')))
    if len({j['gpu_uuid'] for j in jobs})!=1:raise ValueError('Expected both historical scenes on one physical GPU')
    args.output.mkdir(parents=True,exist_ok=False);shutil.copy2(Path(__file__),args.output/'run.py')
    atomic_json(args.output/'manifest.json',dict(created_utc=now(),checkpoint=str(CHECKPOINT),jobs=jobs,protected_sha256=protected,
        steps=2000,repeats_per_scene=2,ordering='112 then 50, twice, sequential on original GPU',
        source_weights_frozen=True,training=False,full_ten_cycle_success_measurement=False,
        private_evaluator=str(args.evaluator) if args.evaluator else None,lighting_profile=args.lighting_profile,
        reference='Historical baseline runs plus two fresh runs per scene; short prefix stability only',
        interpretation='Variation may originate in initial observations, inference or simulator trajectory; do not attribute all changes to weights'))
    atomic_json(args.output/'status.json',dict(status='checking_fixed_input_inference'))
    env={**os.environ,'CUDA_VISIBLE_DEVICES':jobs[0]['gpu_uuid'],'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}
    with (args.output/'inference_check.log').open('w') as log:
        subprocess.run([str(RUNTIME),'-u','-B',str(ROOT/'scripts/check_act_inference_repeatability.py'),'--output',str(args.output/'fixed_input_inference.json')],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    atomic_json(args.output/'status.json',dict(status='evaluating',requested_runs=4,gpu_uuid=jobs[0]['gpu_uuid']))
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(work,args.output,jobs,protected,args.evaluator)
        while not future.done():
            report=comparison(args.output,jobs,args.lighting_profile);print(now(),report['completed_runs'],'/4 complete',flush=True);time.sleep(10)
        future.result()
    report=comparison(args.output,jobs,args.lighting_profile)
    state='completed' if report['completed_runs']==4 and not report['infrastructure_errors'] else 'completed_with_infrastructure_errors'
    atomic_json(args.output/'status.json',dict(status=state,finished_utc=now(),completed_runs=report['completed_runs']))
    atomic_json(args.output/'completion.json',dict(status=state,finished_utc=now()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=lambda p:Path(p).resolve(),required=True)
    parser.add_argument('--evaluator',type=lambda p:Path(p).resolve())
    parser.add_argument('--lighting-profile',default='authored_original')
    main(parser.parse_args())
