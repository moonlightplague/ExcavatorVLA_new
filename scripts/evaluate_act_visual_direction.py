"""Frozen three-arm comparison under one versioned HDR evaluation profile."""
import argparse,json,os,shutil,subprocess,sys,time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(os.environ.get('ACT_PROJECT_ROOT',Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(ROOT/'scripts'))
from eval_smolvla_repeat40 import atomic_json,read,now,digest,evaluation_environment
from analyze_act_visual_direction import analyze,means
SOURCE=ROOT/'outputs/experiments/act_visual_robustness_20261120_01'
HISTORY=ROOT/'outputs/experiments/act_velocity_holdout8_20261119_01'
HDR=ROOT/'outputs/experiments/act_hdr_baseline_repeatability_20261120_02'
GPUS=['GPU-36fd8a97-912c-376e-4db4-a99c0c782d59','GPU-337b7e31-96ff-1982-3593-c5e72801d4fd']
CHECKPOINTS=dict(baseline=ROOT/'outputs/experiments/act_training_alternatives_20261119_01/velocity/train/checkpoints/008000/pretrained_model',
    control=Path(read(SOURCE/'control/status.json')['checkpoint']),robust=Path(read(SOURCE/'robust/status.json')['checkpoint']))
ARMS=['baseline','control','robust']
RUNTIME=ROOT/'outputs/experiments/act_training_alternatives_runtime_20261119_01'


def command(job,out):
    cmd=list(read(HISTORY/f"scene_{job['scene']:04d}/launch.json")['command'])
    entry=str(ROOT/'scripts/eval_smolvla_success.py')
    if cmd.count(entry)!=1:raise ValueError('Evaluator anchor changed')
    cmd[cmd.index(entry)]=str(HDR/'evaluator_single_hdr.py')
    for key,value in {'--checkpoint':str(CHECKPOINTS[job['arm']]),'--output':str(out/job['name']/'evaluation'),
        '--steps':str(job['steps']),'--gpus':job['gpu_uuid']}.items():
        if cmd.count(key)!=1:raise ValueError('Ambiguous argument '+key)
        cmd[cmd.index(key)+1]=value
    return cmd


def prepare(output):
    protected=read(ROOT/'outputs/experiments/act_training_alternatives_20261119_01/external_files.json')
    protected['scripts/analyze_act_visual_direction.py']=digest(ROOT/'scripts/analyze_act_visual_direction.py')
    for name,sha in read(HDR/'source_manifest.json')['isolated'].items():protected[str((HDR/name).relative_to(ROOT))]=sha
    for checkpoint in CHECKPOINTS.values():
        for path in checkpoint.iterdir():
            if path.is_file():protected[str(path.relative_to(ROOT))]=digest(path)
    split=read(CHECKPOINTS['baseline']/'benchmark_provenance.json')['split_manifest']
    jobs=[]
    # Identical GPU, scene, seed and budget for every arm within a scene.
    # Counterbalance execution order to reduce monotonic launch-time confounding.
    for i,scene in enumerate([50,112,11,80,141,152]):
        steps=5000 if scene in [50,112] else 2000;gpu=1 if scene in [50,112] else 0
        if scene not in split['test'] or scene in split['train']+split['validation']:raise ValueError('Parent split overlap')
        arms=ARMS[i%3:]+ARMS[:i%3]
        old=read(HISTORY/f'scene_{scene:04d}/launch.json')['command'];ref=Path(old[old.index('--reference-episode')+1])
        protected[str(ref.relative_to(ROOT))]=digest(ref)
        for arm in arms:
            jobs.append(dict(name=f'scene_{scene:04d}_{arm}',scene=scene,arm=arm,steps=steps,gpu_uuid=GPUS[gpu],gpu_lane=gpu))
    for path,sha in protected.items():
        if digest(ROOT/path)!=sha:raise ValueError('Protected source changed '+path)
    commands=[command(j,output) for j in jobs]
    assert len(jobs)==18 and len({j['name'] for j in jobs})==18
    assert all(c[c.index('--checkpoint')+1]==str(CHECKPOINTS[j['arm']]) for j,c in zip(jobs,commands,strict=True))
    return dict(created_utc=now(),jobs=jobs,checkpoints={k:str(v) for k,v in CHECKPOINTS.items()},protected_sha256=protected,
        lighting_profile='single_authored_hdr_v1',private_evaluator=str(HDR/'evaluator_single_hdr.py'),
        data_usage='Frozen original test parents used as development evaluations; no rollout data used for fitting or checkpoint selection',
        scopes=dict(known_scenes=[50,112],additional_scenes=[11,80,141,152],known_budget=5000,additional_budget=2000),
        interpretation='One fresh run per arm/scene; earlier two HDR baseline/robust repeats remain separate. No statistical significance claim or automatic promotion.',
        model_control='Original action and success logic; no added controller, phase input, auxiliary intervention or action guard',
        continuation_criterion='Continue optimization only after reviewing reproducibility, additional-scene breadth and return/ten-cycle regressions. Three-arm comparison distinguishes ordinary fine-tuning from augmentation-specific benefit.',
        next_training_automatic=False,launch_checks='18 unique jobs; strict protected hashes, disjoint original split, identical scene seed/GPU/budget per arm and explicit checkpoint command verified')


def assert_protected(manifest):
    for path,sha in manifest['protected_sha256'].items():
        if digest(ROOT/path)!=sha:raise ValueError('Protected file changed '+path)


def row_result(folder):return folder/'evaluation/episodes/episode_0001/episode_status.json'


def worker(output,manifest,lane):
    for job in [j for j in manifest['jobs'] if j['gpu_lane']==lane]:
        folder=output/job['name'];folder.mkdir()
        try:
            assert_protected(manifest);cmd=command(job,output)
            env=evaluation_environment(folder);env['CUDA_VISIBLE_DEVICES']=job['gpu_uuid']
            env['PYTHONPATH']=str(RUNTIME)+':'+str(ROOT/'ExcavatorVLA-vlm')
            cudnn=Path(sys.executable).parent.parent/'lib/python3.12/site-packages/nvidia/cudnn/lib'
            env['LD_LIBRARY_PATH']=str(cudnn)+':'+os.environ.get('LD_LIBRARY_PATH','')
            atomic_json(folder/'launch.json',dict(command=cmd,started_utc=now(),job=job))
            with (folder/'launch.log').open('w') as log:
                process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                atomic_json(folder/'process.json',dict(pid=process.pid));code=process.wait()
            atomic_json(folder/'launch_result.json',dict(exit_code=code,finished_utc=now()))
            if code:raise RuntimeError('Evaluator exited '+str(code))
            row=read(row_result(folder))
            if row.get('status')!='completed' or row.get('error'):raise RuntimeError(str(row.get('error','Incomplete rollout')))
            if row.get('gpu_id')!=job['gpu_uuid']:raise ValueError('Unexpected GPU')
            analysis=analyze(folder);audit=analysis['model_action_audit']
            if audit['nonmodel_steps'] or audit['guarded_steps'] or audit['predicted_applied_differences']:raise ValueError('Model action path changed')
            atomic_json(folder/'phase_analysis.json',analysis)
            assert_protected(manifest)
        except Exception as error:
            atomic_json(folder/'infrastructure_error.json',dict(error=repr(error),finished_utc=now()))


def summary(output,manifest):
    done=[];active=[];errors=[]
    for job in manifest['jobs']:
        folder=output/job['name'];path=row_result(folder)
        if (folder/'infrastructure_error.json').exists():errors.append(dict(job=job,**read(folder/'infrastructure_error.json')))
        if path.exists() and read(path).get('status')=='completed':
            row=read(path);cycles=[c for c in row['repeat_cycles'] if c.get('returned') and isinstance(c.get('end_step'),int)]
            prefix=[c for c in cycles if c['end_step']<2000]
            done.append(dict(**job,completed_cycles=len(cycles),successful_cycles=sum(bool(c.get('success')) for c in cycles),
                prefix_completed_cycles=len(prefix),prefix_successful_cycles=sum(bool(c.get('success')) for c in prefix),
                ten_cycle_task_success=bool(row.get('repeat_collection_success')),executed_steps=row.get('policy_steps_executed'),error=row.get('error')))
        elif (folder/'evaluation/workers/worker_00/progress.json').exists():active.append(dict(job=job,**read(folder/'evaluation/workers/worker_00/progress.json')))
    pairs={}
    for scene in [50,112,11,80,141,152]:
        rows={r['arm']:r for r in done if r['scene']==scene}
        if len(rows)!=3:continue
        pairs[str(scene)]={arm:dict(prefix_successful_cycle_change=rows[arm]['prefix_successful_cycles']-rows['baseline']['prefix_successful_cycles'],
            completed_cycle_change=rows[arm]['completed_cycles']-rows['baseline']['completed_cycles'],successful_cycle_change=rows[arm]['successful_cycles']-rows['baseline']['successful_cycles'],
            ten_cycle_task_success=rows[arm]['ten_cycle_task_success']) for arm in ['control','robust']}
    report=dict(updated_utc=now(),requested_runs=len(manifest['jobs']),completed_runs=len(done),results=done,paired_changes=pairs,active=active,infrastructure_errors=errors,
        scope='Frozen three-arm development comparison. One fresh run per arm/scene; prefix results and full-budget results separate; additional short runs do not measure complete ten-cycle success rate.')
    atomic_json(output/'summary.json',report);return report


def main(args):
    if args.finalize_existing:
        manifest=read(args.output/'manifest.json');assert_protected(manifest)
        for job in manifest['jobs']:
            folder=args.output/job['name'];row=read(row_result(folder))
            if read(folder/'launch_result.json')['exit_code']!=0 or row.get('status')!='completed' or row.get('error'):raise ValueError('Invalid completed job '+job['name'])
            if row.get('gpu_id')!=job['gpu_uuid']:raise ValueError('Unexpected GPU')
            audit=read(folder/'phase_analysis.json')['model_action_audit']
            if audit['nonmodel_steps'] or audit['guarded_steps'] or audit['predicted_applied_differences']:raise ValueError('Action audit failed')
        report=summary(args.output,manifest)
        if report['completed_runs']!=18 or report['infrastructure_errors']:raise ValueError('Cannot finalize incomplete jobs')
        atomic_json(args.output/'summary_recovery.json',dict(recovered_utc=now(),reason='Summary dict duplicated job steps key; retained step budget and renamed measured count executed_steps',all_original_rollouts_preserved=True,protected_hashes_and_18_launch_results_and_action_audits_verified=True))
        atomic_json(args.output/'status.json',dict(status='completed',completed_runs=18,finished_utc=now(),baseline_replaced=False,next_training_automatic=False))
        atomic_json(args.output/'completion.json',dict(status='completed',finished_utc=now()))
        print(json.dumps(dict(status='completed',completed_runs=18,paired_changes=report['paired_changes'])));return
    manifest=prepare(args.output)
    if args.check_only:
        print(json.dumps(dict(status='passed',jobs=len(manifest['jobs']),scenes=[50,112,11,80,141,152],scope=manifest['scopes'])));return
    args.output.mkdir(parents=True,exist_ok=False);shutil.copy2(Path(__file__),args.output/'run.py')
    shutil.copy2(ROOT/'scripts/analyze_act_visual_direction.py',args.output/'analyze_act_visual_direction.py')
    atomic_json(args.output/'manifest.json',manifest);atomic_json(args.output/'status.json',dict(status='evaluating',requested_runs=18))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,args.output,manifest,lane) for lane in [0,1]]
        while not all(f.done() for f in futures):
            report=summary(args.output,manifest);print(now(),report['completed_runs'],'/18 complete',flush=True);time.sleep(10)
        for f in futures:f.result()
    report=summary(args.output,manifest)
    status='completed' if len(report['results'])==18 and not report['infrastructure_errors'] else 'completed_with_infrastructure_errors'
    atomic_json(args.output/'status.json',dict(status=status,completed_runs=report['completed_runs'],finished_utc=now(),baseline_replaced=False,next_training_automatic=False))
    atomic_json(args.output/'completion.json',dict(status=status,finished_utc=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=lambda p:Path(p).resolve(),required=True);p.add_argument('--check-only',action='store_true');p.add_argument('--finalize-existing',action='store_true');main(p.parse_args())
