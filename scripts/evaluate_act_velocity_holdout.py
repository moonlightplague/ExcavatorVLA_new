"""Frozen-weight, fixed-sample ACT validation; ordinary evaluator is unchanged."""
import argparse,json,os,random,shutil,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(os.environ.get('ACT_PROJECT_ROOT',Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(ROOT/'scripts'))
from eval_smolvla_repeat40 import atomic_json,read,now,digest,evaluation_environment
SOURCE=ROOT/'outputs/experiments/act_training_alternatives_20261119_01'
CHECKPOINT=SOURCE/'velocity/train/checkpoints/008000/pretrained_model'
DATA=ROOT/'outputs/experiments/act_repeat_control_repair_data_20261118_01/dataset'


def prepare(out):
    previous=read(ROOT/'outputs/experiments/act_original_40k_closedloop_test40_20261117_01/manifest.json')
    provenance=read(CHECKPOINT/'benchmark_provenance.json');split=provenance['split_manifest']
    eligible=sorted(s['source_export_episode'] for s in previous['scenes'] if s['source_export_episode'] not in (16,35))
    ids=sorted(random.Random(20261119).sample(eligible,8));lookup={s['source_export_episode']:s for s in previous['scenes']}
    if not set(ids).issubset(split['test']) or set(ids)&set(split['train']+split['validation']):raise ValueError('Selection overlaps training/validation')
    scenes=[lookup[i] for i in ids]
    for scene in scenes:
        if digest(scene['reference'])!=scene['reference_sha256']:raise ValueError('Scene reference changed')
    protected={str(p.relative_to(ROOT)):digest(p) for p in sorted(CHECKPOINT.iterdir()) if p.is_file()}
    protected.update(read(SOURCE/'external_files.json'))
    for path,sha in read(SOURCE/'external_files.json').items():
        if digest(ROOT/path)!=sha:raise ValueError('External evaluator/controller changed')
    manifest=dict(created_utc=now(),checkpoint=str(CHECKPOINT),dataset=str(DATA),scenes=scenes,
        episodes=8,repeat_cycles=10,steps=5000,replan_interval=3,chunk_size=10,policy_type='act',
        selection=dict(method='uniform random sample without replacement from 38 eligible held-out parents; sorted execution order',seed=20261119,excluded_prior_tuning_scenes=[16,35],selected=ids),
        dataset_fingerprint=provenance['dataset_fingerprint'],protected_sha256=protected,
        success='Original ten-cycle criterion: each cycle delivers >=50kg and physically returns; final >=400kg with >=80% mass retention; strict 100-particle metric separate',
        scope='Further fixed-weight validation on eight additional held-out parents, not a forty-scene population estimate; no checkpoint selection or training during this run',
        stop_after_first_success=False,video=True,soil_backend='newton_mpm')
    out.mkdir(parents=True,exist_ok=False);atomic_json(out/'manifest.json',manifest)
    shutil.copy2(Path(__file__),out/'run.py')
    return manifest


def summarize(out,manifest):
    rows=[];active=[];errors=[]
    for scene in manifest['scenes']:
        episode=scene['source_export_episode'];folder=out/f'scene_{episode:04d}'
        status=folder/'evaluation/episodes/episode_0001/episode_status.json'
        if status.exists():rows.append(dict(source_export_episode=episode,**read(status)))
        progress=folder/'evaluation/workers/worker_00/progress.json'
        if progress.exists() and not status.exists():active.append(dict(source_export_episode=episode,**read(progress)))
        error=folder/'infrastructure_error.json'
        if error.exists():errors.append(dict(source_export_episode=episode,**read(error)))
    done=[r for r in rows if r.get('status')=='completed' and not r.get('error')]
    failures={}
    for row in done:
        reason=row.get('repeat_failure_mode','none')
        if not row.get('repeat_collection_success'):failures[reason]=failures.get(reason,0)+1
    summary=dict(updated_utc=now(),requested_episodes=8,completed_episodes=len(done),
        successful_episodes=sum(bool(r.get('repeat_collection_success')) for r in done),
        physical_cycles_completed=sum(r.get('repeat_cycles_completed',0) for r in done),
        physical_returns_completed=sum(r.get('repeat_returns_completed',0) for r in done),
        successful_cycles=sum(r.get('repeat_cycles_successful',0) for r in done),
        failed_task_reasons=failures,active=active,infrastructure_errors=errors,episodes=rows)
    atomic_json(out/'summary.json',summary);return summary


def worker(out,manifest,gpu,scenes):
    for scene in scenes:
        folder=out/f"scene_{scene['source_export_episode']:04d}";folder.mkdir()
        try:
            for path,sha in manifest['protected_sha256'].items():
                if digest(ROOT/path)!=sha:raise ValueError('Protected model/evaluation file changed: '+path)
            if digest(scene['reference'])!=scene['reference_sha256']:raise ValueError('Reference changed')
            cmd=[sys.executable,'-B',str(ROOT/'scripts/eval_smolvla_success.py'),'--policy-type','act',
                '--checkpoint',str(CHECKPOINT),'--dataset',str(DATA),'--output',str(folder/'evaluation'),
                '--reference-episode',scene['reference'],'--episodes','1','--repeat-cycles','10','--repeat-cycle-steps','0',
                '--min-particles','100','--steps','5000','--replan-interval','3','--seed',str(scene['policy_seed']),
                '--gpus',str(gpu),'--workers','1','--soil-backend','newton_mpm','--cpu-threads','12',
                '--video','--headless','--fresh-simulator','--render-every-physics-step']
            env=evaluation_environment(folder);env['CUDA_VISIBLE_DEVICES']='0,1'
            atomic_json(folder/'launch.json',dict(started_utc=now(),command=cmd))
            with (folder/'launch.log').open('w') as log:
                process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                atomic_json(folder/'process.json',dict(pid=process.pid,gpu=gpu))
                code=process.wait()
            atomic_json(folder/'launch_result.json',dict(exit_code=code,finished_utc=now()))
            if code:raise RuntimeError('Evaluator exited with code '+str(code))
            row=read(folder/'evaluation/episodes/episode_0001/episode_status.json')
            if row.get('error'):raise RuntimeError(row['error'])
            total=modified=nonmodel=guard=0
            with (folder/'evaluation/episodes/episode_0001/steps.jsonl').open() as trace:
                for line in trace:
                    r=json.loads(line);total+=1;modified+=r['predicted']!=r['applied'];nonmodel+=r['action_source']!='model';guard+=r.get('carry_guard') is not None
            atomic_json(folder/'model_action_audit.json',dict(steps=total,modified_steps=modified,nonmodel_steps=nonmodel,guard_steps=guard))
            if nonmodel or guard:raise ValueError('Nonmodel actions detected')
            for path,sha in manifest['protected_sha256'].items():
                if digest(ROOT/path)!=sha:raise ValueError('Protected file changed during evaluation: '+path)
        except Exception as error:
            atomic_json(folder/'infrastructure_error.json',dict(error=repr(error),finished_utc=now()))


def main(args):
    manifest=prepare(args.output)
    atomic_json(args.output/'status.json',dict(status='evaluating',scenes=manifest['selection']['selected'],requested_episodes=8))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,args.output,manifest,gpu,manifest['scenes'][gpu::2]) for gpu in range(2)]
        while not all(f.done() for f in futures):
            result=summarize(args.output,manifest);print(now(),result['completed_episodes'],'/8 completed',flush=True);time.sleep(10)
        for future in futures:future.result()
    result=summarize(args.output,manifest)
    state='completed' if result['completed_episodes']==8 and not result['infrastructure_errors'] else 'completed_with_infrastructure_errors'
    atomic_json(args.output/'status.json',dict(status=state,finished_utc=now(),completed_episodes=result['completed_episodes'],successful_episodes=result['successful_episodes']))
    atomic_json(args.output/'completion.json',dict(status=state,finished_utc=now()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=lambda s:Path(s).resolve(),required=True)
    main(parser.parse_args())
