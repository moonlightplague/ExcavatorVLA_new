"""Resume missing alternative evaluations, retaining all previous scene results."""
import json,os,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from eval_smolvla_repeat40 import evaluation_environment,atomic_json,now,digest
BASE=ROOT/'outputs/experiments/act_training_alternatives_20261119_01'
OUT=BASE/'recovery_20261119_01'


def read(path):return json.loads(Path(path).read_text())


def check_external():
    for name,sha in read(BASE/'external_files.json').items():
        if digest(ROOT/name)!=sha:raise ValueError('External evaluation file changed: '+name)


def result_path(folder):return folder/'evaluation/episodes/episode_0001/episode_status.json'


def worker(gpu,jobs):
    for kind,scene,folder in jobs:
        check_external();folder.mkdir(parents=True,exist_ok=False)
        checkpoint=read(BASE/kind/'status.json')['checkpoint']
        cmd=read(BASE/f'velocity/evaluation_pair/scene_{scene:04d}/launch.json')['command']
        for key,value in {'--checkpoint':checkpoint,'--output':str(folder/'evaluation'),'--gpus':str(gpu)}.items():
            if cmd.count(key)!=1:raise ValueError('Launch argument changed: '+key)
            cmd[cmd.index(key)+1]=value
        env=evaluation_environment(folder);env['CUDA_VISIBLE_DEVICES']='0,1'
        atomic_json(folder/'launch.json',dict(command=cmd,started_utc=now(),retry_reason='Original orchestrator masked GPU 1' if kind=='velocity' else None))
        with (folder/'launch.log').open('w') as log:
            process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            code=process.wait()
        atomic_json(folder/'launch_result.json',dict(exit_code=code,finished_utc=now()))
        if code:raise RuntimeError('Evaluation infrastructure failed: '+str(folder))
        check_external()


def summarize(jobs):
    variants={}
    for kind in ('velocity','pose'):
        rows=[];active=[]
        for k,scene,folder in jobs:
            if k!=kind:continue
            if result_path(folder).exists():rows.append(dict(scene=scene,**read(result_path(folder))))
            progress=folder/'evaluation/workers/worker_00/progress.json'
            if progress.exists() and not result_path(folder).exists():active.append(dict(scene=scene,**read(progress)))
        done=[r for r in rows if r.get('status')=='completed']
        variants[kind]=dict(completed=len(done),requested=2,successful_episodes=sum(bool(r.get('repeat_collection_success')) for r in done),
            cycles_completed=sum(r.get('repeat_cycles_completed',0) for r in done),cycles_successful=sum(r.get('repeat_cycles_successful',0) for r in done),
            episodes=rows,active=active)
    atomic_json(OUT/'summary.json',dict(updated_utc=now(),variants=variants))
    return variants


def main():
    check_external();OUT.mkdir(parents=True,exist_ok=False)
    old=BASE/'velocity/evaluation_pair/scene_0016'
    if read(result_path(old)).get('status')!='completed':raise ValueError('Completed scene 16 required')
    retry=BASE/'velocity/evaluation_scene35_retry_01'
    pose16=BASE/'pose/evaluation_pair/scene_0016';pose35=BASE/'pose/evaluation_pair/scene_0035'
    all_jobs=[('velocity',16,old),('velocity',35,retry),('pose',16,pose16),('pose',35,pose35)]
    atomic_json(OUT/'manifest.json',dict(created_utc=now(),jobs=[dict(variant=k,scene=s,output=str(f)) for k,s,f in all_jobs],
        source_files_unchanged=True,success_criteria_unchanged=True,completed_scene16_preserved=True))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,0,[('pose',16,pose16)]),pool.submit(worker,1,[('velocity',35,retry),('pose',35,pose35)])]
        while not all(f.done() for f in futures):
            summarize(all_jobs)
            atomic_json(BASE/'status.json',dict(status='evaluating',recovery=str(OUT),updated_utc=now()))
            time.sleep(10)
        errors=[repr(f.exception()) for f in futures if f.exception()]
    summary=summarize(all_jobs)
    atomic_json(OUT/'completion.json',dict(status='failed' if errors else 'completed',errors=errors,finished_utc=now()))
    atomic_json(BASE/'status.json',dict(status='evaluation_infrastructure_failed' if errors else 'completed',recovery=str(OUT),updated_utc=now()))


if __name__=='__main__':main()
