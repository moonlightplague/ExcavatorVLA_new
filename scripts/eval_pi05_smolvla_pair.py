"""Run two-model videos on a precommitted random sample of the frozen test split."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'scripts'))
from success_eval_runtime import atomic_json
from eval_smolvla_repeat40 import evaluation_environment


def read(path):return json.loads(Path(path).read_text())
def now():return datetime.now(timezone.utc).isoformat()


def link_video(out, name, source):
    dest=out/'videos'/name
    if dest.is_symlink():
        if dest.resolve()==source.resolve():return
        dest.unlink()
    if dest.exists():raise ValueError(f'Unexpected video alias: {dest}')
    dest.symlink_to(os.path.relpath(source,dest.parent))


def summarize(out, manifest):
    with (out/'.summary.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        rows=[]
        for policy in ('pi05','smolvla'):
            for eid in manifest['selected_source_episodes']:
                parent=out/'jobs'/policy/f'scene_{eid:04d}'
                attempts=sorted(parent.glob('attempt_*'))
                row=dict(policy=policy,source_export_episode=eid,status='pending',steps=0)
                if attempts:
                    attempt=attempts[-1]; evaluation=attempt/'evaluation'
                    folder=evaluation/'episodes/episode_0001'
                    status=folder/'episode_status.json';progress=evaluation/'workers/worker_00/progress.json'
                    if progress.exists():
                        live=read(progress);row.update(status=live.get('phase'),steps=live.get('steps',0),
                            physical_cycles=live.get('repeat_cycles_completed'),successful_cycles=live.get('repeat_cycles_successful'))
                    preview=folder/'preview_status.json'
                    if preview.exists() and read(preview).get('status')=='ready':
                        name=f'{policy}_episode_{eid:04d}_preview.mp4'
                        link_video(out,name,folder/'preview.mp4');row['preview']=str(out/'videos'/name)
                    if status.exists():
                        saved=read(status)
                        row.update(status=saved['status'],steps=saved['policy_steps_executed'],
                            task_success=saved.get('repeat_collection_success'),
                            physical_cycles=saved.get('repeat_cycles_completed'),successful_cycles=saved.get('repeat_cycles_successful'),
                            task_payload_mass_kg=saved.get('repeat_final_payload_mass_kg'),
                            live_final_truck_mass_kg=saved.get('live_payload_mass_kg'),
                            task_completion_step=saved.get('repeat_task_completion_step'),
                            policy_latency_ms_mean=saved.get('policy_latency_ms_mean'),
                            clipped_action_fraction=saved.get('clipped_action_fraction'),
                            failure_mode=saved.get('repeat_failure_mode'),error=saved.get('error'),status_file=str(status))
                        if saved['status']=='completed':
                            audit=read(folder/'repeat_audit.json')
                            if saved['policy_steps_executed']!=manifest['steps_per_rollout'] or audit['status']!='passed':
                                raise ValueError('Completed outcome violated the fixed step budget or trace audit')
                            name=f'{policy}_episode_{eid:04d}_full5000.mp4'
                            link_video(out,name,folder/'evaluation.mp4');row['video']=str(out/'videos'/name)
                rows.append(row)
        completed=sum(row['status']=='completed' for row in rows)
        states={p:read(out/'runtime'/f'{p}.json') for p in ('pi05','smolvla') if (out/'runtime'/f'{p}.json').exists()}
        result=dict(status='complete' if completed==4 else 'running',updated_utc=now(),
            requested_rollouts=4,completed_rollouts=completed,selected_source_episodes=manifest['selected_source_episodes'],
            scope=manifest['domain_note'],runtime=states,rollouts=rows)
        atomic_json(out/'summary.json',result)
        lines=['# π0.5 / SmolVLA — two sampled test scenes',
            '',f"Selected source episodes: **{manifest['selected_source_episodes']}**; sampling seed: {manifest['selection_seed']}.",
            f'Completed fixed-budget rollouts: **{completed}/4**. Both models use their final 20k checkpoints.',
            '',manifest['domain_note'],
            '', 'Two random scenes per model are a pilot observation, not the success rate of all 40 test scenes.',
            '', 'Each rollout executes 5,000 model-controlled 10 Hz simulation steps. The first ten task cycles are scored at completion; '
                'if completed early, model actions continue against the final goal to the budget. No expert actions or controller guard.',
            'Preview files show only the first 60 simulation seconds and are not final success results.',
            '', '| Scene | Policy | Status | Actions | Physical / qualified cycles | Ten-cycle success | Preview | Full video |',
            '| --- | --- | --- | --- | --- | --- | --- | --- |']
        for row in rows:
            preview=f"[MP4]({Path(row['preview']).relative_to(out)})" if row.get('preview') else 'pending'
            video=f"[MP4]({Path(row['video']).relative_to(out)})" if row.get('video') else 'pending'
            lines.append(f"| {row['source_export_episode']} | {row['policy']} | {row['status']} | {row['steps']}/5000 | "
                         f"{row.get('physical_cycles',0)} / {row.get('successful_cycles',0)} | "
                         f"{row.get('task_success','pending')} | {preview} | {video} |")
        if completed:
            lines+=['', 'Completed task results:']
            for row in rows:
                if row['status']=='completed':
                    lines.append(f"- Scene {row['source_export_episode']} / {row['policy']}: "
                        f"task success={row['task_success']}; reason={row['failure_mode']}; "
                        f"scored truck load={row.get('task_payload_mass_kg')} kg; "
                        f"live truck load after 5000 actions={row.get('live_final_truck_mass_kg')} kg. "
                        f"[Status JSON]({Path(row['status_file']).relative_to(out)})")
        (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
        if completed==4:atomic_json(out/'completion.json',result)
        return result


def command(manifest, out, policy, eid, attempt, gpu):
    python=BASE/('.venv-pi05/bin/python' if policy=='pi05' else '.venv-smolvla-history/bin/python')
    args=[str(python),str(BASE/'scripts/eval_smolvla_success.py'),
        '--policy-type',policy,'--checkpoint',manifest['policies'][policy],
        '--dataset',manifest['dataset'],'--output',str(attempt/'evaluation'),
        '--reference-episode',str(out/'references'/f'export_episode_{eid:04d}.json'),
        '--episodes','1','--steps',str(manifest['steps_per_rollout']),
        '--repeat-cycles','10','--repeat-cycle-steps','0','--run-full-step-budget',
        '--replan-interval','3','--seed',str((manifest['selection_seed']+eid)%(2**31)),
        '--gpus',gpu,'--workers','1','--soil-backend','newton_mpm',
        '--cpu-threads','12','--video','--preview-steps','600','--headless',
        '--fresh-simulator','--render-every-physics-step','--progress-interval','15']
    if policy=='pi05':args+=['--allow-tighter-action-transfer']
    return args


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--policies',nargs='+',choices=['pi05','smolvla'],default=['pi05','smolvla'])
    args=parser.parse_args();out=args.output.resolve();manifest=read(out/'manifest.json')
    (out/'runtime').mkdir(exist_ok=True);(out/'videos').mkdir(exist_ok=True)
    stop=threading.Event()
    signal.signal(signal.SIGTERM,lambda *_:stop.set())
    signal.signal(signal.SIGINT,lambda *_:stop.set())
    source_paths=[BASE/'scripts'/n for n in ['eval_smolvla_success.py','repeat_eval.py','repeat_mass_physical.py',
        'evaluation_action_transfer.py','success_eval_runtime.py','eval_video.py','smolvla_history.py']]
    source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    atomic_json(out/'runtime/source_hashes.json',source_hashes)

    def state(policy,**values):
        atomic_json(out/'runtime'/f'{policy}.json',dict(policy=policy,controller_pid=os.getpid(),updated_utc=now(),**values))

    def worker(policy):
        gpu='0' if policy=='pi05' else '1'
        with (out/'runtime'/f'{policy}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if policy=='pi05':
                memory=read(out/'preflight/pi05_memory.json')
                while not memory.get('estimated_fits_with_old_paused_eval') and not (out/'resources_ready.json').exists():
                    if stop.is_set():state(policy,phase='stopped');return
                    if (out/'PI05_DEFERRED.json').exists():state(policy,phase='deferred_by_user');return
                    state(policy,phase='waiting_for_old_paused_eval_memory_release')
                    summarize(out,manifest);time.sleep(10)
            for eid in manifest['selected_source_episodes']:
                if stop.is_set():break
                if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=expected for p,expected in source_hashes.items()):
                    state(policy,phase='source_changed_before_next_job');return
                parent=out/'jobs'/policy/f'scene_{eid:04d}';parent.mkdir(parents=True,exist_ok=True)
                old=sorted(parent.glob('attempt_*'))
                if any((p/'evaluation/episodes/episode_0001/episode_status.json').exists() and
                       read(p/'evaluation/episodes/episode_0001/episode_status.json')['status']=='completed' for p in old):continue
                attempt=parent/f'attempt_{len(old)+1:02d}';attempt.mkdir()
                cmd=command(manifest,out,policy,eid,attempt,gpu)
                atomic_json(attempt/'launch.json',dict(command=cmd,started_utc=now(),source_export_episode=eid,policy=policy))
                with (attempt/'launch.log').open('w') as log:
                    process=subprocess.Popen(cmd,cwd=BASE,env=evaluation_environment(out),stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    state(policy,phase='running',source_export_episode=eid,child_pid=process.pid,gpu=gpu)
                    while process.poll() is None:
                        if stop.is_set():process.terminate();break
                        summary=summarize(out,manifest)
                        print(now(),policy,eid,f"completed={summary['completed_rollouts']}/4",flush=True);time.sleep(10)
                    process.wait()
                    atomic_json(attempt/'launch_result.json',dict(exit_code=process.returncode,finished_utc=now()))
                status=attempt/'evaluation/episodes/episode_0001/episode_status.json'
                if process.returncode or not status.exists() or read(status)['status']!='completed':
                    state(policy,phase='infrastructure_error',source_export_episode=eid,attempt=str(attempt));return
                if read(status)['policy_steps_executed']!=5000:
                    state(policy,phase='invalid_step_budget',source_export_episode=eid);return
                summarize(out,manifest)
            state(policy,phase='stopped' if stop.is_set() else 'completed')

    atomic_json(out/'controller.json',dict(pid=os.getpid(),policies=args.policies,started_utc=now()))
    with ThreadPoolExecutor(max_workers=len(args.policies)) as pool:
        jobs=[pool.submit(worker,p) for p in args.policies]
        for job in jobs:job.result()
    summarize(out,manifest)


if __name__=='__main__':main()
