"""Resume only missing ACT trials after disk exhaustion, preserving complete trials."""
import argparse,importlib.util,os,shutil,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(os.environ.get('ACT_PROJECT_ROOT',Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(ROOT/'scripts'))
from eval_smolvla_repeat40 import read,atomic_json,now,digest
from audit_act_model_only import audit_many,audit_run
from act_evaluation_storage import require_space,release_completed_caches
import run_act_training_alternatives as common
CURRENT=ROOT/'outputs/experiments/act_executed_drift_20261121_02'
HDR=CURRENT/'hdr_repeats'


def main(args):
    manifest=read(HDR/'manifest.json');jobs=manifest['jobs'];protected=manifest['protected_sha256']
    require_space(ROOT,16)
    for path,sha in protected.items():
        if digest(ROOT/path)!=sha:raise ValueError('Protected source changed '+path)
    spec=importlib.util.spec_from_file_location('frozen_current_evaluation',CURRENT/'evaluate_selected.py');repeat=importlib.util.module_from_spec(spec);spec.loader.exec_module(repeat)
    missing=[];preserved=[]
    for job in jobs:
        folder=Path(job['output']);status=repeat.result(folder)
        if status.exists() and read(status).get('status')=='completed' and not read(status).get('error'):
            if audit_run(folder)['status']!='passed':raise ValueError('Existing trial failed action audit')
            preserved.append(job['output']);continue
        if folder.exists():
            trace=folder/'evaluation/episodes/episode_0001/steps.jsonl'
            if trace.exists() and trace.stat().st_size:raise RuntimeError('Partial model trajectory requires explicit recovery policy; no splicing permitted')
            if not args.check_only:
                archive=CURRENT/'aborted_disk_full_20261122'/folder.name;archive.parent.mkdir(exist_ok=True);folder.rename(archive)
        missing.append(job)
    if args.check_only:
        print(dict(status='passed',preserved=len(preserved),missing=len(missing),protected_hashes_verified=True));return
    recovery=dict(started_utc=now(),reason='Disk full while cloning per-run shader cache; original controller/evaluator/model untouched',preserved_runs=preserved,rerun_only_missing_jobs=missing)
    atomic_json(CURRENT/'disk_recovery.json',recovery)
    atomic_json(CURRENT/'status.json',dict(status='evaluating_hdr_development_repeats',recovery='disk_full',completed_runs=len(preserved),missing_runs=len(missing)))
    atomic_json(HDR/'status.json',dict(status='evaluating',requested_runs=4,recovery='disk_full'))
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(repeat.work,HDR,missing,protected,Path(manifest['private_evaluator']))
        while not future.done():
            report=repeat.comparison(HDR,jobs,manifest['lighting_profile']);print(now(),report['completed_runs'],'/4 complete',flush=True);time.sleep(10)
        future.result()
    report=repeat.comparison(HDR,jobs,manifest['lighting_profile'])
    if report['completed_runs']!=4 or report['infrastructure_errors']:raise RuntimeError('Resumed trials incomplete or infrastructure failure')
    atomic_json(HDR/'status.json',dict(status='completed',completed_runs=4,finished_utc=now(),recovery='disk_full'))
    atomic_json(HDR/'completion.json',dict(status='completed',finished_utc=now()))
    audit_many([Path(j['output']) for j in jobs],CURRENT/'model_only_action_audit.json')
    baseline=read(ROOT/'outputs/experiments/act_hdr_baseline_repeatability_20261120_02/repeats/summary.json');comparison={}
    for scene in ['50','112']:
        a=[s for s in baseline['scenes'][scene]['samples'] if s['lighting_profile']=='single_authored_hdr_v1']
        b=[s for s in report['scenes'][scene]['samples'] if s['lighting_profile']=='single_authored_hdr_v1']
        comparison[scene]=dict(baseline=a,candidate=b,mean_successful_cycle_change=sum(s['successful_cycles'] for s in b)/len(b)-sum(s['successful_cycles'] for s in a)/len(a))
    atomic_json(CURRENT/'paired_hdr_comparison.json',dict(scenes=comparison,scope='Same HDR short development prefixes, not full ten-cycle success rates'))
    common.assert_external();release_completed_caches(HDR)
    atomic_json(CURRENT/'status.json',dict(status='completed',baseline_replaced=False,model_only_action_audit='passed',results=str(HDR),recovered_from_disk_full=True))
    print('Recovered audited evaluation complete',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true');a=p.parse_args()
    try:main(a)
    except Exception as e:
        if not a.check_only:atomic_json(CURRENT/'status.json',dict(status='recovery_infrastructure_error',error=repr(e),baseline_replaced=False))
        raise
