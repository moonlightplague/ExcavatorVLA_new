"""Conditionally continue one model-only optimization direction after current evaluation."""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(os.environ.get('ACT_PROJECT_ROOT',Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(ROOT/'scripts'))
from eval_smolvla_repeat40 import atomic_json,read,now,digest
from audit_act_model_only import audit_many
import run_act_training_alternatives as common
from act_evaluation_storage import require_space,release_completed_caches
CURRENT=ROOT/'outputs/experiments/act_executed_drift_20261121_02'
HDR_BASE=ROOT/'outputs/experiments/act_hdr_baseline_repeatability_20261120_02/repeats'
THREE_ARM=ROOT/'outputs/experiments/act_visual_direction_validation_20261120_02'
NEXT=ROOT/'outputs/experiments/act_execution_focused_20261121_01'
RUNTIME=ROOT/'outputs/experiments/act_training_alternatives_runtime_20261119_01/python'
TRAINER=ROOT/'scripts/train_act_execution_focused.py'
PREFLIGHT=ROOT/'outputs/experiments/act_execution_focused_checks_20261121_01/preflight.json'


def process_active(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False


def promising_screen(pairs):
    return all(p['successful_cycle_change']>=-1e-9 and p['completed_cycle_change']>=-1e-9 for p in pairs.values()) and any(p['successful_cycle_change']>=.5-1e-9 for p in pairs.values())


def comparison():
    baseline=read(HDR_BASE/'summary.json');later=read(THREE_ARM/'summary.json');candidate=read(CURRENT/'hdr_repeats/summary.json');pairs={}
    for scene in ['50','112']:
        old=[dict(success=s['successful_cycles'],complete=s['completed_cycles'],source='initial_hdr_repeats') for s in baseline['scenes'][scene]['samples'] if s['lighting_profile']=='single_authored_hdr_v1']
        old += [dict(success=s['prefix_successful_cycles'],complete=s['prefix_completed_cycles'],source='fresh_three_arm_2000_step_prefix') for s in later['results'] if str(s['scene'])==scene and s['arm']=='baseline']
        new=[dict(success=s['successful_cycles'],complete=s['completed_cycles']) for s in candidate['scenes'][scene]['samples'] if s['lighting_profile']=='single_authored_hdr_v1']
        if len(old)!=3 or len(new)!=2:raise ValueError('Incomplete frozen comparison data')
        def stats(rows):return dict(successful_cycle_mean=sum(s['success'] for s in rows)/len(rows),completed_cycle_mean=sum(s['complete'] for s in rows)/len(rows),samples=rows)
        a=stats(old);b=stats(new);pairs[scene]=dict(baseline=a,candidate=b,successful_cycle_change=b['successful_cycle_mean']-a['successful_cycle_mean'],completed_cycle_change=b['completed_cycle_mean']-a['completed_cycle_mean'])
    promising=promising_screen(pairs)
    return dict(promising=promising,scenes=pairs,criterion='No decrease in mean successful or completed cycles on either known scene, and >=0.5 mean successful-cycle gain on at least one scene',
        scope='Development screening from 3 prior source runs and 2 candidate runs, same HDR profile/GPU/seed/2000-step prefix. Unequal sample counts; not statistical significance or model promotion.')


def full_validation(output):
    require_space(ROOT,12)
    script=(CURRENT/'evaluate_selected.py').read_text()
    if script.count("'--steps':'2000'")!=1:raise ValueError('Unexpected short evaluation command')
    script=script.replace("'--steps':'2000'","'--steps':'5000'")
    script=script.replace('steps=2000,repeats_per_scene=2','steps=5000,repeats_per_scene=2')
    script=script.replace('full_ten_cycle_success_measurement=False','full_ten_cycle_success_measurement=True')
    path=output/'full_validation.py';path.write_text(script);compile(script,str(path),'exec')
    atomic_json(output/'status.json',dict(status='running_full_ten_cycle_validation',updated_utc=now(),baseline_replaced=False))
    with (output/'full_validation.log').open('w') as log:
        subprocess.run([str(RUNTIME),'-u','-B',str(path),'--output',str(output/'full_repeats'),'--evaluator',str(ROOT/'outputs/experiments/act_hdr_baseline_repeatability_20261120_02/evaluator_single_hdr.py'),'--lighting-profile','single_authored_hdr_v1'],cwd=ROOT,
            env={**os.environ,'ACT_PROJECT_ROOT':str(ROOT),'CUDA_VISIBLE_DEVICES':'0,1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'},stdout=log,stderr=subprocess.STDOUT,check=True)
    folders=sorted((output/'full_repeats').glob('scene_*'))
    if len(folders)!=4 or read(output/'full_repeats/status.json')['status']!='completed':raise RuntimeError('Full evaluation infrastructure error')
    audit_many(folders,output/'full_model_only_audit.json')
    rows=[]
    for folder in folders:
        row=read(folder/'evaluation/episodes/episode_0001/episode_status.json')
        rows.append(dict(run=folder.name,completed_cycles=row['repeat_cycles_completed'],successful_cycles=row['repeat_cycles_successful'],full_success=bool(row['repeat_collection_success']),steps=row['policy_steps_executed']))
    passed=all(r['full_success'] for r in rows)
    atomic_json(output/'full_results.json',dict(all_four_successful=passed,runs=rows,scope='Four full-budget known-scene development runs; not unseen-scene generalization'))
    release_completed_caches(output/'full_repeats')
    return passed


def main(args):
    proof=read(PREFLIGHT)
    if proof['status']!='passed' or proof['trainer_sha256']!=digest(TRAINER):raise ValueError('Next direction preflight mismatch')
    if args.resume:
        saved=read(args.output/'manifest.json')
        if saved['next_trainer_sha256']!=digest(args.output/'next_trainer.py') or read(args.output/'next_preflight.json')['trainer_sha256']!=digest(args.output/'next_trainer.py'):raise ValueError('Saved followup snapshot mismatch')
        atomic_json(args.output/'recovery_resume.json',dict(updated_utc=now(),reason='Disk full interrupted status writer; source model and prepared next direction unchanged'))
    else:
        args.output.mkdir(parents=True,exist_ok=False);shutil.copy2(Path(__file__),args.output/'run.py');shutil.copy2(TRAINER,args.output/'next_trainer.py');shutil.copy2(PREFLIGHT,args.output/'next_preflight.json')
        atomic_json(args.output/'manifest.json',dict(created_utc=now(),current_experiment=str(CURRENT),next_experiment=str(NEXT),next_direction='Execution-focused expert supervision; no new runtime controller',next_trainer_sha256=digest(TRAINER),next_steps=1500,one_followup_direction_only=True,automatic_model_promotion=False,
            authority='User explicitly authorized a different model optimization direction if current evaluation is inadequate',external_sha256=common.assert_external(),criteria='No mean successful/completed cycle regression on either scene and >=0.5 mean success gain on one -> full validation; otherwise new direction. Any full validation failure -> new direction. Infrastructure/audit failures stop rather than count as model failures.'))
    launch_path=Path(str(CURRENT)+'.recovery_launch.json')
    launcher=read(launch_path if launch_path.exists() else Path(str(CURRENT)+'.launch.json'))
    while True:
        status=read(CURRENT/'status.json');audit=CURRENT/'model_only_action_audit.json'
        if status['status']=='completed' and audit.exists() and read(audit)['status']=='passed':break
        if (CURRENT/'error_run.json').exists() or (not process_active(launcher['pid']) and status['status']!='completed'):raise RuntimeError('Current runner stopped before audited completion')
        atomic_json(args.output/'status.json',dict(status='waiting_for_current_audited_evaluation',current_status=status,updated_utc=now()))
        time.sleep(15)
    common.assert_external();report=read(CURRENT/'hdr_repeats/summary.json')
    if report['completed_runs']!=4 or report['infrastructure_errors']:raise RuntimeError('Current evaluation infrastructure errors')
    audit_many(sorted((CURRENT/'hdr_repeats').glob('scene_*')),args.output/'current_model_only_audit.json')
    release_completed_caches(CURRENT/'hdr_repeats')
    decision=comparison();atomic_json(args.output/'decision.json',decision)
    if decision['promising'] and full_validation(args.output):
        atomic_json(args.output/'status.json',dict(status='completed_full_known_scene_validation',baseline_replaced=False,updated_utc=now()));return
    if NEXT.exists():raise RuntimeError('Refusing duplicate next experiment')
    require_space(ROOT,12)
    if digest(args.output/'next_trainer.py')!=proof['trainer_sha256']:raise RuntimeError('Snapshot changed')
    atomic_json(args.output/'status.json',dict(status='running_next_model_direction',next_experiment=str(NEXT),updated_utc=now()))
    with (args.output/'next_direction.log').open('w') as log:
        subprocess.run([str(RUNTIME),'-u','-B',str(args.output/'next_trainer.py'),'--output',str(NEXT),'--steps','1500'],cwd=ROOT,
            env={**os.environ,'ACT_PROJECT_ROOT':str(ROOT),'ACT_EXECUTION_FOCUS_CHECKS':str(args.output/'next_preflight.json'),'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'},stdout=log,stderr=subprocess.STDOUT,check=True)
    result=read(NEXT/'status.json')
    if result['status']=='completed':
        audit_many(sorted((NEXT/'hdr_repeats').glob('scene_*')),args.output/'next_model_only_audit.json')
        release_completed_caches(NEXT/'hdr_repeats')
    atomic_json(args.output/'status.json',dict(status='completed_one_followup_direction',next_status=result,baseline_replaced=False,updated_utc=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=lambda p:Path(p).resolve(),required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    try:main(a)
    except Exception as e:
        if a.output.exists():atomic_json(a.output/'status.json',dict(status='stopped_infrastructure_or_audit_error',error=repr(e),baseline_replaced=False,updated_utc=now()))
        raise
