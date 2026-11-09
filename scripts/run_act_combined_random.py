"""Single-arm random-scene evaluation, optionally using the frozen handoff wait."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(os.environ.get('ACT_PROJECT_ROOT',str(Path(__file__).resolve().parents[1]))).resolve()
SOURCE=ROOT/'outputs/experiments/act_unload_pose_hold_20261106_01/validation8'
ARCHIVE=ROOT/'outputs/baselines/act_phw40k_carry275_20261105'


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def make_command(template,frozen,checkpoint,out,episodes,seed):
    forbidden={'--reference-episode','--replay-reference','--expert-prefix-steps','--scene-manifest'}
    if forbidden.intersection(template):
        raise ValueError('Random evaluation cannot inherit references, replay or expert prefixes')
    command=list(template)
    command[2]=str(frozen/'scripts/eval_smolvla_success.py')
    for flag,value in [('--checkpoint',str(checkpoint)),('--output',str(out/'evaluation')),
        ('--episodes',str(episodes)),('--seed',str(seed)),('--steps','500'),('--workers','2'),
        ('--gpus','0,1'),('--post-rollout-observe-steps','0')]:
        if command.count(flag)!=1:raise ValueError(f'Unexpected base command flag: {flag}')
        command[command.index(flag)+1]=value
    return command


def install_handoff_wait(frozen,source,checkpoint):
    source=Path(source).resolve()
    protocol=read(source/'protocol.json')
    if (protocol.get('comparison')!='handoff_wait' or protocol.get('checkpoint')!=str(checkpoint)
            or protocol.get('latch_settings',{}).get('wait_for_act_opening_steps')!=20):
        raise ValueError('Expected the tested 20-step handoff-wait candidate with the same ACT checkpoint')
    if not read(source/'postflight_audit.json')['passed'] or not read(source/'results.json')['advance']:
        raise ValueError('Handoff-wait source did not pass its development checks')
    for name,digest in read(source/'inputs_sha256.json').items():
        if sha(name)!=digest:raise ValueError(f'Tested candidate input changed: {name}')
    for name in ('act_carry_guard.py','act_recovery_expert.py','excavator_reachability.py'):
        if sha(frozen/'scripts'/name)!=sha(source/'frozen_sources/scripts'/name):
            raise ValueError(f'Handoff-wait candidate changes an unrelated controller: {name}')
    target=frozen/'scripts/act_stall_recovery.py'
    legacy=target.read_text()
    # The matched-state harness adds a true-by-default switch to its old arm.
    legacy=legacy.replace('if (not self.unload_pose_hold_active and loaded and opening_intent and over_bed',
        'if (self.unload_pose_hold_enabled and not self.unload_pose_hold_active and loaded and opening_intent and over_bed',1)
    init='            self.unload_pose_hold_active = False\n'
    legacy=legacy.replace(init,'            self.unload_pose_hold_enabled = True\n'+init,1)
    if legacy!=(source/'frozen_sources/scripts/act_stall_recovery.py').read_text():
        raise ValueError('Tested candidate does not derive from the frozen combined controller')
    actor=source/'frozen_sources/scripts/act_stall_recovery_latched.py'
    helper=source/'frozen_sources/scripts/act_unload_pose_latch.py'
    shutil.copy2(actor,target)
    shutil.copy2(helper,frozen/'scripts/act_unload_pose_latch.py')
    # Include the added controller dependency in native provenance as well as
    # in this runner's complete frozen-input manifest. Scoring is unchanged.
    runtime=frozen/'scripts/success_eval_runtime.py'
    code=runtime.read_text();marker="    if protocol.get('scene_manifest_sha256'):\n"
    if code.count(marker)!=1:raise ValueError('Unexpected native provenance implementation')
    addition="    sources.extend(base / 'scripts' / name for name in ('act_stall_recovery.py', 'act_recovery_expert.py', 'excavator_reachability.py', 'act_unload_pose_latch.py'))\n"
    runtime.write_text(code.replace(marker,addition+marker))
    return dict(source=str(source),controller_sha256=sha(actor),helper_sha256=sha(helper),
        protocol_sha256=sha(source/'protocol.json'),development_audit_sha256=sha(source/'postflight_audit.json'))


def prepare(out,episodes,seed,handoff_wait_source=None):
    out.mkdir(parents=True,exist_ok=False)
    source_protocol=read(SOURCE/'protocol.json')
    if source_protocol['recovery_mode']!='secure_align':raise ValueError('Unexpected controller mode')
    frozen=out/'frozen_sources';frozen.mkdir()
    for path in (SOURCE/'frozen_sources').iterdir():
        if path.name=='scripts':
            shutil.copytree(path,frozen/path.name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        else:(frozen/path.name).symlink_to(path.resolve(),target_is_directory=path.is_dir())
    shutil.copy2(ROOT/'scripts/run_act_combined_random.py',frozen/'scripts/run_act_combined_random.py')
    shutil.copy2(SOURCE/'robot_model.json',out/'robot_model.json')
    checkpoint=Path(source_protocol['checkpoint'])
    if sha(checkpoint/'model.safetensors')!=source_protocol['model_sha256']:
        raise ValueError('Retained ACT weights differ from the validated candidate')
    original=ARCHIVE/'frozen_sources/ExcavatorVLA-smolvla/run_simulation.py'
    simulator=frozen/'ExcavatorVLA-smolvla/run_simulation.py'
    if sha(simulator)!=sha(original):raise ValueError('Random test must use the original simulator without snapshot restoration')
    command=make_command(read(ARCHIVE/'evidence/protocol.json')['command'],frozen,checkpoint,out,episodes,seed)
    source_names=('act_stall_recovery.py','act_recovery_expert.py','excavator_reachability.py','eval_smolvla_success.py')
    controller_files={name:sha(frozen/'scripts'/name) for name in source_names}
    if any(controller_files[name]!=sha(SOURCE/'frozen_sources/scripts'/name) for name in source_names):
        raise ValueError('Frozen combined controller differs from the validated source')
    wait_source=install_handoff_wait(frozen,handoff_wait_source,checkpoint) if handoff_wait_source else None
    if wait_source:
        source_names=(*source_names,'act_carry_guard.py','act_unload_pose_latch.py')
        controller_files={name:sha(frozen/'scripts'/name) for name in source_names}
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),workspace=str(ROOT),
        baseline_id=source_protocol['baseline_id'],model_sha256=source_protocol['model_sha256'],
        checkpoint=str(checkpoint),dataset=str(ROOT/'data/export_phw'),
        episodes=episodes,seed=seed,seeds=list(range(seed,seed+episodes)),
        variants=['secure_align_plus_unload_pose_hold'+('_plus_unload_handoff_wait' if wait_source else '')],control_group=False,
        execution_source='model_with_carry_guard_and_postdig_recovery',
        candidate_source=str(SOURCE),controller_files_sha256=controller_files,
        secure_align_settings=source_protocol['settings'],
        unload_pose_hold=dict(enabled=True,opening_intent_min_rad_s=.12,
            loaded_fill_min=.002,release_geometry='original opening_pose',maximum_steps_per_activation=50,
            empty_observations_to_release=3,parent_target='current measured swing/boom/arm',
            bucket_command='original ACT, followed by the original carry guard'),
        random_scene_sampler='Original archived sample_random_scene_profile; new seed for each scene',
        sampling_exclusions='Original geometry-only truck/body collision checks; no policy-outcome exclusions or seed replacement',
        initial_pose='Original fixed dataset episode 0, same as retained baseline',
        steps=500,fps=10,replan_interval=3,soil_backend='physx',render_every_physics_step=True,
        fresh_simulator_per_episode=True,success='>=100 initial-pile particles in truck and bucket>=-0.8 rad for 3 consecutive observations',
        denominator=episodes,post_rollout_observe_steps=0,no_training=True,weights_unchanged=True,
        baseline_replaced=False,scope='Single combined hybrid candidate on new random scenes; direct success rate, no causal improvement claim',
        command=command)
    if wait_source:
        protocol.update(handoff_wait_source=wait_source,unload_handoff_wait=dict(enabled=True,maximum_steps=20,
            entry='verified completed_secure_align handoff, loaded, over original bed geometry, legacy hold inactive',
            parent_goal='measured parent pose captured once during pre-opening wait',
            bucket_command='original ACT followed by original carry guard',
            exit='original hold starts, 20-step timeout, three empty observations, or recovery resumes',
            legacy_unloading_hold='preserved verbatim'))
    write(out/'protocol.json',protocol)
    paths=[*frozen.glob('scripts/*.py'),out/'protocol.json',out/'robot_model.json',
           *checkpoint.glob('*'),ROOT/'data/export_phw/meta/info.json']
    write(out/'inputs_sha256.json',{str(p):sha(p) for p in paths if p.is_file()})
    write(out/'status.json',dict(status='prepared',expected_episodes=episodes,variants=protocol['variants']))
    return protocol


def aggregate(records,expected):
    valid=[r for r in records.values() if r['status']=='completed']
    errors=[r for r in records.values() if r['status']!='completed']
    if len(records)>expected:raise ValueError('Unexpected extra episodes')
    successes=sum(r['success'] for r in valid)
    complete=len(valid)==expected and not errors
    return dict(expected_episodes=expected,completed_episodes=len(valid),successes=successes,
        failures=len(valid)-successes,infrastructure_error_episodes=len(errors),
        result_valid=complete,success_rate=successes/expected if complete else None,
        successful_seeds=[r['scene_seed'] for r in valid if r['success']],
        recovery_triggered_episodes=sum(r['recovery_triggered'] for r in valid),
        unload_hold_episodes=sum(r['unload_hold_steps']>0 for r in valid),
        handoff_wait_episodes=sum(r.get('handoff_wait_steps',0)>0 for r in valid),
        handoff_wait_steps=sum(r.get('handoff_wait_steps',0) for r in valid),
        handoff_wait_timeout_events=sum(r.get('handoff_wait_timeout_events',0) for r in valid),
        handoff_wait_timeout_episodes=sum(r.get('handoff_wait_timeout_events',0)>0 for r in valid),
        episodes_ended_waiting=sum(r.get('episode_ended_waiting',False) for r in valid))


def controller_metrics(rows,require_wait=False):
    traces=[(r.get('carry_guard') or {}).get('stall_recovery',{}) for r in rows]
    if any('unload_pose_hold' not in t for t in traces):raise ValueError('Missing combined-controller trace')
    if require_wait and any('unload_handoff_wait' not in t for t in traces):
        raise ValueError('Missing handoff-wait trace in enabled candidate')
    waits=[t.get('unload_handoff_wait',{}) for t in traces]
    for t,w in zip(traces,waits):
        if w.get('steps',0)>20:raise ValueError('Handoff wait exceeded the frozen budget')
        if w.get('started') and not (t.get('handback') and t.get('phase')=='completed_secure_align'
                and not t['unload_pose_hold'].get('active')):
            raise ValueError('Handoff wait started without a verified legacy-inactive handoff')
    def first(predicate):
        return next((row['step'] for row,t,w in zip(rows,traces,waits) if predicate(t,w)),None)
    reasons=Counter(w['release_reason'] for w in waits if w.get('release_reason'))
    return dict(recovery_triggered=any(t.get('triggered',False) for t in traces),
        recovery_steps=sum(t.get('override',False) and not t['unload_pose_hold'].get('active',False)
            and not w.get('active',False) for t,w in zip(traces,waits)),
        unload_hold_steps=sum(t['unload_pose_hold'].get('active',False) for t in traces),
        handoff_wait_starts=sum(w.get('started',False) for w in waits),
        handoff_wait_steps=sum(w.get('active',False) for w in waits),
        handoff_wait_timeout_events=reasons['act_opening_wait_timeout'],
        handoff_wait_release_reasons=dict(reasons),
        episode_ended_waiting=bool(waits and waits[-1].get('active',False)),
        first_verified_alignment_step=first(lambda t,w:t.get('handback') and t.get('phase')=='completed_secure_align'),
        first_unload_hold_step=first(lambda t,w:t['unload_pose_hold'].get('started')),
        first_handoff_wait_step=first(lambda t,w:w.get('started')))


def assess(folder,index,protocol,evaluation_id):
    from audit_smolvla_success import audit_episode
    status=read(folder/'episode_status.json')
    if status['status']!='completed':return dict(status='infrastructure_error',episode_status=status)
    if status['scene_seed']!=protocol['seeds'][index-1] or status['evaluation_id']!=evaluation_id:
        raise ValueError('Wrong episode seed or evaluation identity')
    if status['execution_source']!=protocol['execution_source'] or status['evaluation_profile']!='standard':
        raise ValueError('Expected the single combined candidate under standard random evaluation')
    audit=audit_episode(folder,status)
    rows=[json.loads(line) for line in (folder/'steps.jsonl').read_text().splitlines()]
    metrics=controller_metrics(rows,protocol.get('unload_handoff_wait',{}).get('enabled',False))
    if not (folder/'evaluation.mp4').is_file() or (folder/'evaluation.mp4').stat().st_size<=0:
        raise ValueError('Episode video missing')
    terminal=rows[-1]['particle_transfer_metrics']
    return dict(status='completed',episode_index=index,scene_seed=status['scene_seed'],success=status['success'],
        policy_steps=status['policy_steps_executed'],payload_particles=terminal['payload_particles'],
        retained_particles=terminal['retained_particles'],spillage_particles=terminal['spillage_particles'],
        **metrics,audit=audit,video=str(folder/'evaluation.mp4'))


def report(out,protocol,records,state):
    summary=aggregate(records,protocol['episodes'])
    write(out/'results.json',dict(summary,episodes=records))
    rate=f"{summary['success_rate']:.1%}" if summary['result_valid'] else '运行中，尚无最终成功率'
    with_wait=protocol.get('unload_handoff_wait',{}).get('enabled',False)
    title='ACT脱困＋卸料保持'+('＋交接等待' if with_wait else '')
    lines=[f'# {title}：随机{protocol["episodes"]}场成功率','',
        f"状态：{state}；完成{summary['completed_episodes']}/{protocol['episodes']}场。",
        f"完整任务成功：{summary['successes']}/{protocol['episodes']}；最终成功率：{rate}。",'',
        '单一组合候选：secure_align＋unload_pose_hold'+('＋unload_handoff_wait' if with_wait else '')+'，沿用PHW40k和原opening_pose保料层。',
        f"预先固定新seed {protocol['seed']}–{protocol['seed']+protocol['episodes']-1}，每场全新模拟器，随机料堆/车斗位置，原固定初姿。",
        'PhysX、逐物理步渲染、10Hz、每3步重规划、500步上限。成功要求至少100粒原料堆粒子入车且斗角≥−0.8rad连续三观测。',
        '不设对照，不从旧场景回放或恢复快照，不按结果换seed或删除失败。测试结果仅描述当前组合的本批成功率。','',
        '| 场次 | seed | 完整成功 | 入车粒子 | 末态斗内 | 步数 | 脱困步数 | 卸料保持步数 | 交接等待步数 | 等待超时次数 | 验证到位步 | 视频 |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for key,r in sorted(records.items(),key=lambda item:int(item[0])):
        if r['status']!='completed':
            lines.append(f'| {key} | — | 基础设施错误，仍保留分母 | — | — | — | — | — | — | — | — | — |')
            continue
        lines.append(f"| {key} | {r['scene_seed']} | {'是' if r['success'] else '否'} | {r['payload_particles']} | "
            f"{r['retained_particles']} | {r['policy_steps']} | {r['recovery_steps']} | {r['unload_hold_steps']} | "
            f"{r.get('handoff_wait_steps',0)} | {r.get('handoff_wait_timeout_events',0)} | {r.get('first_verified_alignment_step')} | [视频]({r['video']}) |")
    lines+=['',f"触发脱困{summary['recovery_triggered_episodes']}场，触发卸料保持{summary['unload_hold_episodes']}场。",
        '模型权重与27.5%保留基准不替换。逐步命令、图像哈希、场景几何和物理时序保存在evaluation/episodes。']
    if with_wait:
        lines+=['',f"交接等待触发{summary['handoff_wait_episodes']}场，超时{summary['handoff_wait_timeout_events']}次；"+
            f"{summary['episodes_ended_waiting']}场在仍等待时到达终止点。",
            '开斗前等待最多20步；原卸料保持目标、触发阈值和50步预算保留。到位步是已验证secure_align交接动作的从零开始编号；自然到车未经过该交接时记None。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return summary


def run(out):
    protocol=read(out/'protocol.json');frozen=out/'frozen_sources'
    sys.path[:0]=[str(frozen/'scripts'),str(ARCHIVE/'runtime_dependencies')]
    environment=dict(os.environ,ACT_POSTDIG_RECOVERY='1',ACT_POSTDIG_MODE='secure_align',
        ACT_STALL_ROBOT_MODEL=str(out/'robot_model.json'),PYTHONDONTWRITEBYTECODE='1',
        HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',HF_HOME=str(ROOT/'models/hf-cache'),
        TORCH_HOME=str(ROOT/'models/torch-cache'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        PYTHONPATH=str(ARCHIVE/'runtime_dependencies'))
    environment.pop('CUDA_VISIBLE_DEVICES',None)
    records={};child=None
    def interrupted(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        with (out/'evaluation.log').open('x') as log:
            child=subprocess.Popen(protocol['command'],cwd=frozen,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(out/'launch.json',dict(controller_pid=os.getpid(),evaluation_pid=child.pid,command=protocol['command']))
            while True:
                changed=False
                provenance=out/'evaluation/provenance.json'
                if provenance.exists():
                    evaluation_id=read(provenance)['evaluation_id']
                    for p in sorted((out/'evaluation/episodes').glob('*/episode_status.json')):
                        index=int(p.parent.name.split('_')[-1])
                        if str(index) in records:continue
                        records[str(index)]=assess(p.parent,index,protocol,evaluation_id);changed=True
                summary=report(out,protocol,records,'running')
                progress={p.parent.name:read(p) for p in (out/'evaluation/workers').glob('*/progress.json')}
                write(out/'status.json',dict(status='running',**summary,workers=progress,
                    updated_utc=datetime.now(timezone.utc).isoformat()))
                if changed:print(json.dumps(summary,ensure_ascii=False),flush=True)
                code=child.poll()
                if code is not None:
                    if code or not summary['result_valid']:
                        raise RuntimeError(f'Evaluation exited {code}: completed {summary["completed_episodes"]}/{protocol["episodes"]}')
                    break
                time.sleep(5)
        from success_eval_runtime import content_id,file_manifest
        provenance=read(out/'evaluation/provenance.json')
        config=read(out/'evaluation/config.json')
        if provenance['evaluation_id']!=config['evaluation_id'] or provenance['evaluation_id']!=content_id(
                {k:v for k,v in provenance.items() if k!='evaluation_id'}):
            raise ValueError('Evaluation provenance identity differs')
        if file_manifest([frozen/p for p in provenance['source_files']],frozen)!=provenance['source_files']:
            raise ValueError('Frozen evaluation dependencies changed')
        changed=[p for p,digest in read(out/'inputs_sha256.json').items() if sha(p)!=digest]
        if changed:raise ValueError(f'Protected input changed: {changed}')
        native=read(out/'evaluation/success_rate_summary.json')
        # The native "official" flag is reserved for unassisted raw-model
        # execution. Preserve that distinction and validate this hybrid batch
        # against the native completed counts instead of relabelling it.
        if (native['completed_episodes']!=protocol['episodes'] or native['infrastructure_error_episodes']
                or native['successes']!=summary['successes']
                or native['success_rate_over_completed_episodes']!=summary['success_rate']):
            raise ValueError('Independent and native success summaries differ')
        write(out/'evaluation_audit.json',dict(status='passed',evaluation_id=provenance['evaluation_id'],
            execution_source=protocol['execution_source'],expected_episodes=protocol['episodes'],
            source_files_verified=True,hybrid_success_rate=summary['success_rate'],
            native_raw_model_official_result_valid=native['official_result_valid'],
            episodes=[r['audit'] for r in records.values()]))
        summary=report(out,protocol,records,'completed')
        write(out/'status.json',dict(status='completed',**summary,inputs_unchanged=True,
            baseline_replaced=False,workers={},completed_utc=datetime.now(timezone.utc).isoformat()))
        print(json.dumps(summary,ensure_ascii=False),flush=True)
    except BaseException as error:
        write(out/'status.json',dict(status='failed',error=str(error),**aggregate(records,protocol['episodes'])))
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--episodes',type=int,default=40)
    parser.add_argument('--seed',type=int,default=4100000)
    parser.add_argument('--handoff-wait-source',type=Path,help='Frozen handoff_wait study that passed its development audit')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--run-prepared',action='store_true')
    args=parser.parse_args()
    if args.episodes<1 or args.seed<0:parser.error('Positive episode count and nonnegative seed required')
    out=args.output.resolve()
    if not args.run_prepared:prepare(out,args.episodes,args.seed,args.handoff_wait_source)
    if not args.prepare_only:run(out)


if __name__=='__main__':main()
