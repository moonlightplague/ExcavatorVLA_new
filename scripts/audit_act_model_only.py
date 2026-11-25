"""Fail closed when a reported ACT rollout contains any external action intervention."""
import argparse,json
from pathlib import Path


def read(path):return json.loads(Path(path).read_text())


def audit_run(folder):
    folder=Path(folder);evaluation=folder/'evaluation';episode=evaluation/'episodes/episode_0001'
    config=read(evaluation/'config.json');status=read(episode/'episode_status.json')
    forbidden=['supervisor','replay_reference','replay_full_reference','act_carry_guard']
    errors=[key+' enabled or missing' for key in forbidden if config.get(key) is not False]
    if config.get('expert_prefix_steps')!=0 or status.get('expert_prefix_steps_executed',0)!=0:errors.append('expert prefix')
    if config.get('execution_source')!='model' or status.get('execution_source')!='model':errors.append('execution_source')
    if config.get('act_temporal_window',0)!=0:errors.append('external temporal action ensemble')
    updates=read(episode/'goal_updates.json') if (episode/'goal_updates.json').exists() else []
    goal_updates={u['after_step']:u for u in updates};goal_changes=0
    total=nonmodel=guard=changed=missing=0
    with (episode/'steps.jsonl').open() as trace:
        for line in trace:
            row=json.loads(line);total+=1
            nonmodel+=row.get('action_source')!='model';guard+=row.get('carry_guard') is not None
            missing+=any(k not in row for k in ['predicted','applied','action_source','carry_guard'])
            changed+=row.get('predicted')!=row.get('applied')
            if row.get('step') in goal_updates:
                update=goal_updates[row['step']]
                if update.get('physics_ticks')!=0 or update.get('simulated_seconds')!=0 or update.get('joint_command_positions')!=row.get('joint_command_positions') or update.get('camera_sha256')!=row.get('camera_sha256'):
                    goal_changes+=1
    if total!=status.get('policy_steps_executed'):errors.append('executed step count mismatch')
    if status.get('status')!='completed' or status.get('error'):errors.append('rollout incomplete')
    if nonmodel or guard or changed or missing:errors.append('action path audit failed')
    if goal_changes:errors.append('Goal update changed physics, command pose or observations')
    return dict(status='passed' if not errors else 'failed',folder=str(folder),steps=total,nonmodel_steps=nonmodel,guard_steps=guard,
        predicted_applied_differences=changed,missing_trace_fields=missing,goal_updates_checked=len(updates),goal_update_physics_or_pose_changes=goal_changes,errors=errors,
        environment_role='Preset per-cycle task coordinates, physical execution and event counting; no action decisions from task phase or success logic',
        scope='Evaluator command provenance and per-step action equality; ordinary simulator joint drives/physics are not a learned action policy')


def audit_many(folders,output):
    rows=[audit_run(folder) for folder in folders]
    result=dict(status='passed' if all(r['status']=='passed' for r in rows) else 'failed',runs=len(rows),steps=sum(r['steps'] for r in rows),
        nonmodel_steps=sum(r['nonmodel_steps'] for r in rows),guard_steps=sum(r['guard_steps'] for r in rows),
        predicted_applied_differences=sum(r['predicted_applied_differences'] for r in rows),results=rows)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    if result['status']!='passed':raise ValueError('External actions or unverified rollout detected; see '+str(output))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--roots',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    folders=sorted(set(path.parent for root in a.roots for path in root.rglob('evaluation') if (path/'config.json').is_file()))
    if not folders:raise ValueError('No evaluation runs found')
    result=audit_many(folders,a.output);print(json.dumps({k:v for k,v in result.items() if k!='results'}))
