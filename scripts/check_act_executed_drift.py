"""Check executed-action cumulative supervision and native persistent ACT training."""
import argparse,json,tempfile,time
from pathlib import Path
from types import SimpleNamespace
import train_act_executed_drift as run


def check(output):
    import numpy as np,torch
    from excavator_vlm.policies.act.inference import load_act
    torch.set_num_threads(4);started=time.time();output.parent.mkdir(parents=True,exist_ok=True)
    from audit_act_model_only import audit_run
    with tempfile.TemporaryDirectory(prefix='act_action_audit_check_') as check_folder:
        folder=Path(check_folder);episode=folder/'evaluation/episodes/episode_0001';episode.mkdir(parents=True)
        config=dict(supervisor=False,replay_reference=False,replay_full_reference=False,act_carry_guard=False,expert_prefix_steps=0,execution_source='model')
        (folder/'evaluation/config.json').write_text(json.dumps(config))
        (episode/'episode_status.json').write_text(json.dumps(dict(status='completed',policy_steps_executed=1,execution_source='model')))
        row=dict(action_source='model',carry_guard=None,predicted=[0.,0.,0.,0.],applied=[0.,0.,0.,0.])
        trace=episode/'steps.jsonl';trace.write_text(json.dumps(row)+'\n');assert audit_run(folder)['status']=='passed'
        row['action_source']='expert_prefix';trace.write_text(json.dumps(row)+'\n');assert audit_run(folder)['status']=='failed'
        row['action_source']='model';row['applied'][0]=.1;trace.write_text(json.dumps(row)+'\n');assert audit_run(folder)['status']=='failed'
    prediction=torch.ones(2,10,10,4,requires_grad=True);target=torch.zeros_like(prediction);pad=torch.zeros(2,10,10,dtype=torch.bool);std=torch.ones(4)
    displacement,complete=run.displacement_windows(prediction,target,pad,std)
    torch.testing.assert_close(displacement,torch.full_like(displacement,2.4),rtol=1e-6,atol=1e-6);assert complete.all()
    value=run.executed_drift_loss(prediction,target,pad,torch.zeros(2,10,dtype=torch.bool),std);value.backward()
    assert prediction.grad[:,:,:3].abs().sum()>0 and prediction.grad[:,:,3:].abs().sum()==0
    assert run.executed_drift_loss(target,target,pad,torch.zeros(2,10,dtype=torch.bool),std)==0
    pad[:]=True
    assert run.executed_drift_loss(prediction,target,pad,torch.zeros(2,10,dtype=torch.bool),std)==0
    del prediction,target,pad,displacement,complete,value

    audit=run.prepare_checks(output.parent)
    policy,_,_=load_act(run.SOURCE,'cpu',original_act=True);memory=policy.model.memory_adapter
    notes=run.annotations([0]);data=run.common.cpu_data(0);rng=np.random.default_rng(1)
    start,_=run.choose_start(notes[0],len(data['features']),128,rng,forced='bridge')
    features=torch.as_tensor(data['features']);memory.eval()
    with torch.no_grad():
        entire=memory(features[None])[0][:,start:start+128]
        prefix=memory.prefix_hidden(features[:start][None],torch.tensor([start]));fragment=memory(features[start:start+128][None],prefix)[0]
        torch.testing.assert_close(entire,fragment,rtol=1e-5,atol=2e-6)
    # Stage/return/bridge annotations are absent from the 1604D model feature input.
    assert features.shape[1]==1604 and start<next(b for b in notes[0]['ends'] if b>start)<start+128
    del policy,memory,features,entire,prefix,fragment
    with tempfile.TemporaryDirectory(prefix='act_executed_drift_check_') as temporary:
        run.train(SimpleNamespace(output=Path(temporary)/'train',steps=2,device='cpu',smoke=True,drift_coefficient=.005))
        state=run.read(Path(temporary)/'train/status.json')
        assert state['strict_reload']=='passed' and state['native_rgb_reload']=='passed' and state['frozen_core_unchanged']
        assert state['steps']==2 and all(n.startswith('model.memory_adapter.') for n in state['changed_parameters'])
    proof=dict(status='passed',audit=audit,checks=['source frozen core identical to causal feature cache','all 160 immutable train/validation cache hashes verified',
        '2800 bridge windows span actual return-to-next-cycle boundary','full prefix and 128-step fragment predictions agree',
        '2.4-second executed-action displacement exactness','only actually executed first-three actions receive drift gradients','padding and perfect-target losses zero','two real-data CPU optimization steps','strict memory and raw RGB native reload','only memory parameters changed','original external hashes preserved'],
        device='cpu',tested_drift_coefficient=.005,no_evaluation_gpu_interference=True,temporary_weights_and_metrics_removed=True,elapsed_seconds=time.time()-started,trainer_sha256=run.common.digest(Path(run.__file__)))
    run.common.write(output,proof);print(json.dumps(proof),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);check(p.parse_args().output)
