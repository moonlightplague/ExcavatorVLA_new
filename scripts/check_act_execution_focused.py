"""Check execution-focused horizon gradients and native ACT memory fine-tuning."""
import argparse,json,tempfile,time
from pathlib import Path
from types import SimpleNamespace
import train_act_execution_focused as run


def check(output):
    import numpy as np,torch
    from excavator_vlm.policies.act.inference import load_act
    torch.set_num_threads(4);started=time.time();output.parent.mkdir(parents=True,exist_ok=True)
    pred=torch.zeros(1,8,10,4,requires_grad=True);target=torch.ones_like(pred);pad=torch.zeros(1,8,10,dtype=torch.bool);ret=torch.zeros(1,8,dtype=torch.bool)
    loss,_=run.loss_terms(pred,target,pred.detach(),pad,ret,torch.ones(4),torch.zeros(4));loss.backward()
    torch.testing.assert_close(pred.grad[:,:,0],pred.grad[:,:,3]*20.,rtol=1e-6,atol=1e-7)
    assert (pred.grad[:,:,3:].abs()>0).all()
    pad[:]=True
    zero,_=run.loss_terms(pred,target,pred.detach(),pad,ret,torch.ones(4),torch.zeros(4));assert zero==0
    del pred,target,pad,ret,loss,zero

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
    with tempfile.TemporaryDirectory(prefix='act_execution_focus_check_') as temporary:
        run.train(SimpleNamespace(output=Path(temporary)/'train',steps=2,device='cpu',smoke=True))
        state=run.read(Path(temporary)/'train/status.json')
        assert state['strict_reload']=='passed' and state['native_rgb_reload']=='passed' and state['frozen_core_unchanged']
        assert state['steps']==2 and all(n.startswith('model.memory_adapter.') for n in state['changed_parameters'])
    proof=dict(status='passed',audit=audit,checks=['source frozen core identical to causal feature cache','all 160 immutable train/validation cache hashes verified',
        '2800 bridge windows span actual return-to-next-cycle boundary','full prefix and 128-step fragment predictions agree',
        'first three horizon steps have 20x expert gradient weight','later horizon remains supervised','all-padding loss zero','two real-data CPU optimization steps','strict memory and raw RGB native reload','only memory parameters changed','original external hashes preserved'],
        device='cpu',no_evaluation_gpu_interference=True,temporary_weights_and_metrics_removed=True,elapsed_seconds=time.time()-started,trainer_sha256=run.common.digest(Path(run.__file__)))
    run.common.write(output,proof);print(json.dumps(proof),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);check(p.parse_args().output)
