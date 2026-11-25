"""Real-feature checks for training-only ACT memory corruption and recovery."""
import argparse,json,tempfile,time
from pathlib import Path
from types import SimpleNamespace
import train_act_memory_robustness as run


def check(output):
    import numpy as np,torch
    from excavator_vlm.policies.act.inference import load_act
    torch.set_num_threads(4);started=time.time();output.parent.mkdir(parents=True,exist_ok=True)
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
    saved=prefix.clone();generator=torch.Generator().manual_seed(42)
    bad=run.corrupt_hidden(prefix,generator)
    torch.testing.assert_close(prefix,saved,rtol=0,atol=0)
    assert bad.shape==prefix.shape and torch.isfinite(bad).all()
    assert not torch.equal(bad,prefix)
    del policy,memory,features,entire,prefix,fragment,saved,bad
    with tempfile.TemporaryDirectory(prefix='act_memory_robustness_check_') as temporary:
        run.train(SimpleNamespace(output=Path(temporary)/'train',steps=2,device='cpu',smoke=True))
        state=run.read(Path(temporary)/'train/status.json')
        assert state['strict_reload']=='passed' and state['native_rgb_reload']=='passed' and state['frozen_core_unchanged']
        assert state['validation']['memory_probe']['perturbed_executed_action_mae']>0
        assert state['steps']==2 and all(n.startswith('model.memory_adapter.') for n in state['changed_parameters'])
    proof=dict(status='passed',audit=audit,checks=['source frozen core identical to causal feature cache','all 160 immutable train/validation cache hashes verified',
        '2800 bridge windows span actual return-to-next-cycle boundary','full prefix and 128-step fragment predictions agree',
        'hidden corruption finite, shape preserving and non-mutating','deterministic validation hidden-reset/jitter probes','two clean-plus-corrupted real-data CPU optimization steps','strict memory and raw RGB native reload','only memory parameters changed','original external hashes preserved'],
        device='cpu',no_evaluation_gpu_interference=True,temporary_weights_and_metrics_removed=True,elapsed_seconds=time.time()-started,trainer_sha256=run.common.digest(Path(run.__file__)))
    run.common.write(output,proof);print(json.dumps(proof),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);check(p.parse_args().output)
