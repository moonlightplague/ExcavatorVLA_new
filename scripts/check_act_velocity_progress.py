"""Real-cache trainer checks; test checkpoints and metrics are temporary."""
import argparse,copy,importlib.util,json,tempfile
from pathlib import Path
from types import SimpleNamespace


def main(output):
    import torch,numpy as np
    spec=importlib.util.spec_from_file_location('progress',Path(__file__).with_name('train_act_velocity_progress.py'))
    run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
    from excavator_vlm.policies.act.inference import load_act
    torch.set_num_threads(4)
    new=run.construct(True);old,_,_=load_act(run.SOURCE,'cuda:0',original_act=True)
    with torch.no_grad():
        feature=torch.as_tensor(run.common.cpu_data(0)['features'][:20],device='cuda')[None]
        torch.testing.assert_close(new.model.memory_adapter(feature)[0],old.model.memory_adapter(feature)[0],rtol=0,atol=0)
        # Nonzero learned periodic branch must agree for streaming and prefix reconstruction.
        memory=new.model.memory_adapter
        memory.angle_projection.weight.normal_(std=.01)
        whole=memory(feature)[0];hidden=None;stream=[]
        for frame in feature.unbind(1):
            value,hidden=memory(frame[:,None],hidden);stream.append(value)
        torch.testing.assert_close(torch.cat(stream,1),whole,rtol=1e-4,atol=1e-5)
        hidden=memory.prefix_hidden(feature[:,:9],torch.tensor([9]))
        torch.testing.assert_close(memory(feature[:,9:],hidden)[0],whole[:,9:],rtol=1e-4,atol=1e-5)
    tiny=np.arange(6*1604,dtype=np.float32).reshape(6,1604)
    future,valid=run.future_delta(tiny)
    assert valid.tolist()==[True,True,False,False,False,False]
    np.testing.assert_equal(future[:2],tiny[4:6][:,run.DIMS]-tiny[:2][:,run.DIMS])
    _,same_goal=run.future_delta(tiny,segment_ids=np.array([0,0,0,0,0,1]))
    assert same_goal.tolist()==[True,False,False,False,False,False]
    original=run.common.metadata()
    def metadata():
        value=copy.deepcopy(original);value['split_manifest']=dict(train=[0],validation=[1],test=[]);return value
    run.common.metadata=metadata
    with tempfile.TemporaryDirectory(prefix='act_velocity_progress_check_') as temp:
        for variant in ['balanced','progress']:
            run.train(SimpleNamespace(output=Path(temp),variant=variant,balanced_steps=2,progress_steps=2))
            result=json.loads((Path(temp)/variant/'status.json').read_text())
            assert result['status']=='completed' and result['strict_reload']=='passed' and result['frozen_core_unchanged']
    proof=dict(status='passed',checks=['zero-angle-branch exactly preserves source cache predictions',
        'learned periodic projection streaming and prefix parity','future targets mask missing future and do not cross scoop/goal boundaries',
        'two-step real-cache GPU training of both variants','strict checkpoint reload','frozen ACT core unchanged'],
        temporary_weights_and_metrics_removed=True)
    output.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();main(args.output)
