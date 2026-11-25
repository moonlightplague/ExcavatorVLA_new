"""Two-step real-data checks for both alternative trainers; temporary weights removed."""
import argparse,copy,importlib.util,json,os,tempfile
from pathlib import Path
from types import SimpleNamespace


def check(output):
    spec=importlib.util.spec_from_file_location('alternative',Path(__file__).with_name('run_act_training_alternatives.py'))
    run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
    original=run.metadata()
    def metadata():
        value=copy.deepcopy(original);value['split_manifest']=dict(train=[0],validation=[1],test=[]);return value
    run.metadata=metadata
    with tempfile.TemporaryDirectory(prefix='act_alternatives_preflight_') as temp:
        for kind in ('velocity','pose'):
            run.train(SimpleNamespace(output=Path(temp),kind=kind,velocity_steps=2,pose_steps=2))
            status=json.loads((Path(temp)/kind/'status.json').read_text())
            assert status['status']=='completed' and status['strict_reload']=='passed' and status['frozen_core_unchanged']
            if kind=='pose':
                # Exercise ordinary raw-observation ACT inference after checkpoint reload.
                from excavator_vlm.policies.act.inference import load_act
                from lerobot.datasets.lerobot_dataset import LeRobotDataset
                import torch
                model,pre,_=load_act(status['checkpoint'],'cuda:0',original_act=True)
                data=LeRobotDataset(repo_id='local/repeat',root=run.DATA,episodes=[0],video_backend='pyav')
                row=data[0];raw={k:row[k][None] for k in ('observation.state',*model.config.image_features)}
                with torch.no_grad():pred=model.predict_action_chunk(pre(raw))
                assert pred.shape==(1,10,4) and torch.isfinite(pred).all()
                model.reset();assert model.model._memory_hidden is None
    result=dict(status='passed',variants=['velocity','pose'],optimizer_steps_each=2,
        checks=['training-only pose range fit','causal full-prefix GPU optimization','finite losses',
                'unchanged frozen core','strict checkpoint reload','raw-observation pose inference','episode reset'],
        temporary_weights_and_metrics_removed=True)
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();check(args.output)
