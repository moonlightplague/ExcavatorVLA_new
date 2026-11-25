"""Exercise real RGB alignment, native ACT parity and both gradient paths."""
import argparse,gc,json,tempfile,time
from pathlib import Path
from types import SimpleNamespace
import train_act_visual_robustness as run


def check(output):
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    torch.set_num_threads(4)
    started=time.time();v=run.helper();reader=v.RGBEpisodes(run.common.DATA)
    ep=reader.load(0,6)
    dataset=LeRobotDataset(repo_id='local/repeat',root=run.common.DATA,episodes=[0],video_backend='pyav')
    policy,pre,_=run.load_policy();policy.reset();sequence=v.VisualSequence(policy.model)
    maximum=0.
    with torch.no_grad():
        for i in range(6):
            raw,_,_,_=v.batch_at([ep],i);native=dataset[i*3]
            for key in policy.config.image_features:torch.testing.assert_close(raw[key][0],native[key],rtol=0,atol=0)
            torch.testing.assert_close(raw['observation.state'][0],native['observation.state'],rtol=0,atol=0)
            torch.testing.assert_close(raw['action'][0,0],native['action'],rtol=0,atol=0)
            batch=pre(raw);actual=sequence.step(batch);expected=policy.predict_action_chunk(dict(batch))
            maximum=max(maximum,float((actual-expected).abs().max()))
            torch.testing.assert_close(actual,expected,rtol=1e-5,atol=2e-5)
    # Loss masking must exclude every padded observation in a finished lane.
    short={**ep,'frames':ep['frames'][:1]}
    _,pad,groups,valid=v.batch_at([short,ep],2)
    assert not valid[0] and pad[0].all() and not groups[0].any() and valid[1]
    names=v.enable_visual_gradients(policy)
    assert not policy.model.training and all(not p.requires_grad for p in policy.model.memory_adapter.parameters())
    raw,_,_,_=v.batch_at([ep],0);seq=v.VisualSequence(policy.model,gradients=True)
    loss=seq.step(pre(raw)).square().mean();loss.backward()
    gradient_groups={prefix:sum(float(p.grad.abs().sum()) for name,p in policy.named_parameters() if name.startswith(prefix) and p.grad is not None)
        for prefix in ('model.backbone.layer4.','model.decoder.','model.action_head.')}
    assert all(value>0 for value in gradient_groups.values()),gradient_groups
    assert all(p.grad is None for n,p in policy.named_parameters() if n not in names)
    del seq,sequence,policy,pre,dataset,reader,ep;gc.collect();torch.cuda.empty_cache()
    results={}
    with tempfile.TemporaryDirectory(prefix='act_visual_check_') as temporary:
        for variant in ('control','robust'):
            run.train(SimpleNamespace(output=Path(temporary),variant=variant,epochs=1,lanes=2,smoke=True))
            status=run.read(Path(temporary)/variant/'status.json')
            assert status['status']=='completed' and status['memory_and_other_frozen_weights_unchanged']
            results[variant]=dict(optimizer_updates=status['optimizer_updates'],strict_online_reload=status['strict_online_reload'])
            gc.collect();torch.cuda.empty_cache()
    proof=dict(status='passed',native_online_max_difference=maximum,gradient_magnitudes=gradient_groups,variants=results,
        checks=['six real RGB/state/action timestamps identical to LeRobot pyav reader','native persistent online action parity',
            'finished lane padding excluded','only visual layer4, decoder and action head receive gradients',
            'both variants optimize real RGB and reload strictly','all memory and other frozen weights unchanged'],
        temporary_weights_and_metrics_removed=True,elapsed_seconds=time.time()-started,
        trainer_sha256=run.common.digest(Path(run.__file__)),helper_sha256=run.common.digest(Path(v.__file__)))
    output.parent.mkdir(parents=True,exist_ok=True);run.common.write(output,proof);print(json.dumps(proof),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);check(p.parse_args().output)
