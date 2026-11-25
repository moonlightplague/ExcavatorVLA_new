"""Real-checkpoint and raw-image integration checks, without rollout edits."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import torch


def check():
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location('experiment', Path(__file__).with_name('run_act_persistent_experiment.py'))
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    from excavator_vlm.policies.act.inference import load_act
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.temporal_extension import HISTORY_MASK
    new, pre, _ = run.model()
    old, _, _ = load_act(run.SOURCE, 'cuda:0', original_act=True)
    dataset = LeRobotDataset(repo_id='local/repeat', root=run.DATA, episodes=[0], video_backend='pyav')
    maximum = 0.
    raw = []
    with torch.no_grad():
        for frame in (0, 3, 6, 9, 12, 15):
            row = dataset[frame]
            batch = pre({k:row[k][None] for k in ('observation.state', *new.config.image_features)})
            a = new.predict_action_chunk(batch)
            b = old.predict_action_chunk(batch)
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            maximum = max(maximum,(a-b).abs().max().item())
            raw.append(batch)
        assert new.model._memory_hidden is not None
        new.reset()
        assert new.model._memory_hidden is None and new.model.history_size == 0
        first = new.predict_action_chunk(raw[0])
        old.reset()
        torch.testing.assert_close(first,old.predict_action_chunk(raw[0]),rtol=0,atol=0)
        batch = raw[0]
        spatial = [new.model.backbone(batch[k])['feature_map'] for k in new.config.image_features]
        state = batch['observation.state'][:,None].expand(-1,3,-1)
        pooled = torch.stack([x.mean((-2,-1)) for x in spatial],1)[:,None].expand(-1,3,-1,-1)
        features, cached = new.model.features_from_spatial(state, pooled,
                           torch.tensor([[True,True,False]],device='cuda'), spatial)
        torch.testing.assert_close(cached, first, rtol=1e-5,atol=1e-6)
        # Cache construction uses larger batches than online inference.
        batched = {k: torch.cat([raw[i % len(raw)][k] for i in range(16)])
                   for k in ('observation.state', *new.config.image_features)}
        maps = [new.model.backbone(batched[k])['feature_map'] for k in new.config.image_features]
        pools = torch.stack([value.mean((-2,-1)) for value in maps],1)
        _, batched_base = new.model.features_from_spatial(
            batched['observation.state'][:,None].expand(-1,3,-1),
            pools[:,None].expand(-1,3,-1,-1),
            torch.tensor([[True,True,False]],device='cuda').expand(16,-1), maps)
        torch.testing.assert_close(batched_base[:1], first, rtol=1e-4,atol=2e-5)
        batch_parity_error = (batched_base[:1]-first).abs().max().item()
        # Check our sequential RGB decoder against the actual LeRobot data loader.
        import pyarrow.parquet as pq
        metadata = pq.read_table(run.DATA/'meta/episodes/chunk-000/file-000.parquet').to_pylist()[0]
        import numpy as np
        for camera in new.config.image_features:
            path = run.DATA/f'videos/{camera}/chunk-000/file-000.mp4'
            decoded = list(run.video_frames(path, [0,3,6], metadata[f'videos/{camera}/from_timestamp'],10))
            for frame, image in zip((0,3,6), decoded):
                torch.testing.assert_close(torch.from_numpy(image).float()/255, dataset[frame][camera], rtol=0,atol=0)
    new.model.memory_adapter.train()
    optimizer = torch.optim.AdamW(new.model.memory_adapter.parameters(),lr=1e-3)
    # Two actual updates ensure gradients propagate past the zero initialized head.
    for _ in range(2):
        optimizer.zero_grad()
        delta = new.model.memory_adapter(features.detach()[:,None].repeat(1,4,1))[0]
        loss = (delta-.1).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(new.model.memory_adapter.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
    assert new.model.memory_adapter.gru.weight_ih_l0.grad.abs().sum()>0
    for key,value in old.state_dict().items():
        if not torch.equal(value,new.state_dict()[key]):
            raise ValueError('Source core changed: '+key)
    new.eval()
    new.reset()
    with torch.no_grad():
        expected = [new.predict_action_chunk(item) for item in raw]
        with tempfile.TemporaryDirectory() as folder:
            new.save_pretrained(folder)
            for p in run.SOURCE.glob('policy_*'):
                __import__('shutil').copy2(p,Path(folder)/p.name)
            restored,_,_ = load_act(folder,'cuda:0',original_act=True)
            assert restored.model._memory_hidden is None
            actual = [restored.predict_action_chunk(item) for item in raw]
        for a,b in zip(actual,expected):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
    return dict(status='passed',zero_adapter_max_difference=maximum, batch16_max_difference=batch_parity_error, tests=[
        'six-call raw-image source equivalence','clears short and persistent memory on reset',
        'cached spatial features match raw pixels','batch16 cache matches singleton inference','sequential decoder equals LeRobot RGB exactly',
        'finite optimization and GRU gradients','frozen source weights bitwise unchanged',
        'strict checkpoint reload reproduces six-call stream'],peak_gpu_bytes=torch.cuda.max_memory_allocated())


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); result=check();args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
