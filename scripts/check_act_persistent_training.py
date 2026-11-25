"""Exercise the full trainer on two cached training episodes; discard test weights."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace


def check(cache_root, output):
    spec = importlib.util.spec_from_file_location('experiment', Path(__file__).with_name('run_act_persistent_experiment.py'))
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    original = run.provenance()
    available = [episode for episode in original['split_manifest']['train']
                 if (cache_root/'features'/f'episode_{episode:04d}.npz').is_file()]
    if len(available)<2:
        raise RuntimeError('Wait for two complete training-episode feature caches')
    chosen = available[:2]
    def test_provenance():
        metadata = copy.deepcopy(original)
        metadata['split_manifest'] = dict(train=chosen[:1],validation=chosen[1:],test=[])
        return metadata
    run.provenance = test_provenance
    with tempfile.TemporaryDirectory(prefix='act_memory_trainer_check_') as temp:
        folder=Path(temp)
        (folder/'features').mkdir()
        for episode in chosen:
            name=f'episode_{episode:04d}.npz'
            (folder/'features'/name).symlink_to((cache_root/'features'/name).resolve())
        (folder/'external_files.json').write_text((cache_root/'external_files.json').read_text())
        run.train(SimpleNamespace(output=folder,steps=2))
        status=json.loads((folder/'status.json').read_text())
        assert status['status']=='completed' and status['frozen_core_bitwise_unchanged']
        assert status['strict_reload']=='passed' and status['external_files_unchanged']
    result=dict(status='passed',optimizer_steps=2,training_episode=chosen[0],
                temporary_validation_episode=chosen[1],both_episodes_from_training_split=True,
                checks=['cached real features','full episode prefix reconstruction','finite GPU optimization',
                        'ordered validation','unchanged frozen core','final checkpoint save and strict reload'],
                temporary_weights_and_metrics_removed=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    check(args.cache_root,args.output)
