"""Prepare PI05 on the unchanged Newton publication and literal SmolVLA split."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BASE), str(BASE/'ExcavatorVLA-vlm')]
from excavator_vlm.data.contract import audit_dataset, file_hash, validate_split
from excavator_vlm.data.collection_evidence import audit_external_backend_provenance
from excavator_vlm.policies.pi05.dataset import training_stats


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def prepare(dataset, split_source, output):
    dataset, split_source, output = Path(dataset).resolve(), Path(split_source).resolve(), Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    if (output/'dataset_audit.json').exists():
        raise FileExistsError('This preparation is immutable; use a new experiment output')
    split = read(split_source)
    if {k:len(split[k]) for k in ('train','validation','test')} != {'train':140,'validation':20,'test':40}:
        raise ValueError('Expected the completed SmolVLA 140/20/40 split')
    report, frames = audit_dataset(dataset,10)
    validate_split(split,frames.episode_index.unique(),report['fingerprint'])
    if report['episodes']!=200 or report['fingerprint']!=split['dataset_fingerprint']:
        raise ValueError('The source publication differs from the frozen SmolVLA dataset')
    shutil.copy2(split_source,output/'split_manifest.json')
    if file_hash(split_source)!=file_hash(output/'split_manifest.json'):
        raise ValueError('The copied split changed')
    print('Source audit passed; frozen train/validation/test lists copied byte for byte.',flush=True)
    receipts={}
    for path in (dataset.parent/'raw').glob('run_*/episode_*/export.json'):
        receipt=read(path)
        if not receipt.get('validated') or Path(receipt['export_dir']).resolve()!=dataset:
            continue
        eid=receipt['export_episode_index']
        if eid in receipts:raise ValueError(f'Duplicate export receipt for episode {eid}')
        receipts[eid]=path
    if set(receipts)!=set(range(200)):
        raise ValueError('A validated source receipt is required for every one of the 200 parents')
    proof=output/'collection_backend';proof.mkdir()
    entries=[]
    for eid,path in sorted(receipts.items()):
        receipt,meta=read(path),read(path.parent/'meta.json')
        baseline=meta.get('repeat_baseline',{})
        if (baseline.get('soil_backend')!='newton_mpm' or meta.get('success') is not True
                or meta.get('scoops_completed')!=10 or meta.get('repeat_returns_completed')!=10
                or len(meta.get('repeat_cycles',[]))!=10
                or receipt['episode_id']!=meta['episode_id']):
            raise ValueError(f'Parent {eid} is not a verified complete Newton ten-cycle source')
        dest=proof/f'episodes/episode_{eid:04d}';dest.mkdir(parents=True)
        for name in ['meta.json','export.json']:
            shutil.copy2(path.parent/name,dest/name)
        entries.append(dict(episode_index=eid,parent_id=meta['episode_id'],soil_backend='newton_mpm',
            source_metadata=str(path.parent/'meta.json'),source_receipt=str(path),
            evidence=dict(path=str((dest/'meta.json').relative_to(proof)),sha256=file_hash(dest/'meta.json'),
                          field='repeat_baseline.soil_backend'),
            publication_receipt=dict(path=str((dest/'export.json').relative_to(proof)),sha256=file_hash(dest/'export.json'))))
    write(proof/'backend_provenance.json',dict(schema_version='excavator_backend_provenance_v1',
        soil_backend='newton_mpm',dataset_root=str(dataset),dataset_fingerprint=report['fingerprint'],episodes=entries))
    backend=audit_external_backend_provenance(dataset,report['fingerprint'],proof,range(200),'newton_mpm')
    stats=training_stats(frames,split['train'])
    old_stats=read(split_source.parent/'normalization_stats.json')
    for feature in stats:
        for key in stats[feature]:
            if not np.allclose(stats[feature][key],old_stats[feature][key],rtol=0,atol=1e-12):
                raise ValueError(f'Train-only statistics differ from the SmolVLA source subset: {feature}/{key}')
    write(output/'normalization_stats.json',stats)
    inventory=read(split_source.parent/'source_inventory.json')
    for name,entry in inventory.items():
        path=dataset/name
        if not path.is_file() or path.stat().st_size!=entry['bytes'] or file_hash(path)!=entry['sha256']:
            raise ValueError(f'Source bytes differ from SmolVLA preparation: {name}')
    report.update(collection_backend=backend,source_dataset=str(dataset),
        split_source=str(split_source),split_sha256=file_hash(split_source),
        split_counts={k:len(split[k]) for k in ('train','validation','test')},
        split_frame_counts={k:int(frames.episode_index.isin(split[k]).sum()) for k in ('train','validation','test')},
        source_inventory_sha256=file_hash(split_source.parent/'source_inventory.json'),
        all_source_file_hashes_verified=len(inventory),source_publication_modified=False,
        normalization='PI05 quantiles fitted only to the same 140 train parents; held-out frames excluded',
        action_labels='Original SmolVLA source labels, unchanged; no controller relabel or bucket20 profile')
    write(output/'dataset_audit.json',report)
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=BASE/'data/repeat/export_repeat')
    parser.add_argument('--split-source',type=Path,default=BASE/'outputs/experiments/smolvla_history_repeat20k_20261116_01/prepared/split_manifest.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();prepare(args.dataset,args.split_source,args.output)
