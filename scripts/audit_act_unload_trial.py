"""Check completed unloading trials against traces and preserved input hashes."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def advance(consecutive,row):
    count=row['particle_transfer_metrics']['payload_particles']
    return consecutive+1 if count>=100 and row['state'][3]>=-.8 else 0


def audit(out,root):
    protocol=read(out/'protocol.json');summary=read(out/'results.json')
    assert read(out/'status.json')['status']=='completed'
    expected={(s,r) for s in protocol['scenes'] for r in range(1,protocol['repeats']+1)}
    actual={(r['scene'],r['repetition']) for r in summary['results'].values()}
    assert actual==expected,'Incomplete or unplanned cases'
    protected=read(out/'inputs_sha256.json')
    for name,value in protected.items():
        assert sha(Path(name))==value,f'Changed trial input: {name}'
    analysis_manifest=out/'analysis_sources/sources_sha256.json'
    if not analysis_manifest.exists():analysis_manifest=out/'analysis_sources/sha256.json'
    analyses=read(analysis_manifest)
    for name,value in analyses.items():
        path=Path(name)
        if not path.is_absolute() and not path.is_file():path=out/'analysis_sources'/path
        assert sha(path)==value,f'Changed analysis source: {name}'
    prefix_rows=0;branch_rows=0;restores=0;max_landmark_error=0.;case_records=[]
    for name,result in sorted(summary['results'].items()):
        folder=out/name
        prefix=[json.loads(line) for line in (folder/'prefix_steps.jsonl').read_text().splitlines()]
        prefix_rows+=len(prefix);consecutive=0
        for row in prefix:consecutive=advance(consecutive,row)
        if result['status']=='not_reached':
            assert len(prefix)==result['prefix_steps']
            case_records.append(dict(case=name,status='not_reached',prefix_steps=len(prefix)))
            continue
        assert result['status']=='completed'
        assert len(prefix)==result['prefix_steps']
        snapshot=read(folder/'snapshot.json')
        assert sha(folder/'physical_snapshot.npz')==snapshot['physical_file_sha256']
        for branch,endpoint in result['branches'].items():
            branch_folder=folder/branch
            assert (branch_folder/'evaluation.mp4').stat().st_size>1000
            restore=read(branch_folder/'restore_audit.json');details=restore['audit'];restores+=1
            assert restore['initial_observation_equal'] and restore['first_camera_sha256']==snapshot['camera_sha256']
            for key in ('particle_positions_sha256','particle_velocities_sha256'):
                assert details[key]==snapshot[key]
            for key,value in details['max_abs_errors'].items():
                assert value<=details['tolerance'][key]
            landmark=max(details['bucket_landmark_max_abs_errors_m'].values())
            max_landmark_error=max(max_landmark_error,landmark)
            assert landmark<=1e-4
            rows=[json.loads(line) for line in (branch_folder/'steps.jsonl').read_text().splitlines()]
            branch_rows+=len(rows)
            assert len(rows)==endpoint['branch_steps']==min(protocol['branch_steps'],500-len(prefix))
            first=None;counter=consecutive
            for index,row in enumerate(rows):
                assert row['step']==len(prefix)+index and row['step']<500
                assert len(row['state'])==28 and len(row['applied'])==4
                assert all(math.isfinite(v) for v in row['state']+row['applied'])
                assert all(abs(v)<=limit+1e-5 for v,limit in zip(row['applied'],(6,2.5,2.5,6)))
                cohort=row['unload_cohort_metrics']
                assert sum(cohort[k] for k in ('truck','bucket','outside'))==cohort['total']==result['cohort_count']
                counter=advance(counter,row)
                if counter>=3 and first is None:first=row['step']
            assert (first is not None)==endpoint['original_task_completed']
            assert first==endpoint['first_success_step']
            final=rows[-1]['unload_cohort_metrics']
            assert final==endpoint['cohort_terminal']
            for key,metric in (('truck','truck_fraction'),('bucket','retained_fraction'),('outside','outside_fraction')):
                assert final[key]/final['total']==endpoint[metric]
            case_records.append(dict(case=name,branch=branch,steps=len(rows),success=first is not None,
                final_truck_particles=rows[-1]['particle_transfer_metrics']['payload_particles']))
    baseline=read(root/'docs/baselines/act_phw40k_carry275_20261105.json')
    baseline_files=baseline['files_sha256']
    for name,value in baseline_files.items():
        assert sha(root/baseline['archive']/name)==value,f'Changed preserved baseline: {name}'
    combined=root/'outputs/experiments/act_combined_random40_20261106_01'
    combined_evidence=read(combined/'evidence_sha256.json')
    for name,value in combined_evidence.items():
        path=Path(name)
        if not path.is_absolute():path=combined/path
        assert sha(path)==value,f'Changed original random40 evidence: {path}'
    result=dict(passed=True,planned_cases=len(expected),matched_pairs=summary['matched_pairs'],
        not_reached=summary['not_reached'],protected_trial_files=len(protected),analysis_sources=len(analyses),
        prefix_observations=prefix_rows,branch_observations=branch_rows,restores=restores,
        maximum_landmark_restore_error_m=max_landmark_error,
        preserved_baseline_files_unchanged=len(baseline_files),original_random40_evidence_unchanged=len(combined_evidence),
        advance_gate_passed=summary['advance'],baseline_replaced=False,records=case_records)
    (out/'postflight_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    evidence={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*'))
        if p.is_file() and p.name!='evidence_sha256.json' and '__pycache__' not in p.parts}
    (out/'evidence_sha256.json').write_text(json.dumps(evidence,indent=2)+'\n')
    return {k:v for k,v in result.items() if k!='records'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    print(json.dumps(audit(args.output.resolve(),args.root.resolve()),ensure_ascii=False))
