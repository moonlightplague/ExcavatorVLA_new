"""Verify or rerun the preserved 11/40 ACT plus carry-guard baseline."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / 'docs/baselines/act_phw40k_carry275_20261105.json'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--output', type=Path,
                        help='A new directory; print the original protocol with this output path')
    parser.add_argument('--run', action='store_true', help='Actually run the printed evaluation')
    parser.add_argument('--verify-videos', action='store_true')
    args = parser.parse_args()
    if args.run and args.output is None:
        parser.error('--run requires a new --output directory')
    manifest = json.loads(args.manifest.read_text())
    if manifest['schema'] != 'act_preserved_baseline_v1':
        raise ValueError('Unknown baseline manifest schema')
    archive = (ROOT / manifest['archive']).resolve()
    for name, expected in manifest['files_sha256'].items():
        path = (archive / name).resolve()
        if not path.is_relative_to(archive) or digest(path) != expected:
            raise ValueError(f'Preserved baseline file differs: {name}')
    # The archived code and weights are self-contained. The dataset and Isaac
    # installation remain explicitly identified external dependencies.
    original = json.loads((archive / 'evidence/protocol.json').read_text())
    dataset = Path(manifest['external_dependencies']['dataset']).resolve()
    for name, expected in original['protected_files'].items():
        path = Path(name).resolve()
        if path.is_relative_to(dataset) and digest(path) != expected:
            raise ValueError(f'Baseline dataset input differs: {name}')
    isaac = archive / 'frozen_sources/isaacsim'
    if not isaac.is_dir() or str(isaac.resolve()) != manifest['external_dependencies']['isaacsim']:
        raise ValueError('Baseline Isaac runtime link differs or is unavailable')
    if args.verify_videos:
        for name, expected in manifest['retained_video_files_sha256'].items():
            if digest(ROOT / name) != expected:
                raise ValueError(f'Preserved evaluation video differs: {name}')
    print(json.dumps(dict(status='verified', baseline_id=manifest['baseline_id'],
                          files=len(manifest['files_sha256']),
                          result=manifest['baseline_result']), ensure_ascii=False), flush=True)
    if args.output is None:
        return
    output = args.output.resolve()
    original_output = (ROOT / manifest['original_experiment']).resolve()
    if output.exists() or output.is_relative_to(archive) or output.is_relative_to(original_output):
        raise ValueError('Use a new output directory outside preserved baseline/evaluation artifacts')
    command = [str(output) if part == '{output}' else part for part in manifest['command']]
    env = dict(os.environ, HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
               HF_HOME=str(ROOT / 'models/hf-cache'), TORCH_HOME=str(ROOT / 'models/torch-cache'),
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               PYTHONPATH=os.pathsep.join(filter(None, [str(archive / 'runtime_dependencies'),
                                                        os.environ.get('PYTHONPATH', '')])))
    print('PYTHONPATH=' + shlex.quote(env['PYTHONPATH']) + ' ' + shlex.join(command), flush=True)
    if args.run:
        subprocess.run(command, cwd=archive / 'frozen_sources', env=env, check=True)


if __name__ == '__main__':
    main()
