"""Run the existing PI05 trainer on a prepared Newton/SmolVLA split."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

BASE=Path(__file__).resolve().parents[1]
SUBMODULE=BASE/'ExcavatorVLA-vlm'
sys.path.insert(0,str(SUBMODULE))
from excavator_vlm.data.contract import file_hash


def read(path):return json.loads(Path(path).read_text())
def now():return datetime.now(timezone.utc).isoformat()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();out=args.output.resolve()
    request=read(out/'request.json');audit=read(out/'prepared/dataset_audit.json')
    if audit['status']!='passed' or audit['split_counts']!={'train':140,'validation':20,'test':40}:
        raise ValueError('Expected audited 200-parent preparation and the exact frozen SmolVLA split')
    if file_hash(out/'prepared/split_manifest.json')!=audit['split_sha256']:
        raise ValueError('Frozen split file changed')
    train=out/'train';config=out/'training_config.yaml'
    if train.exists() and not args.resume:
        raise FileExistsError('Refusing to overwrite training; explicitly use --resume')
    source_paths=[config,out/'prepared/split_manifest.json',out/'prepared/collection_backend/backend_provenance.json']
    source_paths += [p for folder in ('policies/pi05','data','common')
                     for p in (SUBMODULE/'excavator_vlm'/folder).rglob('*.py')]
    hashes={str(p):file_hash(p) for p in source_paths}
    state=dict(status='running',phase='starting',pid=os.getpid(),started_utc=now(),output=str(out),
        target_steps=20000,gpu_ids=[0,1],source_hashes=hashes,stages=[])
    stopping=False

    def stop(*_):
        nonlocal stopping
        stopping=True

    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)

    def publish():
        state['updated_utc']=now()
        log=train/'train.log'
        if log.exists():
            rows=log.read_text().splitlines()
            for row in reversed(rows):
                try:state['latest_training_update']=json.loads(row);break
                except json.JSONDecodeError:continue
        pointer=train/'last_checkpoint.json'
        try:state['latest_checkpoint']=read(pointer)
        except (FileNotFoundError,json.JSONDecodeError):pass
        state['free_disk_gib']=shutil.disk_usage(out).free/2**30
        temporary=out/'status.json.tmp';temporary.write_text(json.dumps(state,indent=2)+'\n')
        temporary.replace(out/'status.json')

    def check_sources_and_space():
        if any(file_hash(Path(path))!=digest for path,digest in hashes.items()):
            raise RuntimeError('Prepared configuration, split, evidence or PI05 training source changed')
        # Measured old checkpoint size is 4.85 GiB. Ten milestones, the latest
        # rolling state and the next atomic save peak below 59 GiB. Count the
        # already committed new-run bytes toward that budget on exact resume.
        existing=sum(p.stat().st_size for p in (train/'checkpoints').rglob('*') if p.is_file()) if train.exists() else 0
        if shutil.disk_usage(out).free+existing < 65*2**30:
            raise RuntimeError('Less than the 65 GiB total budget for ten complete checkpoints and rolling saves')

    env={**os.environ,'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','PYTHONDONTWRITEBYTECODE':'1',
        'TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1','NCCL_P2P_DISABLE':'1','NCCL_IB_DISABLE':'1',
        'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}
    command=[str(BASE/'.venv-pi05/bin/python'),'-B','-u',str(SUBMODULE/'scripts/train_pi05.py'),
             '--config',str(config)]

    def stage(name,extra):
        check_sources_and_space()
        item=dict(name=name,command=[*command,*extra],started_utc=now(),status='running')
        state['stages'].append(item);state['phase']=name
        with (out/'logs'/f'{name}.log').open('a') as log:
            p=subprocess.Popen(item['command'],cwd=BASE,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                state['child_pid']=p.pid;publish()
                while p.poll() is None:
                    if stopping:
                        os.killpg(p.pid,signal.SIGTERM)
                        try:p.wait(timeout=25)
                        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                        break
                    publish();time.sleep(10)
                p.wait()
            except BaseException:
                if p.poll() is None:
                    os.killpg(p.pid,signal.SIGTERM)
                    try:p.wait(timeout=25)
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                raise
        item.update(status='completed' if p.returncode==0 else 'stopped' if stopping else 'failed',
                    exit_code=p.returncode,finished_utc=now())
        publish()
        if stopping:raise InterruptedError('User requested stopping training')
        if p.returncode:raise RuntimeError(f'{name} exited with code {p.returncode}; inspect its log')

    publish()
    try:
        if not args.resume:
            stage('configuration_data_check',['--dry-run'])
            stage('two_update_ddp_check',['--stop-after','2'])
            checkpoint=read(train/'last_checkpoint.json')
            provenance=read(train/'provenance.json')
            if (checkpoint['step']!=2 or read(train/'reload_check.json')['status']!='passed'
                    or provenance['batch_contract']!={'world_size':2,'batch_size':1,'accumulation':4}
                    or provenance['dataset_fingerprint']!=audit['fingerprint']
                    or provenance['split_manifest']!=read(out/'prepared/split_manifest.json')
                    or read(train/'normalization_stats.json')!=read(out/'prepared/normalization_stats.json')):
                raise ValueError('Real DDP update/reload/split/statistics contract check failed')
            state['real_ddp_preflight']='passed';publish()
        stage('training_20000',['--resume',str(train)])
        latest=read(train/'last_checkpoint.json')
        if latest['step']!=20000 or read(train/'reload_check.json')['status']!='passed':
            raise ValueError('Trainer exited without a verified 20k checkpoint')
        for step in range(2000,20001,2000):
            folder=train/f'checkpoints/{step:06d}'
            if not (folder/'training_state.pt').is_file() or read(folder/'pretrained_model/checkpoint.json')['step']!=step:
                raise ValueError(f'Missing complete requested milestone {step}')
        state.update(status='completed',phase='completed',finished_utc=now(),final_checkpoint=latest['checkpoint'])
        request.update(status='completed',finished_utc=now())
        (out/'request.json').write_text(json.dumps(request,indent=2)+'\n')
        publish();(out/'completion.json').write_text(json.dumps(state,indent=2)+'\n')
    except BaseException as error:
        state.update(status='stopped_by_user' if stopping else 'failed',phase=state['phase'],
                     error=f'{type(error).__name__}: {error}',finished_utc=now())
        publish();raise


if __name__=='__main__':main()
