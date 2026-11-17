"""Own the gated 20k SmolVLA history run, retained saves and final offline diagnostics."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from train_smolvla import BASE, load_settings
from prepare_smolvla_history import inventory, write


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def plot_losses(train):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path=train/'losses.jsonl'
    if not path.exists():return
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    if not rows:return
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True,constrained_layout=True)
    x=[r['step'] for r in rows];y=[r['loss'] for r in rows]
    axes[0].plot(x,y,lw=.4,alpha=.3,label='Per-update rank-0 weighted flow loss')
    window=min(100,len(rows))
    import numpy as np
    axes[0].plot(x[window-1:],np.convolve(y,np.ones(window)/window,mode='valid'),lw=1,label=f'{window}-step mean')
    axes[0].set_ylabel('Training loss');axes[0].legend()
    axes[1].plot(x,[r['lr'] for r in rows]);axes[1].set_ylabel('Learning rate');axes[1].set_xlabel('Optimizer updates')
    fig.suptitle('SmolVLA causal 3 x 3 visual history — training, not task success')
    fig.savefig(train/'loss_curve.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=BASE/'training_config_smolvla_history.yaml')
    args=parser.parse_args()
    settings,launcher=load_settings(args.config)
    train=Path(settings['output_dir']);root=train.parent;prepared=Path(launcher['prepared_data'])
    # Check plotting imports before owning or launching a costly training process.
    plot_losses(train)
    status_path=root/'status.json'
    if status_path.exists() or train.exists():raise FileExistsError('This experiment is already owned; inspect status/resume explicitly')
    preflight=json.loads((root/'preflight.json').read_text())
    if preflight['status']!='passed' or preflight['reload_action_max_abs_error_rad_s']>1e-5:
        raise RuntimeError('Real-model preflight has not passed')
    if (preflight['training_config_sha256']!=digest(args.config.resolve()) or
            preflight['normalization_stats_sha256']!=digest(prepared/'normalization_stats.json')):
        raise RuntimeError('Training configuration/statistics differ from the passed GPU gate')
    if json.loads((root/'unit_tests.json').read_text())['status']!='passed':
        raise RuntimeError('Focused unit/regression tests have not passed')
    if settings['steps']!=20000 or settings['save_freq']!=2000 or settings['policy']['n_obs_steps']!=3:
        raise ValueError('Run must retain the requested 20k/2k/three-timestamp contract')
    source_paths=[args.config.resolve(),*[BASE/'scripts'/name for name in (
        'train_smolvla.py','smolvla_history.py','prepare_smolvla_history.py',
        'validate_smolvla_history.py','eval_smolvla_history_offline.py','run_smolvla_history_experiment.py')],
        BASE/'ExcavatorVLA-vlm/excavator_vlm/policies/pi05/history.py',BASE/'requirements-smolvla-history.txt']
    source_hashes={str(p):digest(p) for p in source_paths}
    state={'status':'running','phase':'checking_inputs','pid':os.getpid(),
        'created_utc':datetime.now(timezone.utc).isoformat(),'source_hashes':source_hashes,
        'training_steps_target':20000,'checkpoint_steps':list(range(2000,20001,2000)),
        'logical_batch_per_gpu':settings['batch_size'],'gpu_ids':launcher['gpu_ids'],
        'effective_global_batch':settings['batch_size']*len(launcher['gpu_ids']),
        'software':{name:version(name) for name in ('lerobot','torch','transformers','accelerate','safetensors','datasets','av','matplotlib','scipy')},
        'test_policy':'40 complete episodes sealed until the final 20k checkpoint; no checkpoint selection on test data'}
    (root/'logs').mkdir(exist_ok=True)
    process=None

    def save():
        state['updated_utc']=datetime.now(timezone.utc).isoformat()
        temporary=status_path.with_suffix('.tmp');write(temporary,state);temporary.replace(status_path)

    def check_sources():
        if any(digest(Path(path))!=expected for path,expected in source_hashes.items()):
            raise RuntimeError('Experiment source changed; preserve the run and inspect before resuming')

    def launch(label,command):
        nonlocal process
        with (root/'logs'/f'{label}.log').open('x') as log:
            process=subprocess.Popen(command,cwd=BASE,env={**os.environ,'HF_HUB_OFFLINE':'1',
                'HF_DATASETS_OFFLINE':'1','HF_HOME':launcher['hf_home'],'PYTHONDONTWRITEBYTECODE':'1',
                'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'},
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state.update(phase=label,child_pid=process.pid);save()
            plotted=0
            while process.poll() is None:
                progress=train/'progress.json'
                if progress.exists():
                    state['training_progress']=json.loads(progress.read_text())
                    step=state['training_progress'].get('step',0)
                    if step-plotted>=250:
                        try:
                            plot_losses(train)
                            state.pop('loss_plot_error',None)
                        except Exception as exc:
                            state['loss_plot_error']=f'{type(exc).__name__}: {exc}'
                            print('Loss plot failed; training and numeric logs continue: '+str(exc),flush=True)
                        plotted=step
                state['available_checkpoints']=[step for step in range(2000,20001,2000)
                    if (train/f'checkpoints/step_{step}/training_state/training_step.json').is_file()]
                save();time.sleep(10)
            code=process.returncode;process=None
            if code:raise RuntimeError(f'{label} exited with {code}; inspect logs/{label}.log')

    def interrupted(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted)
    try:
        save()
        report=json.loads((prepared/'dataset_audit.json').read_text())
        if inventory(Path(report['dataset']))!=json.loads((prepared/'source_inventory.json').read_text()):
            raise RuntimeError('Dataset changed since preparation')
        state['base_sha256']=digest(Path(settings['policy']['path'])/'model.safetensors');save()
        check_sources()
        sys.path.insert(0,str(BASE/'ExcavatorVLA-vlm/scripts'))
        from eval_pi05_checkpoints import gpu_idle
        while not all(gpu_idle(gpu) for gpu in launcher['gpu_ids']):
            state['phase']='waiting_for_idle_gpus';save();time.sleep(10)
        launch('training',[sys.executable,'-B','-u',str(BASE/'scripts/train_smolvla.py'),'--config',str(args.config.resolve())])
        plot_losses(train);check_sources()
        for step in state['checkpoint_steps']:
            folder=train/f'checkpoints/step_{step}'
            if json.loads((folder/'training_state/training_step.json').read_text())['step']!=step:
                raise RuntimeError(f'Checkpoint {step} step mismatch')
            for name in ('model.safetensors','config.json','observation_history.json','policy_preprocessor.json',
                         'policy_postprocessor.json','backbone_assets/config.json','backbone_assets/tokenizer_config.json'):
                if not (folder/'pretrained_model'/name).is_file():raise RuntimeError(f'Incomplete checkpoint {step}: {name}')
            for name in ('optimizer_state.safetensors','scheduler_state.json','rng_state.safetensors'):
                if not (folder/'training_state'/name).is_file():raise RuntimeError(f'Missing resume state {step}: {name}')
        final=train/'checkpoints/step_20000/pretrained_model'
        for split in ('validation','test'):
            launch(f'final_{split}',[sys.executable,'-B','-u',str(BASE/'scripts/eval_smolvla_history_offline.py'),
                '--checkpoint',str(final),'--prepared',str(prepared),'--split',split,
                '--output',str(root/f'final_{split}'),'--gpu','0'])
        baseline=BASE/'outputs/train/repeat/excavator_smolvla_20261101_100523_535020_1399282/checkpoints/last/pretrained_model'
        if baseline.is_dir():
            state['baseline_comparison_scope']='Existing 80k single-frame checkpoint, older training publication; same test observations but unmatched training data/budget. Not a causal estimate of history improvement.'
            launch('baseline_test',[sys.executable,'-B','-u',str(BASE/'scripts/eval_smolvla_history_offline.py'),
                '--checkpoint',str(baseline.resolve()),'--prepared',str(prepared),'--split','test',
                '--output',str(root/'baseline_test'),'--gpu','0'])
        if inventory(Path(report['dataset']))!=json.loads((prepared/'source_inventory.json').read_text()):
            raise RuntimeError('Source dataset changed during experiment')
        state.update(status='completed',phase='completed',final_optimizer_step=20000,
                     final_checkpoint=str(final),source_dataset_unchanged=True,
                     completed_utc=datetime.now(timezone.utc).isoformat());save()
        write(root/'completion.json',state)
        lines=['# SmolVLA causal visual history — completed experiment','',
            'Final optimizer step: **20000**. All ten requested checkpoints include full model, '
            'processors, history contract, local backbone/tokenizer assets and optimizer/scheduler/RNG state.','',
            'The source dataset is unchanged. These metrics evaluate recorded observations; they do not establish closed-loop excavation success.','',
            '[Training loss curve](train/loss_curve.png) · [Training losses](train/losses.jsonl)','',
            '| Checkpoint | Model and processors |','| --- | --- |']
        for step in state['checkpoint_steps']:
            lines.append(f'| {step} | [step_{step}](train/checkpoints/step_{step}/pretrained_model/) |')
        for label in ('final_validation','final_test','baseline_test'):
            path=root/label/'report.json'
            if not path.exists():continue
            result=json.loads(path.read_text())
            lines+=['',f'## {label} ({result["episodes"]} episodes)','',
                '| Phase | Decisions | First-action MAE (rad/s) | Chunk MAE (rad/s) | Direction accuracy |',
                '| --- | --- | --- | --- | --- |']
            for phase,metrics in result['phase_metrics'].items():
                lines.append(f'| {phase} | {metrics["decision_observations"]} | {metrics["first_action_mae_rad_s"]:.4f} | '
                    f'{metrics["chunk_mae_rad_s"]:.4f} | {metrics["direction_accuracy_abs_target_at_least_0_05"]:.3f} |')
        if state.get('baseline_comparison_scope'):lines+=['',state['baseline_comparison_scope']]
        (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    except BaseException as exc:
        state.update(status='failed',error=f'{type(exc).__name__}: {exc}');save()
        raise
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()


if __name__=='__main__':main()
