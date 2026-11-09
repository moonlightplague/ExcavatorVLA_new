"""Run a fixed, source-pinned round of ACT spillage development comparisons."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_act_phw_diagnostics import snapshot, sha, read, write
from run_act_carry_guard import analyze
from act_transport_diagnostics import analyze_transport
import math

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variants', type=Path, required=True,
                        help='JSON mapping names to extra evaluator CLI arguments')
    parser.add_argument('--case-variants', type=Path,
                        help='Optional per-case mapping overriding the default compared variants.')
    parser.add_argument('--cases', default='seed4')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--steps', type=int, choices=[350, 500], default=350,
                        help='Diagnostic horizon, fixed identically across compared variants.')
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--transport', action='store_true',
                        help='Recorded pre-truck loss cases; variants explicitly opt into carry guard.')
    args = parser.parse_args()
    cases = args.cases.split(',')
    allowed = {'seed4', 'reference'} | ({'1000013', '1000020', '1000026', '1000045'} if args.transport else set())
    if not set(cases) <= allowed or not 1 <= args.repeats <= 3:
        parser.error('Use the two fixed development scenes, at most three repeats')
    variants = read(args.variants)
    if not variants or any(not isinstance(flags, list) for flags in variants.values()):
        parser.error('Expected named CLI argument lists')
    case_variants = read(args.case_variants) if args.case_variants else {}
    if (not set(case_variants) <= set(cases) or any(not values or
            any(not isinstance(flags,list) for flags in values.values()) for values in case_variants.values())):
        parser.error('Invalid per-case variants')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    checkpoint = ROOT / 'outputs/train/excavator_act_phw_20261028_150222_336724_1423224/checkpoints/040000/pretrained_model'
    protected = {str(p): sha(p) for p in checkpoint.iterdir() if p.is_file()}
    if protected[str(checkpoint / 'model.safetensors')] != '665554b8c4b48533f0606c5506e5ea8f12558ab76912beb80191043ae1c60924':
        raise ValueError('Expected original PHW40k weights')
    references = {
        'seed4': ROOT / 'outputs/experiments/act_reachable_target_prepare_20261103_01/reference_original_target_seed4.json',
        'reference': ROOT / 'outputs/experiments/act_phw_optimization_20261103_01/reference_setup.json'}
    source_episodes = {}
    for case in cases:
        if case.isdecimal():
            index = int(case)-1000000+1
            source = ROOT / f'outputs/experiments/act_phw_optimization_20261103_01/heldout_baseline/episodes/episode_{index:04d}'
            scene, initial = read(source/'scene_context.json'), read(source/'initial_observation.json')
            setup = read(references['seed4'])
            setup.update(profile=scene['scene_profile'], task=initial['task'],
                initial_joints_deg=[math.degrees(v) for v in initial['state'][:4]],
                expected_initial_state=initial['state'],
                dig_target_xyz=scene['observation_context']['dig_target_xyz'],
                sand_amount=scene['scene_profile']['sand_amount_multiplier'],
                actions=[json.loads(line)['predicted'] for line in (source/'steps.jsonl').read_text().splitlines()],
                provenance=dict(source_episode=str(source), scene_sha256=sha(source/'scene_context.json'),
                    purpose='recorded transport-loss development case; no expert actions are executed'))
            write(out/f'{case}.json', setup)
            source_episodes[case] = str(source)
    for case in cases:
        if case in references:
            (out / f'{case}.json').write_bytes(references[case].read_bytes())
    protocol = dict(scope='fixed-case development, not independent success-rate estimation',
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        variants=variants, case_variants=case_variants, cases=cases, repeats=args.repeats, checkpoint_files=protected,
        carry_settings=dict(pitch_deg=-130, parent='arm_first', near_edge_m=.75, opening_target_rad=-.7),
        physics='PhysX, walls moved before settling, fresh simulator, render every tick',
        timing=dict(fps=10, steps=args.steps, replan=3, post_rollout_observation_steps=30),
        original_success='100 initial-pile particles in truck and bucket >= -0.8 rad, three observations',
        reachability='Far-distance/unreachable cases are outside this spillage round; no old scores rewritten',
        tail='separate zero-command continuation; original terminal success, metrics and timing frozen')
    write(out / 'protocol.json', protocol)
    if args.transport:
        protocol.update(focus='Loss during loaded transportation before the truck; no opening optimization',
            reachability='Far/distant task noncompletion is pending, not counted as transport-policy failure. '
                         'Pre-truck sand loss and effective loaded motion are measured separately.',
            source_episodes=source_episodes,
            carry_settings='Explicit per variant; raw baseline has no carry supervisor')
        write(out/'protocol.json', protocol)
    snapshot(out)
    frozen = out / 'frozen_sources'
    jobs = []
    # Reverse GPU assignment on alternating repeats to avoid permanently
    # confounding a variant with a physical GPU.
    gpus = args.gpus.split(',')
    for case in cases:
        for repeat in range(args.repeats):
            for variant_index, (name, flags) in enumerate(case_variants.get(case, variants).items()):
                destination = out / f'{case}_{name}_r{repeat + 1}'
                gpu = gpus[(variant_index + repeat) % len(gpus)]
                command = [sys.executable, '-u', str(frozen / 'scripts/eval_smolvla_success.py'),
                    '--policy-type', 'act', '--checkpoint', str(checkpoint),
                    '--dataset', str(ROOT / 'data/export_phw'), '--output', str(destination),
                    '--episodes', '1', '--seed', case if case.isdecimal() else '4' if case == 'seed4' else '2',
                    '--steps', str(args.steps), '--min-particles', '100', '--replan-interval', '3',
                    '--reference-episode', str(out / f'{case}.json'), '--workers', '1',
                    '--gpus', gpu, '--cpu-threads', '8', '--headless', '--fresh-simulator',
                    '--render-every-physics-step', '--soil-backend', 'physx',
                    '--no-sand-walls-during-settle', '--video', '--no-save-trajectory',
                    '--post-rollout-observe-steps', '30']
                if not args.transport:
                    command += ['--act-carry-guard', '--act-carry-parent-projection', 'arm_first',
                        '--act-carry-release-geometry', 'opening_pose', '--act-carry-max-pitch-deg', '-130',
                        '--act-carry-opening-near-edge-m', '.75', '--act-carry-opening-target-rad', '-.7']
                command += flags
                jobs.append(dict(name=destination.name, case=case, variant=name, repeat=repeat + 1,
                                 gpu=gpu, command=command, output=str(destination)))
    write(out / 'jobs.json', jobs)
    state = dict(status='running', completed={}, running=[])
    write(out / 'status.json', state)

    def run(job):
        env = dict(os.environ, HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                   HF_HOME=str(ROOT / 'models/hf-cache'), TORCH_HOME=str(ROOT / 'models/torch-cache'),
                   OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        with (out / (job['name'] + '.log')).open('x') as log:
            result = subprocess.run(job['command'], cwd=frozen, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{job['name']} exited {result.returncode}; see its log")
        folder = Path(job['output']) / 'episodes/episode_0001'
        summary = analyze(folder)
        tail = read(folder / 'post_rollout_summary.json')
        if tail['status'] != 'completed':
            raise RuntimeError(f"{job['name']} post-rollout observations failed: {tail}")
        summary.update(post_rollout=tail, transport=analyze_transport(folder))
        return dict(**job, summary=summary)

    try:
        # No two evaluators share one physical GPU within a concurrent batch.
        pending = list(jobs)
        while pending:
            batch = []
            used = set()
            for job in pending:
                if job['gpu'] not in used:
                    batch.append(job); used.add(job['gpu'])
            state['running'] = [j['name'] for j in batch]
            write(out / 'status.json', state)
            with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                futures = [(job, executor.submit(run, job)) for job in batch]
                for job, future in futures:
                    value = future.result()
                    state['completed'][job['name']] = value
                    write(out / 'results.json', state['completed'])
                    write(out / 'status.json', state)
                    print(json.dumps(dict(name=job['name'], success=value['summary']['task_success'],
                        terminal=value['summary']['final_deposited'],
                        after_observation=value['summary']['post_rollout']['final']), ensure_ascii=False), flush=True)
                    pending.remove(job)
        if any(sha(Path(p)) != expected for p, expected in protected.items()):
            raise RuntimeError('Protected checkpoint input changed')
        manifest = read(out / 'source_snapshot.json')['files']
        if any(sha(frozen / p) != expected for p, expected in manifest.items()):
            raise RuntimeError('Frozen source changed')
        state.update(status='completed', running=[], inputs_unchanged=True)
    except BaseException as error:
        state.update(status='error', error=str(error))
        raise
    finally:
        write(out / 'status.json', state)


if __name__ == '__main__':
    main()
