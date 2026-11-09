"""Apply a user-requested episode cap to an already-running paired ACT test.

Execution processes and their original provenance are retained. The old orchestration
process is replaced without signalling its evaluation children. At the requested
boundary, only those evaluation process trees are stopped. Prefix summaries are
derived separately, with the original 100-episode protocol and user amendment linked.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

import psutil


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prefix_rows(rows, limit, complete=False):
    selected = [row for row in rows if 1 <= row['episode_index'] <= limit]
    indices = [row['episode_index'] for row in selected]
    if len(indices) != len(set(indices)):
        raise ValueError('Duplicate episode indices in the requested prefix')
    if any(row['status'] != 'completed' or row.get('error') for row in selected):
        raise ValueError('An episode inside the requested prefix failed infrastructure validation')
    if complete and set(indices) != set(range(1, limit + 1)):
        raise ValueError('Requested prefix has missing episodes')
    return sorted(selected, key=lambda row: row['episode_index'])


def pair_rows(baseline, candidate, limit):
    baseline, candidate = prefix_rows(baseline, limit, True), prefix_rows(candidate, limit, True)
    result = []
    for left, right in zip(baseline, candidate):
        if left['scene_seed'] != right['scene_seed']:
            raise ValueError('The capped comparison contains unpaired scene seeds')
        result.append({'scene_seed': left['scene_seed'], 'baseline_success': left['success'],
                       'candidate_success': right['success']})
    return result


def alive(process):
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def validate_evaluator(process, task):
    command = process.cmdline()
    expected = task['command']
    if command != expected or '--output' not in command:
        raise RuntimeError('Evaluation PID no longer matches the recorded command')
    if Path(command[command.index('--output') + 1]).resolve() != Path(task['output']).resolve():
        raise RuntimeError('Evaluation PID belongs to another output directory')


class CapMonitor:
    def __init__(self, output, limit):
        self.out = Path(output).resolve()
        self.limit = limit
        self.frozen = self.out / 'frozen_sources'
        self.guard = (self.out / 'cap_monitor.lock').open('a')
        fcntl.flock(self.guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.state = read(self.out / 'status.json')
        self.tasks = [t for t in self.state['tasks'] if t['name'].startswith('heldout_')]
        if (self.state['status'] != 'running' or len(self.tasks) != 2
                or any(t['status'] != 'running' or t['episodes'] <= limit for t in self.tasks)):
            raise ValueError('Expected two running held-out evaluations with a larger original target')
        self.processes = {t['name']: psutil.Process(t['pid']) for t in self.tasks}
        self.stopping = {}
        self.rows = {}
        self.taken_over = False
        self.old_controller = psutil.Process(self.state['pid'])
        expected_controller = str(self.frozen / 'scripts/run_act_optimization.py')
        if expected_controller not in self.old_controller.cmdline():
            raise RuntimeError('Controller PID does not match this ACT experiment')
        for task in self.tasks:
            validate_evaluator(self.processes[task['name']], task)

    def takeover(self):
        amendment = self.out / 'episode_cap_amendment.json'
        if amendment.exists():
            raise FileExistsError('An episode-cap amendment already exists')
        self.old_controller.send_signal(signal.SIGSTOP)
        try:
            for _ in range(100):
                if self.old_controller.status() == psutil.STATUS_STOPPED:
                    break
                time.sleep(.01)
            else:
                raise RuntimeError('Could not suspend the old orchestration process')
            self.state = read(self.out / 'status.json')
            self.tasks = [t for t in self.state['tasks'] if t['name'].startswith('heldout_')]
            for task in self.tasks:
                validate_evaluator(self.processes[task['name']], task)
            original = {'status': self.state, 'heldout_protocol': read(self.out / 'heldout_protocol.json'),
                        'report': (self.out / 'REPORT.md').read_text(),
                        'results': read(self.out / 'results.json')}
            write(self.out / 'before_episode_cap.json', original)
            amendment_data = {
                'schema_version': 'user_requested_act_episode_cap_v1', 'created_utc': now(),
                'user_instruction': '可以压缩一下吗，50场结束',
                'original_target_per_policy': self.tasks[0]['episodes'], 'target_per_policy': self.limit,
                'selection_rule': 'The same first N predeclared seed IDs for both policies; no outcome filtering',
                'changed_after_interim_results': True, 'reason': 'User-requested evaluation budget reduction',
                'old_controller_pid': self.old_controller.pid, 'monitor_pid': os.getpid(),
                'monitor_source_sha256': sha(__file__),
                'preserved_completed_counts': {}, 'source_evaluations': {}}
            for task in self.tasks:
                folder = Path(task['output'])
                count = sum(read(p)['status'] == 'completed'
                            for p in folder.glob('episodes/*/episode_status.json'))
                if count > self.limit:
                    raise ValueError('The requested cap has already been exceeded')
                amendment_data['preserved_completed_counts'][task['name']] = count
                amendment_data['source_evaluations'][task['name']] = {
                    'config_sha256': sha(folder / 'config.json'),
                    'provenance_sha256': sha(folder / 'provenance.json'),
                    'evaluation_id': read(folder / 'config.json')['evaluation_id']}
            write(amendment, amendment_data)
        except BaseException:
            self.old_controller.send_signal(signal.SIGCONT)
            raise
        # SIGKILL targets only the old orchestrator, so its finally block cannot
        # cancel either live evaluator. Both children own separate sessions.
        self.old_controller.kill()
        self.taken_over = True
        self.old_controller.wait(timeout=10)
        self.controller_lock = (self.out / 'controller.lock').open('a')
        fcntl.flock(self.controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.state.update(pid=os.getpid(), previous_controller_pid=self.old_controller.pid,
                          phase=f'heldout_paired_{self.limit}_scenes',
                          final_requested_episodes=self.limit, episode_cap_amendment=str(amendment))
        for task in self.tasks:
            task.update(original_requested_episodes=task['episodes'], episodes=self.limit,
                        stopping_rule='user_requested_fixed_prefix', original_command_retained=True)
        self.save()
        self.report()

    def save(self):
        self.state['updated_utc'] = now()
        write(self.out / 'status.json', self.state)

    def report(self):
        lines = ['# ACT PHW 优化实验', '', f'状态：{self.state["status"]}；阶段：{self.state["phase"]}', '',
                 f'用户要求将独立评估从每组 100 场缩减为 {self.limit} 场；两组使用原场景序列相同的前 {self.limit} 场。',
                 '缩减发生在查看部分结果之后，已记录 episode_cap_amendment.json。原始执行配置、来源标识和逐场数据保留。',
                 '模型、动作、物理设置和成功判据保持原实验配置。', '',
                 '| 实验 | 状态 | 已完成 | 成功 |', '| --- | --- | ---: | ---: |']
        for task in self.state['tasks']:
            if task['kind'] == 'evaluation':
                count = task.get('completed_episodes', task['episodes'] if task['status'] == 'completed' else 0)
                lines.append(f'| {task["name"]} | {task["status"]} | {count}/{task["episodes"]} | {task.get("successes", "—")} |')
        lines.extend(['', '开发和选择场景用于调参；最终配对结果使用缩减后的独立场景前缀。'])
        if self.state.get('conclusion'):
            lines.extend(['', self.state['conclusion']])
        (self.out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
        write(self.out / 'results.json', {k: v for k, v in self.state.items() if k not in (
            'source_checkpoint_files', 'generated_checkpoint_files', 'dataset_files', 'experiment_inputs')})

    def request_stop(self, task):
        process = self.processes[task['name']]
        validate_evaluator(process, task)
        descendants = process.children(recursive=True)
        self.stopping[task['name']] = (time.monotonic(), [process, *descendants])
        process.terminate()
        # Stop the policy worker promptly at the completed-episode boundary.
        # Its signal handler closes its own simulator and preserves the completed status.
        for child in descendants:
            try:
                if '--worker-index' in child.cmdline():
                    child.terminate()
            except psutil.NoSuchProcess:
                pass
        task.update(status='stopping_at_requested_limit', stop_requested_utc=now(),
                    stop_reason='user_requested_episode_cap_reached',
                    stopped_process_pids=[p.pid for p in [process, *descendants]])

    def check_stopped(self, task):
        started, processes = self.stopping[task['name']]
        living = [p for p in processes if alive(p)]
        if living and time.monotonic() - started > 40:
            for process in living:
                try:
                    process.kill()
                except psutil.NoSuchProcess:
                    pass
        return not living

    def finalize_case(self, task):
        sys.path[:0] = [str(self.frozen / 'scripts'), str(self.frozen / 'ExcavatorVLA-smolvla'),
                       str(self.frozen / 'ExcavatorVLA-smolvla/scripts/evaluation')]
        from audit_smolvla_success import audit_episode
        from summarize_isaacsim_success_rate import load_episode_rows, build_summary, write_outputs
        source = Path(task['output'])
        rows = prefix_rows(load_episode_rows(source / 'episodes'), self.limit, True)
        expected_id = read(source / 'config.json')['evaluation_id']
        if any(row['evaluation_id'] != expected_id for row in rows):
            raise ValueError('A retained episode has another source evaluation identity')
        if [row['scene_seed'] for row in rows] != list(range(task['seed'], task['seed'] + self.limit)):
            raise ValueError('The retained prefix differs from the predeclared seed sequence')
        audits = [audit_episode(source / f'episodes/episode_{row["episode_index"]:04d}',
                    read(source / f'episodes/episode_{row["episode_index"]:04d}/episode_status.json')) for row in rows]
        summary = build_summary(rows, self.limit)
        if not summary['official_result_valid']:
            raise ValueError('The amended cohort did not pass completeness/standard validation')
        summary.update(analysis_scope='user_requested_prefix_of_original_100_episode_execution',
                       original_requested_episodes=task['original_requested_episodes'],
                       episode_cap_amendment=str(self.out / 'episode_cap_amendment.json'),
                       source_evaluation_directory=str(source))
        destination = self.out / f'amended_heldout_{self.limit}' / task['name']
        write_outputs(destination, rows, summary)
        write(destination / 'trace_audit.json', {'status': 'passed', 'episodes': audits,
            'scope': f'First {self.limit} episodes under the user-amended protocol',
            'source_evaluation_id': expected_id})
        self.rows[task['name']] = rows
        task.update(status='completed', finished_utc=now(), completed_episodes=self.limit,
                    successes=summary['successes'], failure_modes=summary['failure_mode_counts'],
                    mean_payload=summary['metrics']['payload_particles']['mean'],
                    summary_directory=str(destination),
                    intentional_termination=True,
                    excluded_episode_directories=[str(p) for p in sorted((source / 'episodes').glob('episode_*'))
                        if int(p.name.split('_')[-1]) > self.limit])

    def run(self):
        self.takeover()
        last_save = 0.
        while any(task['status'] != 'completed' for task in self.tasks):
            for task in self.tasks:
                if task['status'] == 'completed':
                    continue
                statuses = []
                for path in Path(task['output']).glob('episodes/*/episode_status.json'):
                    index = int(path.parent.name.split('_')[-1])
                    if index <= self.limit:
                        record = read(path)
                        record['episode_index'] = index
                        statuses.append(record)
                rows = prefix_rows(statuses, self.limit)
                task.update(completed_episodes=len(rows), successes=sum(r['success'] for r in rows))
                task['progress'] = [read(p) for p in sorted(Path(task['output']).glob('workers/*/progress.json'))]
                if len(rows) == self.limit and task['name'] not in self.stopping:
                    self.request_stop(task)
                elif task['name'] not in self.stopping and not alive(self.processes[task['name']]):
                    raise RuntimeError(f'{task["name"]} exited before completing its amended target')
                if task['name'] in self.stopping and self.check_stopped(task):
                    self.finalize_case(task)
            if time.monotonic() - last_save >= 5:
                self.save()
                self.report()
                last_save = time.monotonic()
            time.sleep(.2)
        from run_act_optimization import Runner
        verifier = object.__new__(Runner)
        verifier.out, verifier.frozen, verifier.state = self.out, self.frozen, self.state
        verifier.verify()
        baseline = next(t for t in self.tasks if t['name'] == 'heldout_baseline')
        candidate = next(t for t in self.tasks if t['name'] != 'heldout_baseline')
        pairs = pair_rows(self.rows[baseline['name']], self.rows[candidate['name']], self.limit)
        candidate_only = sum(r['candidate_success'] and not r['baseline_success'] for r in pairs)
        baseline_only = sum(r['baseline_success'] and not r['candidate_success'] for r in pairs)
        n = baseline_only + candidate_only
        pvalue = min(1., 2 * sum(math.comb(n, k) for k in range(min(candidate_only, baseline_only) + 1)) / 2**n) if n else 1.
        write(self.out / 'heldout_comparison.json', {
            'baseline': read(Path(baseline['summary_directory']) / 'success_rate_summary.json'),
            'candidate': read(Path(candidate['summary_directory']) / 'success_rate_summary.json'),
            'selection_winner': self.state['selected_winner'], 'seed': baseline['seed'],
            'episodes_per_policy': self.limit, 'paired_results': pairs,
            'candidate_only_successes': candidate_only, 'baseline_only_successes': baseline_only,
            'exploratory_exact_mcnemar_p': pvalue,
            'protocol_amendment': read(self.out / 'episode_cap_amendment.json')})
        self.state.update(status='completed', phase='completed', completed_utc=now(),
            conclusion=f'用户缩减后的 {self.limit} 场独立配对评估：原 PHW {baseline["successes"]}/{self.limit}；'
                       f'时间集成 {candidate["successes"]}/{self.limit}。原始配置和数据保留；100 场计划已按用户要求结束。')
        self.save()
        self.report()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=50)
    args = parser.parse_args()
    if not 1 <= args.limit < 100:
        parser.error('Limit must be between 1 and 99')
    monitor = CapMonitor(args.output, args.limit)
    try:
        monitor.run()
    except BaseException as exc:
        if monitor.taken_over:
            # Fail closed: a broken cap monitor must not leave its adopted jobs
            # running toward the original 100-episode target without a supervisor.
            processes = []
            for process in monitor.processes.values():
                if alive(process):
                    processes.extend(process.children(recursive=True))
                    processes.append(process)
            for process in reversed(processes):
                try:
                    process.terminate()
                except psutil.NoSuchProcess:
                    pass
            _, living = psutil.wait_procs(processes, timeout=20)
            for process in living:
                try:
                    process.kill()
                except psutil.NoSuchProcess:
                    pass
        monitor.state.update(status='cap_monitor_failed', error=f'{type(exc).__name__}: {exc}')
        monitor.save()
        monitor.report()
        raise


if __name__ == '__main__':
    main()
