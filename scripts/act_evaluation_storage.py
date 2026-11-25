"""Release only regenerable shader caches from completed, audited ACT evaluations."""
import shutil
from pathlib import Path
from eval_smolvla_repeat40 import read,atomic_json,now


def require_space(path,gib=12):
    free=shutil.disk_usage(Path(path)).free
    if free<gib*2**30:raise RuntimeError(f'Insufficient evaluation space: {free/2**30:.2f} GiB free, {gib} GiB required')
    return free


def release_completed_caches(root):
    root=Path(root);removed=[]
    for folder in sorted(root.glob('scene_*')):
        status=folder/'evaluation/episodes/episode_0001/episode_status.json';result=folder/'launch_result.json';cache=folder/'runtime_cache'
        if not status.exists() or not result.exists() or not cache.exists():continue
        row=read(status)
        if row.get('status')!='completed' or row.get('error') or read(result).get('exit_code')!=0:continue
        if cache.is_symlink() or not cache.resolve().is_relative_to(root.resolve()) or cache.name!='runtime_cache':raise ValueError('Unsafe cache path')
        size=sum(p.stat().st_size for p in cache.rglob('*') if p.is_file())
        shutil.rmtree(cache);removed.append(dict(path=str(cache),bytes=size))
    atomic_json(root/'shader_cache_cleanup.json',dict(updated_utc=now(),removed=removed,reclaimed_bytes=sum(r['bytes'] for r in removed),weights_videos_traces_preserved=True))
    return removed
