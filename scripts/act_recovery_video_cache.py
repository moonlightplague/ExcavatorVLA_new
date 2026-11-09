"""Exact RGB frame cache for sequential evaluation of the lossless recovery videos."""
from pathlib import Path


class RecoveryRGBFrameCache:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.frames = {}
        self.timestamps = {}

    def __call__(self, fallback, video_path, timestamps, tolerance_s, backend=None):
        import av
        import numpy as np
        import torch
        path = Path(video_path).resolve()
        if not path.is_relative_to(self.root) or backend not in (None, 'pyav'):
            return fallback(video_path, timestamps, tolerance_s, backend)
        if path not in self.frames:
            pixels, times = [], []
            with av.open(str(path)) as container:
                for frame in container.decode(video=0):
                    pixels.append(frame.to_ndarray(format='rgb24').transpose(2, 0, 1))
                    times.append(frame.time)
            self.frames[path] = np.stack(pixels)
            self.timestamps[path] = torch.tensor(times)
        query = torch.tensor(timestamps)
        distance = torch.cdist(query[:, None], self.timestamps[path][:, None], p=1)
        minimum, indices = distance.min(1)
        if not (minimum < tolerance_s).all():
            raise ValueError(f'Recovery frame timestamp mismatch: {path}, {minimum.tolist()}')
        return torch.from_numpy(self.frames[path][indices.numpy()]).float() / 255
