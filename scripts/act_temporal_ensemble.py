"""Blend overlapping predictions for the same action timestamp, newest first."""
from collections import deque
import math

import numpy as np


class TemporalActionEnsemble:
    def __init__(self, window=3, decay=.5):
        if type(window) is not int or window < 1 or not math.isfinite(decay) or decay < 0:
            raise ValueError('Temporal ensemble needs a positive window and finite nonnegative decay')
        self.window = window
        self.decay = decay
        self.chunks = deque(maxlen=window)
        self.step = 0

    def update(self, chunk):
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[1] != 4 or not len(chunk) or not np.isfinite(chunk).all():
            raise ValueError('Expected a finite [chunk, 4] physical-unit action array')
        self.chunks.append((self.step, chunk.copy()))
        predictions, weights = [], []
        for start, prediction in self.chunks:
            age = self.step - start
            if age < len(prediction):
                predictions.append(prediction[age])
                weights.append(math.exp(-self.decay * age))
        self.step += 1
        return np.average(predictions, axis=0, weights=weights).astype(np.float32)
