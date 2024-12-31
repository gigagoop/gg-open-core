import numpy as np


class FpsTracker:
    length = 10
    min_frames = 5    # number of frames needed before a valid fps is returned

    def __init__(self):
        self._frame_index = 0
        self._history = np.nan * np.ones(self.length)

    def update(self, delta_time: float):
        fps = 1 / delta_time
        self._history[self._frame_index % self.length] = fps
        self._frame_index += 1

    @property
    def fps(self):
        if self._frame_index >= self.min_frames - 1:
            return np.nanmedian(self._history)
        else:
            return np.nan

    @property
    def frame_index(self) -> int:
        return self._frame_index