from abc import ABC, abstractmethod

import numpy as np
import timeseries as tss


class Model(ABC):

    def __init__(self):
        self.ts = None
        self.original_ts = None

    @abstractmethod
    def fit(self, ts, train_interval, original_ts=None, scoring="mse",
            **kwargs):
        ...

    @abstractmethod
    def predict(self, pred_interval, original_ts=None, **kwargs):
        ...

    @abstractmethod
    def update(self, ts, new_interval, original_ts=None, **kwargs):
        ...

    def __update_ts__(self, ts, next_interval, original_ts=None):
        assert self.ts is not None
        assert len(self.ts) > 0
        next_index = next_interval.index(ts)
        if len(next_index) == 0 or next_index[-1] <= self.ts.index[-1]:
            next_interval = None
        else:
#             print(f"next: {next_interval.view(ts)}")
            if next_index[-1] > self.ts.index[0]:
                i = np.where(next_index > self.ts.index[-1])[0][0]
                next_interval = tss.Interval(
                    ts,
                    begin=next_index[i],
                    end=next_interval.end
                )
            self.ts = self.ts.append(next_interval.view(ts))
            if self.original_ts is not None and original_ts is not None:
                self.original_ts = self.original_ts.append(
                    next_interval.view(original_ts))
#             print(f"self.ts: {self.ts}")
        return next_interval
