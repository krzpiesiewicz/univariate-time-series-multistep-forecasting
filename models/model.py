from abc import ABC
from abc import abstractmethod

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
        if next_interval.view(ts).index[-1] > self.ts.index[0]:
            i = np.where(next_interval.view(ts).index > self.ts.index[-1])[0][
                0]
            next_interval = tss.Interval(
                ts,
                begin=next_interval.view(ts).index[i],
                end=next_interval.end
            )
        self.ts = self.ts.append(next_interval.view(ts))
        if self.original_ts is not None and original_ts is not None:
            self.original_ts = self.original_ts.append(
                next_interval.view(original_ts))
        return next_interval
