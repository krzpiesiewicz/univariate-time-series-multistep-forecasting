from abc import ABC
from abc import abstractmethod

import numpy as np
import timeseries as tss


class Model(ABC):
    
    def __init__(self):
        self.ts = None
        
    @abstractmethod
    def fit(self, train_interval, scoring="mse", **kwargs):
        ...

    @abstractmethod
    def predict(self, pred_interval, **kwargs):
        ...
        
    @abstractmethod
    def update(self, ts, new_interval, **kwargs):
        ...

    
    def __update_ts__(self, ts, next_interval):
        assert self.ts is not None
        assert len(self.ts) > 0
        if next_interval.view(ts).index[-1] > self.ts.index[0]:
            i = np.where(next_interval.view(ts).index > self.ts.index[-1])[0][0]
            next_interval = tss.Interval(
                ts,
                begin=next_interval.view(ts).index[i],
                end=next_interval.end
            )
        self.ts = self.ts.append(next_interval.view(ts))
        return next_interval