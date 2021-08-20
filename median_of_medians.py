import numpy as np
import pandas as pd
import timeseries as tss

from model import Model


class MedianOfMediansModel(Model):
    def __init__(self, n_groups, group_size):
        self.n_groups = n_groups
        self.group_size = group_size

    def __median__(self, seq):
        seq = sorted(seq)
        n = len(seq)
        s = n // 2
        if 2 * s == n:
            return 0.5 * (seq[s - 1] + seq[s])
        else:
            return seq[s]
        
    def __update_median__(self):
        m = len(self.ts)
        d = self.group_size
        medians = []
        i = 0
        for _ in range(self.n_groups):
            if i + d <= m:
                if i == 0:
                    group = self.ts[-d:]
                else:
                    group = self.ts[-(i + d): -i]
                medians.append(self.__median__(group))
                i += d
            else:
                break
        if len(medians) == 0:
            raise Exception(
                f"time series is too short ({len(self.ts)}) but at least {d} needed"
            )
        self.median = self.__median__(medians)
        

    def fit(self, ts, train_interval, **kwargs):
        self.ts = train_interval.view(ts)
        self.__update_median__() 

    def update(self, ts, new_interval, **kwargs):
        self.__update_ts__(ts, new_interval)
        self.__update_median__()

    def predict(self, ts, pred_interval, is_train=False):
        pred_ts = pred_interval.view(ts)
        pred_index = pred_ts.index
        n = len(pred_index)
        pred_values = np.full(n, self.median)
        return pd.Series(pred_values, pred_index, name=pred_ts.name)
