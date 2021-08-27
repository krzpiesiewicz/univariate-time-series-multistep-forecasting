import numpy as np
import pandas as pd

from models.model import Model


class MedianOfMediansModel(Model):
    def __init__(self, n_groups, group_size, m=1):
        super().__init__()
        self.n_groups = n_groups
        self.group_size = group_size
        self.m = m
        self.medians = [0 for _ in range(m)]

    def __median__(self, seq):
        seq = sorted(seq)
        n = len(seq)
        s = n // 2
        if 2 * s == n:
            return 0.5 * (seq[s - 1] + seq[s])
        else:
            return seq[s]

    def __update_medians__(self):
        n = len(self.ts)
        d = self.group_size
        m = self.m
        if n < d:
            raise Exception(
                f"time series is too short ({len(self.ts)}) but at least {d} needed"
            )
        if m == 1:
            medians = []
            i = 0
            for _ in range(self.n_groups):
                if i + d <= n:
                    if i == 0:
                        group = self.ts.iloc[-d:]
                    else:
                        group = self.ts.iloc[-(i + d): -i]
                    medians.append(self.__median__(group))
                    i += d
                else:
                    break
            self.medians[0] = self.__median__(medians)
        else:
            values = [[] for _ in range(m)]
            j = 0
            i = 0
            for _ in range(min(self.n_groups * d * m, n)):
                i -= 1
                j -= 1
                if j < 0:
                    j += m
                values[j].append(self.ts.iloc[i])
            for i, vals in enumerate(values):
                self.medians[i] = self.__median__(vals)

    def fit(self, ts, train_interval, **kwargs):
        self.ts = train_interval.view(ts)
        self.__update_medians__()

    def update(self, ts, new_interval, **kwargs):
        self.__update_ts__(ts, new_interval)
        self.__update_medians__()

    def predict(self, ts, pred_interval, is_train=False, **kwargs):
        pred_ts = pred_interval.view(ts)
        pred_index = pred_ts.index
        n = len(pred_index)
        m = self.m
        pred_values = np.array(self.medians * ((n + m - 1) // m))[:n]
        return pd.Series(pred_values, pred_index, name=pred_ts.name)
