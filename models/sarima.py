import numpy as np
import pandas as pd
import pmdarima as pm
import timeseries as tss
from timeseries.transform import get_smoothed, get_interpolated

from models.model import Model


class SarimaModel(Model):
    def __init__(self, p, d, q, P=0, D=0, Q=0, s=0, smoothing_std=None,
                 detrans=None,
                 maxiter=None, max_ts_len=np.inf, retrain_ts_len=None,
                 constructor_params={}):
        super().__init__()
        self.constructor_params = constructor_params.copy()
        if "maxiter" not in self.constructor_params:
            maxiter = maxiter if maxiter is not None else 50
            self.constructor_params["maxiter"] = maxiter
        self.order = (p, d, q)
        self.seasonal_order = (P, D, Q, s)
        self.smoothing_std = smoothing_std
        self.detrans = detrans
        self.__create_inner_model__()
        self.max_ts_len = max_ts_len
        if retrain_ts_len is None:
            retrain_ts_len = max_ts_len
        self.retrain_ts_len = retrain_ts_len

    def __create_inner_model__(self):
        self.model = pm.ARIMA(order=self.order,
                              seasonal_order=self.seasonal_order,
                              **self.constructor_params)

    def fit(self, ts, train_interval, fit_params={}, **kwargs):
        for (key, value) in kwargs.items():
            fit_params[key] = value
        if "scoring" not in fit_params:
            fit_params["scoring"] = "mse",
        self.ts = train_interval.view(ts)
        self.model.fit(self.ts, **fit_params)

    def update(self, ts, new_interval, new_fit=False, update_params={},
               fit_params={}, **kwargs):
        new_interval = self.__update_ts__(ts, new_interval)
        if new_interval is not None:
            if not new_fit:
                update_params["out_of_sample_size"] = len(
                    new_interval.view(self.ts)) - 1
                update_params["maxiter"] = 0
            if len(self.ts) <= self.max_ts_len:
                if "scoring" in fit_params:
                    update_params["scoring"] = fit_params["scoring"]
                self.model.update(new_interval.view(self.ts), **update_params)
            else:
                self.ts = self.ts[self.ts.index[-self.retrain_ts_len:]]
                start_params = self.model.params()
                self.constructor_params[
                    "out_of_sample_size"] = 0  # len(self.ts) - 1
                self.constructor_params["maxiter"] = 0
                self.constructor_params["start_params"] = start_params
                self.__create_inner_model__()
                self.fit(self.ts, tss.Interval(self.ts), fit_params=fit_params)

    def predict(self, ts, pred_interval, original_ts=None, **kwargs):
        target_index = pred_interval.index()
        n = len(target_index)
        index = pred_interval.index(ts)
#         print(f"index: {index}")
        m = len(index)
        if n == m:
            pred_values = self.model.predict(m)
            pred = pd.Series(pred_values, target_index, name=pred_interval.ts.name)
            if self.detrans is not None:
                assert original_ts is not None
                pred = self.detrans.detransform(pred, pred_interval.prev_view(
                    original_ts))
        else:
            index = pred_interval.index(ts, prevs=1, nexts=1)
#             print(f"index: {index}")
#             print(f"index2: {index[-1:]}")
#             print(f"self.ts: {self.ts.index}")
            m = len(index)
            pred_values = np.zeros(m)
            pred_values[1:] = self.model.predict(m - 1)
#             print(index)
#             print(self.ts[index[0:1]])
#             print(pred_interval.prev_view(original_ts).iloc[:-1])
            pred_values[0] = self.ts[index[0]]
            pred = pd.Series(pred_values, index)
            assert self.detrans is not None
            assert original_ts is not None
            pred = self.detrans.detransform(pred, pred_interval.prev_view(original_ts).iloc[:-1])
#             print(pred)
            pred = get_interpolated(pred, pred_interval)
        if self.smoothing_std is not None:
            pred = get_smoothed(pred, self.smoothing_std)
        pred.name = pred_interval.ts.name
        return pred
