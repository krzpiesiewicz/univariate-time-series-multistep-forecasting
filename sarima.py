import numpy as np
import pandas as pd
import pmdarima as pm

import timeseries as tss

from model import Model
from scorings import get_scoring


class SarimaModel(Model):
    def __init__(self, p, d, q, P=0, D=0, Q=0, s=0,
                 maxiter=None, max_ts_len=np.inf, retrain_ts_len=None, constructor_params={}):
        self.constructor_params = constructor_params.copy()
        if "maxiter" not in self.constructor_params:
            maxiter = maxiter if maxiter is not None else 50
            self.constructor_params["maxiter"] = maxiter
        self.order=(p, d, q)
        self.seasonal_order=(P, D, Q, s)
        self.__create_inner_model__()
        self.max_ts_len = max_ts_len
        if retrain_ts_len is None:
            retrain_ts_len = max_ts_len
        self.retrain_ts_len=retrain_ts_len
        
    def __create_inner_model__(self):
        self.model = pm.ARIMA(order=self.order, seasonal_order=self.seasonal_order,
                              **self.constructor_params)
        

    def fit(self, ts, train_interval, scoring="mse", fit_params={}):
        scoring = get_scoring(scoring)
        self.ts = train_interval.view(ts)
        self.model.fit(self.ts, scoring=scoring, **fit_params)

    def update(self, ts, new_interval, scoring="mse", new_fit=False, update_params={}, fit_params={}):
        scoring = get_scoring(scoring)
        new_interval = self.__update_ts__(ts, new_interval)
        if not new_fit:
            update_params["out_of_sample_size"] = len(new_interval.view(self.ts)) - 1
            update_params["maxiter"] = 0
        if len(self.ts) <= self.max_ts_len:
            self.model.update(new_interval.view(self.ts), scoring=scoring, **update_params)
        else:
            self.ts = self.ts[self.ts.index[-self.retrain_ts_len:]]
            start_params = self.model.params()
            self.constructor_params["out_of_sample_size"] = 0#len(self.ts) - 1
            self.constructor_params["maxiter"] = 0
            self.constructor_params["start_params"] = start_params
            self.__create_inner_model__()
            self.fit(self.ts, tss.Interval(self.ts), scoring=scoring, fit_params=fit_params)

    def predict(self, ts, pred_interval, is_train=False):
        pred_ts = pred_interval.view()
        index = pred_ts.index
        if is_train:
            pred_values = self.model.predict_in_sample()
        else:
            n = len(index)
            pred_values = self.model.predict(n)
        return pd.Series(pred_values, index, name=pred_ts.name)
