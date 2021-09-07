import numpy as np
import pandas as pd
import torch

from models.model import Model
from dnn.exogenous import get_exogenous_seasonal_array_from_dct_lst


class TimeWindowForecastingModel(Model):
    def __init__(self, module, window_len, device, detrans=None, seasons_dct_lst=None):
        super().__init__()
        self.module = module
        self.window_len = window_len
        self.device = device
        self.detrans = detrans
        self.seasons_dct_lst = seasons_dct_lst

    def fit(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def predict(self, ts, pred_interval, original_ts=None, seasonal_ts_seq=None):
        target_index = pred_interval.index()
        pred_steps = len(target_index)
        assert len(pred_interval.index(ts)) == pred_steps

        x = pred_interval.view(ts, prevs=self.window_len, nexts=-pred_steps).values
        x = x.reshape(1, 1, self.window_len).astype(np.float32)
        x = torch.tensor(x).to(self.device)

        if seasonal_ts_seq is not None:
            assert self.seasons_dct_lst is not None
            assert type(seasonal_ts_seq) is list
            ex = get_exogenous_seasonal_array_from_dct_lst(
                pred_interval, seasonal_ts_seq, self.seasons_dct_lst
            )
            ex = ex.astype(np.float32)
            ex = torch.tensor(ex).reshape(1, ex.shape[0], pred_steps).to(self.device)
        else:
            ex = None
        pred_values = self.module(x, ex).cpu().detach().numpy().reshape(pred_steps)
        pred = pd.Series(pred_values, target_index, name=pred_interval.ts.name)
        if self.detrans is not None:
            assert original_ts is not None
            pred = self.detrans.detransform(pred, pred_interval.prev_view(original_ts))
        pred.name = pred_interval.ts.name
        return pred
