import numpy as np
import pandas as pd
import torch

from models.model import Model
from dnn.exogenous import get_exogenous_seasonal_array_from_dct_lst


class SequentialForecastingModel(Model):
    def __init__(self, module, device, detrans=None, seasons_dct_lst=None):
        super().__init__()
        self.module = module
        self.device = device
        self.detrans = detrans
        self.seasons_dct_lst = seasons_dct_lst

    def fit(self, ts, train_interval, *args, **kwargs):
        self.ts = train_interval.view(ts)

    def update(self, ts, new_interval, **kwargs):
        new_interval = self.__update_ts__(ts, new_interval)
        if new_interval is not None:
            x = new_interval.view(self.ts).values
            x = x.reshape(1, len(x), 1).astype(np.float32)
            x = torch.tensor(x).to(self.device)
            h = self.module.encoder(x)
            self.module.decoder.set_hidden_state(h)

    def predict(self, ts, pred_interval, original_ts=None, seasonal_ts_seq=None):
        target_index = pred_interval.index()
        pred_steps = len(target_index)
        assert len(pred_interval.index(ts)) == pred_steps

        y0 = torch.tensor(
            np.array(self.ts[-1]).reshape((1, 1, 1)).astype(np.float32)
        ).to(self.device)
        x = y0[:, 0:0, :]

        if seasonal_ts_seq is not None:
            assert self.seasons_dct_lst is not None
            assert type(seasonal_ts_seq) is list
            ex = get_exogenous_seasonal_array_from_dct_lst(
                pred_interval, seasonal_ts_seq, self.seasons_dct_lst
            )
            ex = np.swapaxes(ex, 0, 1)
            ex = ex.astype(np.float32)
            ex = torch.tensor(ex).reshape(1, ex.shape[0], ex.shape[1]).to(self.device)
        else:
            ex = None
        pred_values = (
            self.module(x=x, ex=ex, y=y0).cpu().detach().numpy().reshape(pred_steps)
        )
        pred = pd.Series(pred_values, target_index, name=pred_interval.ts.name)
        if self.detrans is not None:
            assert original_ts is not None
            pred = self.detrans.detransform(pred, pred_interval.prev_view(original_ts))
        pred.name = pred_interval.ts.name
        return pred
