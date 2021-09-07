import numpy as np
import pandas as pd
import torch

from models.model import Model
from dnn.exogenous import get_exogenous_seasonal_array_from_dct_lst


class SequentialAutoencodingModel(Model):
    def __init__(self, module, device, prefix_len=None, detrans=None):
        super().__init__()
        self.module = module
        self.device = device
        self.detrans = detrans
        self.prefix_len = prefix_len

    def fit(self, *args, **kwargs):
        pass

    def update(self, **kwargs):
        pass

    def predict(self, ts, pred_interval, original_ts=None, seasonal_ts_seq=None):
        target_index = pred_interval.index()
        steps_back = len(target_index)
        assert len(pred_interval.index(ts)) == steps_back

        prefix_len = self.prefix_len if self.prefix_len is not None else steps_back
        prefix_len -= steps_back
        
        x = pred_interval.view(ts, prevs=prefix_len).values.astype(np.float32)
        x = torch.tensor(x.reshape((1,len(x),1))).to(self.device)
        
        self.module.set(output_len=steps_back)
        pred_values = (
            self.module(x=x).cpu().detach().numpy().reshape(steps_back)
        )
        pred = pd.Series(pred_values, target_index, name=pred_interval.ts.name)
        if self.detrans is not None and original_ts is not None:
            pred = self.detrans.detransform(pred, pred_interval.prev_view(original_ts))
        pred.name = pred_interval.ts.name
        return pred
