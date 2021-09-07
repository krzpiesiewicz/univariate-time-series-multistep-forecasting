import numpy as np
import torch
from torch.utils.data import Dataset

from dnn.exogenous import get_exogenous_seasonal_array


class TimeWindowForecastingDataset(Dataset):
    def __init__(
        self,
        ts,
        intv,
        max_window_len=None,
        window_len=None,
        pred_steps=None,
        original_ts=None,
        original_prevs_len=None,
        exogenous=None,
        seasonal_ts_seq=None,
        debug=False,
    ):
        self.debug = debug
        self.window_len = window_len
        if max_window_len is None and window_len is None:
            raise Exception("One of: max_window_len, window_len has to be provided")
        self.max_window_len = (
            max_window_len if max_window_len is not None else window_len
        )
        self.pred_steps = pred_steps
        self.y = intv.view(ts).values
        self.y = self.y.reshape((len(self.y), 1))
        self.x = intv.view(ts, prevs=self.max_window_len).values
        self.x = self.x.reshape((len(self.x), 1))
        self.exogenous = exogenous
        if seasonal_ts_seq is not None:
            assert self.exogenous is None
            assert type(seasonal_ts_seq) is list
            self.exogenous = get_exogenous_seasonal_array(intv, seasonal_ts_seq)
        if original_ts is not None:
            assert original_prevs_len is not None
            self.original_prevs_len = original_prevs_len
            self.original = intv.view(
                original_ts,
                prevs=self.original_prevs_len,
            ).values
            self.original = self.original.reshape((len(self.original), 1))
        else:
            self.original = None
        self.intv = intv

    def set(self, window_len=None, pred_steps=None):
        if window_len is not None:
            assert window_len <= self.max_window_len
            self.window_len = window_len
        if pred_steps is not None:
            self.pred_steps = pred_steps

    def __len__(self):
        if self.window_len is None:
            raise Exception(
                "window_len not provided – use self.set(window_len=) to set value"
            )
        if self.pred_steps is None:
            raise Exception(
                "pred_steps not provided – use self.set(pred_steps=) to set value"
            )
        return len(self.y) - self.pred_steps

    def __getitem__(self, idx):
        if self.window_len is None:
            raise Exception(
                "window_len not provided – use self.set(window_len=) to set value"
            )
        if self.pred_steps is None:
            raise Exception(
                "pred_steps not provided – use self.set(pred_steps=) to set value"
            )
        x = np.empty((self.x.shape[1], self.window_len), dtype=np.float32)
        x[:] = self.x[idx : idx + self.window_len, :].reshape(
            (self.x.shape[1], self.window_len)
        )
        y = np.empty((self.y.shape[1], self.pred_steps), dtype=np.float32)
        y[:] = self.y[idx : idx + self.pred_steps, :].reshape(
            (self.y.shape[1], self.pred_steps)
        )

        if self.exogenous is not None:
            ex = np.empty((self.exogenous.shape[0], self.pred_steps), dtype=np.float32)
            ex[:, :] = self.exogenous[:, idx : idx + self.pred_steps]
            xs = (x, ex)
        else:
            xs = x
        if self.original is not None:
            orignal_prevs = self.original[
                idx : idx + self.original_prevs_len, :
            ].reshape((self.original.shape[1], self.original_prevs_len))
            ys = (y, orignal_prevs)
        else:
            ys = y
        return xs, ys
