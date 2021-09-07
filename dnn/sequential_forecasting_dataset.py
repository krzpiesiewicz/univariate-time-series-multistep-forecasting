import numpy as np
import torch
from torch.utils.data import Dataset

from dnn.exogenous import get_exogenous_seasonal_array


class SequentialForecastingDataset(Dataset):
    def __init__(
        self,
        ts,
        intv,
        prefix_len,
        pred_steps=None,
        original_ts=None,
        original_prevs_len=None,
        reverse=False,
        exogenous=None,
        seasonal_ts_seq=None,
        batch_size=None,
        jump_update_steps=0,
        debug=False,
    ):
        self.prefix_len = prefix_len
        self.reverse = reverse
        self.y = intv.view(ts).values
        self.y = self.y.reshape((len(self.y), 1))
        self.x = intv.view(ts, prevs=self.prefix_len).values
        self.x = self.x.reshape((len(self.x), 1))
        self.exogenous = exogenous
        if seasonal_ts_seq is not None:
            assert self.exogenous is None
            assert type(seasonal_ts_seq) is list
            self.exogenous = get_exogenous_seasonal_array(intv, seasonal_ts_seq)
        self.pred_steps = pred_steps
        self.batch_size = batch_size
        self.jump_update_steps = jump_update_steps
        self.debug = debug
        if original_ts is not None:
            assert original_prevs_len is not None
            self.original_prevs_len = original_prevs_len
            self.original = intv.view(
                original_ts,
                prevs=self.original_prevs_len,
                #                 nexts=-self.pred_steps,
            ).values
            self.original = self.original.reshape((len(self.original), 1))
        else:
            self.original = None
        self.intv = intv

    def set(self, batch_size=None, pred_steps=None):
        if batch_size is not None:
            self.batch_size = batch_size
        if pred_steps is not None:
            self.pred_steps = pred_steps

    def __len__(self):
        if self.batch_size is None:
            raise Exception(
                "batch size not provided – use self.reset(batch_size=) to set value"
            )
        if self.pred_steps is None:
            raise Exception(
                "pred_steps not provided – use self.set(pred_steps=) to set value"
            )
        samples = len(self.x) - self.pred_steps - self.prefix_len - self.batch_size
        samples //= self.jump_update_steps + self.batch_size
        samples = (samples + 1) * self.batch_size
        return samples

    def __getitem__(self, idx):
        if self.batch_size is None:
            raise Exception(
                "batch size not provided – use self.reset(batch_size=) to set value"
            )
        if self.pred_steps is None:
            raise Exception(
                "pred_steps not provided – use self.set(pred_steps=) to set value"
            )
        if torch.is_tensor(idx):
            idx = idx.numpy()
        end = self.prefix_len + (idx // self.batch_size) * self.jump_update_steps + idx
        prev_end = end - self.batch_size - self.jump_update_steps
        if self.debug:
            print(f"idx: {idx} ({prev_end}-{end})")
        if idx < self.batch_size:
            prev_end = idx
        x = np.empty((end - prev_end, self.x.shape[1]), dtype=np.float32)
        x[:] = self.x[prev_end:end, :]
        x = x.copy()
        prev_end = end - self.prefix_len
        end = prev_end + self.pred_steps
        if self.exogenous is not None:
            ex = np.empty((self.exogenous.shape[0], self.pred_steps), dtype=np.float32)
            ex[:, :] = self.exogenous[:, prev_end:end]
            ex = np.swapaxes(ex, 0, 1)
            xs = (x, ex)
        else:
            xs = x
        y = np.empty((self.x.shape[1], self.pred_steps), dtype=np.float32)
        y[:] = self.y[
            prev_end:end,
            :,
        ].reshape(self.pred_steps)
        if self.reverse:
            y = np.flip(y, 1)
        y = y.copy()
        if self.original is not None:
            orignal_prevs = (
                self.original[
                    prev_end : prev_end + self.original_prevs_len,
                    :,
                ]
                .reshape((self.original.shape[1], self.original_prevs_len))
                .copy()
            )
            ys = (y, orignal_prevs)
        else:
            ys = y
        return xs, ys
