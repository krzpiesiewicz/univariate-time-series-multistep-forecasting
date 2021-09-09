import numpy as np
import torch
from torch.utils.data import Dataset

from dnn.exogenous import get_exogenous_seasonal_array


class SequentialAutoencodingDataset(Dataset):
    def __init__(
        self,
        ts,
        intv,
        prefix_len,
        steps_back=None,
        original_ts=None,
        original_prevs_len=None,
        one_more=False,
        teacher_forcing=False,
        input_equal_to_output=False,
        reverse=False,
        batch_size=None,
        jump_update_steps=0,
        debug=False,
    ):
        self.prefix_len = prefix_len
        self.reverse = reverse
        self.x = intv.view(ts, prevs=self.prefix_len).values
        self.x = self.x.reshape((len(self.x), 1))
        self.steps_back = steps_back
        self.batch_size = batch_size
        self.jump_update_steps = jump_update_steps
        self.debug = debug
        self.one_more = one_more
        self.teacher_forcing = teacher_forcing
        self.input_equal_to_output = input_equal_to_output
        if original_ts is not None:
            assert original_prevs_len is not None
            self.original_prevs_len = original_prevs_len
            self.original = intv.view(
                original_ts, prevs=self.original_prevs_len + 1
            ).values
            self.original = self.original.reshape((len(self.original), 1))
        else:
            self.original = None
        self.intv = intv

    def set(
        self,
        batch_size=None,
        steps_back=None,
        reverse=None,
        one_more=None,
        teacher_forcing=None,
        input_equal_to_output=None,
    ):
        if batch_size is not None:
            self.batch_size = batch_size
        if steps_back is not None:
            self.steps_back = steps_back
        if reverse is not None:
            self.reverse = reverse
        if one_more is not None:
            self.one_more = one_more
        if teacher_forcing is not None:
            self.teacher_forcing = teacher_forcing
        if input_equal_to_output is not None:
            self.input_equal_to_output = input_equal_to_output

    def __len__(self):
        if self.batch_size is None:
            raise Exception(
                "batch size not provided – use self.reset(batch_size=) to set value"
            )
        if self.steps_back is None:
            raise Exception(
                "steps_back not provided – use self.set(steps_back=) to set value"
            )
        samples = len(self.x) - self.prefix_len - self.steps_back - self.batch_size
        samples //= self.jump_update_steps + self.batch_size
        samples = (samples + 1) * self.batch_size
        return samples

    def __getitem__(self, idx):
        if self.batch_size is None:
            raise Exception(
                "batch size not provided – use self.set(batch_size=) to set value"
            )
        if self.steps_back is None:
            raise Exception(
                "steps_back not provided – use self.set(steps_back=) to set value"
            )
        if torch.is_tensor(idx):
            idx = idx.numpy()
        end = (
            self.prefix_len
            + self.steps_back
            + (idx // self.batch_size) * self.jump_update_steps
            + idx
        )
        prev_end = end - self.batch_size - self.jump_update_steps
        if self.debug:
            print(f"idx: {idx} ({prev_end}-{end})")
        if idx < self.batch_size:
            prev_end = idx
        if self.input_equal_to_output:
            prev_end = end - self.steps_back
        x = np.empty((end - prev_end, self.x.shape[1]), dtype=np.float32)
        x[:] = self.x[prev_end:end, :]
        x = x.copy()
        if self.teacher_forcing:
            tf = np.zeros((self.steps_back, self.x.shape[1]), dtype=np.float32)
            tmp = self.x[
                end - self.steps_back + 1 : end,
                :,
            ].astype(np.float32)
            if self.reverse:
                tmp = np.flip(tmp, 0).copy()
            tf[1:, :] = tmp
            xs = (x, tf)
        else:
            xs = x
        y = np.empty((self.x.shape[1], self.steps_back), dtype=np.float32)
        y[:] = self.x[
            end - self.steps_back - self.one_more : end - self.one_more,
            :,
        ].reshape(self.steps_back)
        if self.reverse:
            y = np.flip(y, 1)
        y = y.copy()
        if self.original is not None:
            pos = end - self.steps_back - self.prefix_len - self.one_more + 1
            orignal_prevs = (
                self.original[
                    pos : pos + self.original_prevs_len,
                    :,
                ]
                .reshape((self.original.shape[1], self.original_prevs_len))
                .copy()
            )
            ys = (y, orignal_prevs)
        else:
            ys = y

        return xs, ys
