import numpy as np

import pytorch_fit
from pytorch_fit.metrics import Metric

from scorings import get_scoring


class OriginalMetric(Metric):
    def __init__(self, trans, scoring_name, reverse_output=False):
        super().__init__()
        self.trans = trans
        self.scoring_name = scoring_name
        self.scoring = get_scoring(scoring_name)
        self.reverse_output = reverse_output
        self.steps = None
        self.reset_state()

    def reset_state(self):
        self.sum_of_errors = 0
        self.all_samples = 0

    def set_steps(self, steps):
        self.steps = steps

    def update_state(self, y_pred, y_true, original_prevs, *args):
        y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()
        if self.reverse_output:
            y_pred = np.flip(y_pred, 2)
        if self.steps is not None:
            y_pred = y_pred[:, :, : self.steps]
            y_true = y_true[:, :, : self.steps]
        #             print(f"y_pred.shape: {y_pred.shape}, y_true.shape: {y_true.shape}")
        original_prevs = original_prevs.numpy()
        n = len(y_pred)
        for i in range(n):
            pred = y_pred[i]
            true = y_true[i]
            prevs = original_prevs[i]
            m = len(pred)
            s = 0
            for j in range(m):
                original_pred = self.trans.detransform(pred[j], prevs[j])
                original_true = self.trans.detransform(true[j], prevs[j])
                s += self.scoring(original_true, original_pred)
            self.sum_of_errors += s / m
        self.all_samples += n

    def value(self):
        return self.sum_of_errors / self.all_samples

    def is_value_simple(self):
        return True

    def name(self):
        return self.scoring_name

    def short_name(self):
        return self.scoring_name.lower()


class StepsLoss:
    def __init__(self, loss):
        self.steps = None
        self.loss = loss

    def set_steps(self, steps):
        self.steps = steps

    def __call__(self, y_pred, y_true):
        if self.steps is not None:
            y_pred = y_pred[:, :, : self.steps]
            y_true = y_true[:, :, : self.steps]
        #             print(f"y_pred.shape: {y_pred.shape}, y_true.shape: {y_true.shape}")
        return self.loss(y_pred, y_true)
