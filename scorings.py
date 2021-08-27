import numpy as np
import pandas as pd
import pmdarima as pm
import sklearn.metrics


def mase(true, pred):
    if len(true) == 1:
        raise Exception("Length of sequence has to be at least 2")
    if type(true) is pd.Series:
        true = true.values
    if type(pred) is pd.Series:
        pred = pred.values
    diffs = np.abs(true[1:] - true[:-1])
    errors = np.abs(true - pred)
    return np.mean(errors) / np.mean(diffs)


def rmse(true, pred):
    return np.sqrt(sklearn.metrics.mean_squared_error(true, pred))


def mspe(true, pred):
    if type(true) is pd.Series:
        true = true.values
    if type(pred) is pd.Series:
        pred = pred.values
    errors = true - pred
    return np.mean(np.power((errors / true), 2))


def rmspe(true, pred):
    return np.sqrt(mspe(true, pred))


scoring_dct = dict(
    mse=sklearn.metrics.mean_squared_error,
    MSE=sklearn.metrics.mean_squared_error,
    rmse=rmse,
    RMSE=rmse,
    RMSPE=rmspe,
    rmspe=rmspe,
    mae=sklearn.metrics.mean_absolute_error,
    MAE=sklearn.metrics.mean_absolute_error,
    mape=sklearn.metrics.mean_absolute_percentage_error,
    MAPE=sklearn.metrics.mean_absolute_percentage_error,
    smape=pm.metrics.smape,
    sMAPE=pm.metrics.smape,
    mase=mase,
    MASE=mase
)


def get_scoring(scoring):
    if type(scoring) is str:
        if scoring in scoring_dct:
            return scoring_dct[scoring]
        else:
            raise Exception(
                f"Not known scoring. Available ones are: {scoring_dct.keys()}")
    else:
        return scoring


def get_comparison_scorings():
    return ["RMSE", "MAE", "MASE", "sMAPE"]
