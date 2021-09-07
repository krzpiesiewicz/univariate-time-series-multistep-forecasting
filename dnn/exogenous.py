import numpy as np


def get_exogenous_seasonal_dct_lst(seasonal_ts_seq):
    assert type(seasonal_ts_seq) is list
    seasons_dct_lst = []
    last_i = 0
    for seasonal_ts in seasonal_ts_seq:
        seasonal_values = np.unique(seasonal_ts)
        seasons_dct = {s: last_i + i for i, s in enumerate(seasonal_values)}
        seasons_dct_lst.append(seasons_dct)
        last_i += len(seasons_dct)
    return seasons_dct_lst


def get_exogenous_seasonal_array(intv, seasonal_ts_seq):
    seasons_dct_lst = get_exogenous_seasonal_dct_lst(seasonal_ts_seq)
    return get_exogenous_seasonal_array_from_dct_lst(
        intv, seasonal_ts_seq, seasons_dct_lst
    )


def get_exogenous_seasonal_array_from_dct_lst(intv, seasonal_ts_seq, seasons_dct_lst):
    assert type(seasonal_ts_seq) is list
    m = np.sum([len(seasons_dct) for seasons_dct in seasons_dct_lst])
    n = len(intv.view(seasonal_ts_seq[0]))
    exogenous = np.zeros((m, n))
    for seasonal_ts, seasons_dct in zip(seasonal_ts_seq, seasons_dct_lst):
        s_ts = intv.view(seasonal_ts).values
        for i, s in enumerate(s_ts):
            exogenous[seasons_dct[s], i] = 1.0
    return exogenous
