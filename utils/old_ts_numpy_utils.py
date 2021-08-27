import numpy as np


def to_multivar_ts(singlevar_ts_seq):
    if type(singlevar_ts_seq) is list:
        singlevar_ts_seq = tuple(singlevar_ts_seq)
    arr = np.dstack(singlevar_ts_seq)
    if arr.shape[0] == 1:
        arr = arr.reshape(arr.shape[1:])
    return arr


def split_ts(
        ts_in,
        ts_out,
        n_steps_in,
        n_steps_out,
):
    assert ts_in.shape[0] == ts_out.shape[0]
    n = ts_in.shape[0] - n_steps_in - n_steps_out + 1
    xs = np.empty(shape=(n, n_steps_in) + ts_in.shape[1:])
    ys = np.empty(shape=(n, n_steps_out) + ts_out.shape[1:])

    for i in range(n):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        xs[i,] = ts_in[
                 i:end_ix,
                 ]
        ys[i,] = ts_out[
                 end_ix:out_end_ix,
                 ]
    return xs, ys


def train_val_test_split(xs, ys, test, val):
    n = xs.shape[0]
    nv = val
    if isinstance(val, float):
        nv = n * val
    nt = test
    if isinstance(test, float):
        nt = n * test
    train_slice = slice(0, n - nv - nt)
    val_slice = slice(n - nv - nt, n - nt)
    test_slice = slice(n - nt, n)
    return (
        (xs[train_slice], ys[train_slice]),
        (xs[val_slice], ys[val_slice]),
        (xs[test_slice], ys[test_slice]),
    )


def train_test_split(xs, ys, test):
    train_xy, _, test_xy = train_val_test_split(xs, ys, test, 0)
    return (train_xy, test_xy)
