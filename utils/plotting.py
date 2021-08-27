from timeseries.plotting import plot_ts, plot_hist


def plot_ground_truth(intv, ts=None, title=None, model_name=None, mode=None,
                      data_name=None, color="#ff9999", **kwargs):
    if ts is None:
        ts = intv.view()
    if title is None:
        title = f"{data_name} – {model_name} – Some {mode[0].upper()}{mode[1:]} Predictions"
    return plot_ts(intv.view(ts), color=color, title=title, **kwargs)


def plot_model_test_prediction(intv, time_delta, preds, color="darkred",
                               name=None, fig=None, **kwargs):
    last_dt = intv.begin
    for i, pred_ts in enumerate(preds):
        if intv.end is not None and pred_ts.index[-1] > intv.end:
            break
        if pred_ts.index[0] >= last_dt + time_delta:
            fig = plot_ts(pred_ts, color=color, name=name, **kwargs, fig=fig)
            name = None
            last_dt = pred_ts.index[-1]
    return fig


def plot_hist_model_scores(data_name, model_name, scores, scoring_name,
                           **kwargs):
    return plot_hist(
        scores[scoring_name],
        title=f"{data_name} – {model_name} – Distribution of {scoring_name}",
        **kwargs
    )
