from timeseries.plotting import plot_ts, plot_hist


def plot_ground_truth(intv, ts=None, title=None, model_name=None, model_version=None, mode=None,
                      data_type=None, data_name=None, color="#ff9999", **kwargs):
    if ts is None:
        ts = intv.view()
    if title is None:
        if data_type is not None or data_name is not None:
            if data_type is not None:
                data_type_name = data_type
                if data_name is not None:
                    data_type_name += f": {data_name}"
            else:
                data_type_name = data_name
            data_type_name += " – "
        else:
            data_type_name = ""
        if model_name is not None or model_version is not None:
            if model_name is not None:
                model_name_version = model_name
                if model_version is not None:
                    model_name_version += f": {model_version}"
            else:
                model_name_version = model_version
            model_name_version += " – "
        else:
            model_name_version = ""
        title = f"{data_type_name}{model_name_version}Some {mode[0].upper()}{mode[1:]} Predictions"
    return plot_ts(intv.view(ts), color=color, title=title, **kwargs)


def plot_model_test_prediction(intv, time_delta, preds, color="darkred", model_name=None, model_version=None,
                               name=None, fig=None, **kwargs):
    if name is None and (model_name is not None or model_version is not None):
        if model_name is not None:
            name = model_name
            if model_version is not None:
                name += f" {model_version}"
        else:
            name = model_version
    last_dt = intv.begin
    for i, pred_ts in enumerate(preds):
        if intv.end is not None and pred_ts.index[-1] > intv.end:
            break
        if pred_ts.index[0] >= last_dt + time_delta:
            fig = plot_ts(pred_ts, color=color, name=name, **kwargs, fig=fig)
            name = None
            last_dt = pred_ts.index[-1]
    return fig


def plot_hist_model_scores(
    scores,
    scoring_name,
    data_type=None,
    data_name=None,
    model_name=None,
    model_version=None,
    in_label=False,
    title=None,
    name=None,
    fig=None,
    **kwargs
):
    if model_name is not None or model_version is not None:
        model_name_version = model_name
        if model_version is not None:
            model_name_version += f" {model_version}"
        else:
            name = model_version
    if in_label:
        name = model_name_version
    if fig is None and title is None:
        if data_type is not None or data_name is not None:
            if data_type is not None:
                data_type_name = data_type
                if data_name is not None:
                    data_type_name += f": {data_name}"
            else:
                data_type_name = data_name
            data_type_name += " – "
        else:
            data_type_name = ""
        if in_label:
            title = f"{data_type_name}Distribution of {scoring_name}"
        else:
            title = f"{data_type_name}{model_name_version} – Distribution of {scoring_name}"
        
    return plot_hist(
        scores[scoring_name],
        title=title,
        name=name,
        fig=fig,
        **kwargs
    )
