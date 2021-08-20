import numpy as np
import timeseries as tss

from scorings import get_scoring


def average_scores(
        model,
        ts,
        score_intv,
        scorings,
        n_steps,
        n_steps_jump=1,
        trans=None,
        original_ts=None,
        return_preds=False,
        return_all_scores=False,
        update_params={}
):
    if trans is not None:
        assert original_ts is not None
    else:
        assert original_ts is None
        original_ts = ts
    index = score_intv.view(original_ts).index
    all_scores = {scoring_name: [] for scoring_name in scorings}
    saved_preds = []
    for i in range(0, index.size - n_steps, n_steps_jump):
        begin = index[i]
        end = index[i + n_steps]
        intv = tss.Interval(original_ts, begin, end)
        ts_true = intv.view(original_ts)
        ts_pred = model.predict(ts, intv)
        if trans is not None:
            true_prevs = intv.prev_view(original_ts)
            ts_pred = trans.detransform(ts_pred, true_prevs)
        for scoring_name in scorings:
            scoring = get_scoring(scoring_name)
            all_scores[scoring_name].append(scoring(ts_true, ts_pred))
        if return_preds:
            saved_preds.append(ts_pred)
        if i + n_steps_jump < index.size:
            model.update(ts,
                         tss.Interval(ts, index[i], index[i + n_steps_jump]),
                         update_params=update_params)
    mean_scores = {
        scoring_name: np.mean(all_scores[scoring_name]) for scoring_name in
        scorings
    }
    if return_preds or return_all_scores:
        res = (mean_scores,)
        if return_all_scores:
            res += (all_scores,)
        if return_preds:
            res += (saved_preds,)
        return res
    else:
        return mean_scores
