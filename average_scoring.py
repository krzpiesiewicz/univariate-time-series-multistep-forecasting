import time

import numpy as np
import timeseries as tss

from scorings import get_scoring
from utils.timing import timedelta_str


def average_scores(
        model,
        ts,
        score_intv,
        scorings,
        n_steps,
        n_steps_jump=1,
        trans=None,
        score_ts=None,
        original_ts=None,
        seasonal_ts_seq=None,
        return_preds=False,
        return_all_scores=False,
        mute=False,
        update_params={}
):
    if trans is not None:
        assert original_ts is not None
    if score_ts is None:
        score_ts = original_ts if original_ts is not None else ts
    index = score_intv.view(score_ts).index
    all_scores = {scoring_name: [] for scoring_name in scorings}
    saved_preds = []

    i_range = range(0, index.size - n_steps, n_steps_jump)
    all_examples = len(i_range)
    past_examples = 0
    sum_of_scores = {scoring_name: 0 for scoring_name in scorings}

    def print_progress(start_time=None, last_time=None):
        print(f"\r{past_examples}/{all_examples} – ", end="")
        for j, scoring_name in enumerate(scorings):
            mean_score = sum_of_scores[
                             scoring_name] / past_examples if past_examples > 0 else 0
            print(f"{scoring_name}: {mean_score:.5f}, ", end="")
        if start_time is not None:
            print(f"elapsed time: {timedelta_str(time.time() - start_time)}",
                  end="")
        if last_time is not None:
            print(f" (last: {timedelta_str(time.time() - last_time)})", end="")
        print(" " * 6, end="")

    if not mute:
        print_progress()
    start_time = time.time()
    for i in i_range:
        last_time = time.time()
        begin = index[i]
        end = index[i + n_steps]
        intv = tss.Interval(score_ts, begin, end)
        ts_true = intv.view(score_ts)
        ts_pred = model.predict(ts, intv, original_ts=original_ts, seasonal_ts_seq=seasonal_ts_seq)
        if trans is not None:
            ts_true = intv.view(score_ts)
            true_prevs = intv.prev_view(original_ts)
            ts_pred = trans.detransform(ts_pred, true_prevs)
        for scoring_name in scorings:
            scoring = get_scoring(scoring_name)
            score = scoring(ts_true, ts_pred)
            sum_of_scores[scoring_name] += score
            all_scores[scoring_name].append(score)
        if return_preds:
            saved_preds.append(ts_pred)
        if i + n_steps_jump < index.size:
            model.update(ts,
                         tss.Interval(ts, index[i], index[i + n_steps_jump]),
                         original_ts=original_ts,
                         seasonal_ts_seq=seasonal_ts_seq,
                         update_params=update_params)
        past_examples += 1
        if not mute:
            print_progress(start_time, last_time)
    if not mute:
        print("")
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
