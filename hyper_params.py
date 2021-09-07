import sys
import time
import numpy as np

from utils.timing import timedelta_str
from average_scoring import average_scores


def sorted_scores(scores):
    return sorted(scores, key=lambda t: t[0], reverse=False)


def grid_search_hyper_params(
        create_model,
        grid,
        ts,
        train_intv,
        val_intv,
        best=10,
        max_fails=20,
        scores=None,
        model_params={},
        fit_fun=None,
        fit_params={},
        score_fun=None,
        score_params={},
        max_line_len=111
):
    assert best is None or type(best) is int
    
    def default_fit_fun(model, ts, train_intv, **fit_params):
        model.fit(ts, train_intv, **fit_params)

    def default_score_fun(model, ts, val_intv, **score_params):
        scores = average_scores(model, ts, val_intv, **score_params)
        return list(scores.values())[0]
    
    def valuation_str(valuation, l=120):
        
        if valuation is None:
            return "None"
        
        def dct_to_string(dct):
            buff = ""
            first = True
            for key, value in dct.items():
                if not first:
                    buff += ", "
                first = False
                buff += f"{key}={value}"
            return buff
        
        original_str = dct_to_string(valuation)
        if len(original_str) <= l:
            return original_str
        
        dct_with_strs = {}
        for key, value in valuation.items():
            if type(value) is int:
                value = f"{value}"
            else:
                if type(value) is float:
                    value = "{value:.2f}"
            dct_with_strs[key] = value
            
        long_str = dct_to_string(dct_with_strs)
        if len(long_str) <= l:
            return long_str
        
        simple_dct = {}
        for key, value in valuation.items():
            if len(key) > 8:
                key = key = f"{key[:4]}..{key[-2:]}"
            new_key = key
            i = 1
            while new_key in simple_dct:
                new_key = f"{key}_{i}"
                i += 1
            simple_dct[new_key] = value
        
        short_str = dct_to_string(simple_dct)
        if len(short_str) > l:
            short_str = short_str[:l-3] + "..."
        return short_str
        

    if fit_fun is None:
        fit_fun = default_fit_fun
    if score_fun is None:
        score_fun = default_score_fun

    fails = 0
    if scores is None:
        scores = []
    best_score = np.inf
    best_valuation = None
    
    past_valuations = []
    for _, valuation in scores:
        if valuation not in past_valuations:
            past_valuations.append(valuation)
    grid = [valuation for valuation in grid if valuation not in past_valuations]
    
    start_time = time.time()
    n = len(grid)
    print(f"0/{n}", end="", file=sys.stderr)
    for i, hyper_params_values in enumerate(grid):
        if max_fails is not None and fails > max_fails:
            print(f"\nTerminating: more than {max_fails} fails", file=sys.stderr)
            break
        try:
            model = create_model(**hyper_params_values, **model_params)
            #             old_stderr = sys.stderr
            #             sys.stderr = open(os.devnull, "w")
            fit_fun(model, ts, train_intv, **fit_params)
            score = score_fun(model, ts, val_intv, **score_params)
        except:
            score = np.inf
            fails += 1
        finally:
            past_valuations.append(hyper_params_values)
        #             sys.stderr.close()
        #             sys.stderr = old_stderr
        scores.append((score, hyper_params_values))
        if score < best_score:
            best_score = score
            best_valuation = hyper_params_values
        buff = f"\r{i + 1}/{n}"
        if start_time is not None:
            buff += f" ({timedelta_str(time.time() - start_time)})"
        buff += f" best: {best_score:.6f} ("
        buff += f"{valuation_str(best_valuation, max_line_len - len(buff) - 1)})"
        buff += " " * (max_line_len - len(buff))
        print(buff, end="", file=sys.stderr)
    scores = sorted_scores(scores)
    best_score, best_valuation = scores[0]
    buff = f"the best of all: {best_score:.6f} ("
    buff = f"\r{buff}{valuation_str(best_valuation, max_line_len - len(buff) - 1)})"
    buff += " " * (max_line_len - len(buff))
    print(buff, file=sys.stderr)
    time.sleep(1)
    if best is None:
        return scores
    else:
        return scores[:best]


def make_grid(*args, **kwargs):
    if len(args) > 0:
        dct = args[0]
    else:
        dct = kwargs
    grid = []
    hyper_params_and_values = list(dct.items())

    def create_valuation(current, rest):
        if rest == []:
            grid.append(current.copy())
        else:
            param, values = rest[0]
            for value in values:
                current[param] = value
                create_valuation(current, rest[1:])

    create_valuation({}, hyper_params_and_values)

    return grid


def grid_from_values_sets(hyper_params, values_sets):
    grid = []
    for values in values_sets:
        assert len(values) == len(hyper_params)
        valuation = {}
        for param, value in zip(hyper_params, values):
            valuation[param] = value
        grid.append(valuation)
    return grid
