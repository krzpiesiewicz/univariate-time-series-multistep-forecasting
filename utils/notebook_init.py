import warnings

import matplotlib as plt


def notebook_init():
    warnings.filterwarnings("ignore")
    plt.rcParams.update({
        "figure.max_open_warning": 0,
    })
