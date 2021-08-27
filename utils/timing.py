from datetime import timedelta


def timedelta_str(s):
    if type(s) is timedelta:
        s = s.total_seconds()
    m = int(s // 60)
    h = int(m // 60)
    m -= 60 * h
    s -= 3600 * h + 60 * m
    buff = ""
    if h > 0:
        buff += f"{h}h"
        if m > 0 or s > 0:
            buff += " "
    if m > 0 or (h > 0 and s > 0):
        buff += f"{m}min"
        if s > 0 or h > 0:
            buff += f" {int(s)}s"
    if h == 0 and m == 0:
        buff = f"{s:.1f}s"
    return buff
