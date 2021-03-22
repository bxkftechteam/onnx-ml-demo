#!/usr/bin/env python3

"""A mapping from probability to discrete values"""


def map_pred(x):
    if x < 0.1:
        return 0
    elif x < 0.3:
        return 1
    elif x < 0.5:
        return 2
    elif x < 0.7:
        return 3
    elif x < 0.9:
        return 4
    else:
        return 5
