"""
spatial pathfinding problem and utilities

some code adapted from https://github.com/brean/python-pathfinding

"""

import math
SQRT2 = math.sqrt(2)




def manhattan(dx, dy) -> float:
    """manhattan heuristics"""
    return dx + dy


def euclidean(dx, dy) -> float:
    """euclidean distance heuristics"""
    return math.sqrt(dx * dx + dy * dy)


def chebyshev(dx, dy) -> float:
    """ Chebyshev distance. """
    return max(dx, dy)


def octile(dx, dy) -> float:
    f = SQRT2 - 1
    if dx < dy:
        return f * dx + dy
    else:
        return f * dy + dx