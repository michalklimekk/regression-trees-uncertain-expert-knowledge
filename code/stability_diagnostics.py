"""Deletion-based diagnostic for empirical stump directions."""

import numpy as np


EPS = 1e-12


def best_stump_direction(x, y, min_samples_leaf=10):
    """Return the orientation of the best admissible squared-error stump.

    Returns +1 when the right-child mean is larger, -1 when it is smaller, and
    0 when no admissible split exists or the best split has equal child means.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("x and y must be one-dimensional arrays of equal length")

    order = np.argsort(x, kind="mergesort")
    x = x[order]
    y = y[order]
    n = len(y)
    if n < 2 * min_samples_leaf:
        return 0

    csum = np.cumsum(y)
    csum2 = np.cumsum(y * y)
    total_sum = csum[-1]
    total_sum2 = csum2[-1]
    best_sse = np.inf
    best_direction = 0

    for cut in range(min_samples_leaf, n - min_samples_leaf + 1):
        if cut >= n or x[cut - 1] >= x[cut]:
            continue
        n_left = cut
        n_right = n - cut
        sum_left = csum[cut - 1]
        sum_right = total_sum - sum_left
        mean_left = sum_left / n_left
        mean_right = sum_right / n_right
        sse_left = csum2[cut - 1] - n_left * mean_left**2
        sse_right = total_sum2 - csum2[cut - 1] - n_right * mean_right**2
        sse = sse_left + sse_right
        if sse < best_sse - EPS:
            best_sse = float(sse)
            difference = mean_right - mean_left
            if difference > EPS:
                best_direction = 1
            elif difference < -EPS:
                best_direction = -1
            else:
                best_direction = 0
    return best_direction


def deletion_stability(
    X,
    y,
    n_resamples=500,
    remove_fraction=0.10,
    min_samples_leaf=10,
    seed=1,
):
    """Return modal stump directions and stability proportions for all predictors.

    Replicates with no admissible nonzero orientation are omitted from the
    denominator.  Consequently, whenever at least one valid orientation is
    observed, the reported stability lies in [1/2, 1].
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError("X and y have incompatible shapes")
    if not 0.0 < remove_fraction < 1.0:
        raise ValueError("remove_fraction must lie in (0, 1)")

    n, p = X.shape
    keep = int(round(n * (1.0 - remove_fraction)))
    if keep < 2 * min_samples_leaf:
        raise ValueError("too few observations remain for an admissible stump")

    rng = np.random.default_rng(seed)
    votes = np.zeros((n_resamples, p), dtype=int)
    for b in range(n_resamples):
        idx = rng.choice(n, size=keep, replace=False)
        for j in range(p):
            votes[b, j] = best_stump_direction(X[idx, j], y[idx], min_samples_leaf)

    directions = np.zeros(p, dtype=int)
    stability = np.full(p, np.nan, dtype=float)
    for j in range(p):
        plus = int(np.sum(votes[:, j] == 1))
        minus = int(np.sum(votes[:, j] == -1))
        valid = plus + minus
        if plus > minus:
            directions[j] = 1
        elif minus > plus:
            directions[j] = -1
        if valid > 0:
            stability[j] = max(plus, minus) / valid
    return directions, stability
