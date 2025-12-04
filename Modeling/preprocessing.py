import numpy as np
import pandas as pd
from typing import Sequence, Optional
from functools import reduce


def choose_alpha_with_mse(s: pd.Series) -> float:
    y = s.dropna().to_numpy(dtype=float)

    def mse_for_alpha(a: float) -> float:
        prev_y = y[0]
        mse_sum = 0.0
        for i in range(1, y.size):
            mse_sum += (prev_y - y[i]) ** 2
            prev_y = a * y[i] + (1.0 - a) * prev_y

        return mse_sum

    grid = np.linspace(0.01, 0.99, 99)
    mses = [mse_for_alpha(a) for a in grid]
    return grid[np.argmin(mses)]


def rolling_features(
    y: pd.Series | np.ndarray | Sequence[float],
    window: int,
    weights: Optional[Sequence[float]] = None,
    alpha: Optional[float] = None,
) -> pd.DataFrame:
    if isinstance(y, pd.Series):
        s = y.copy()
    else:
        s = pd.Series(y, name="y")

    if alpha is not None:
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be between 0 and 1 (exclusive)")
    elif window <= 100:
        alpha = 0.9
    else:
        alpha = choose_alpha_with_mse(s)

    roll = s.shift(1).rolling(window=window, min_periods=window)
    if weights is None:
        weights = np.arange(1, window + 1, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        if len(weights) != window:
            raise ValueError("len(weights) must equal window size")

    features = pd.DataFrame(index=s.index)
    features["mean"] = roll.mean()
    features["weighted_mean"] = roll.apply(
        lambda x: np.average(x, weights=weights[-len(x) :]), raw=True
    )
    features["exp_smooth"] = roll.apply(
        lambda x: reduce(lambda s, v: alpha * v + (1.0 - alpha) * s, x[1:], x[0]),
        raw=True,
    )
    features["median"] = roll.median()
    features["min"] = roll.min()
    features["max"] = roll.max()
    features["std"] = roll.std()
    return features
