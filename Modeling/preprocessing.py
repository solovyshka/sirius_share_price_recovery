import numpy as np
import pandas as pd
from typing import Sequence, Optional, Callable
from functools import reduce

Feature = Callable[[pd.DataFrame], pd.DataFrame]


def get_price_series(df: pd.DataFrame) -> pd.Series:
    assert "timestamp" in df.columns, "DataFrame must contain 'timestamp' column"

    price_cols = [col for col in df.columns if col != "timestamp"]
    if len(price_cols) != 1:
        raise ValueError(
            f"Expected exactly 1 non-'timestamp' column, got {len(price_cols)}: {price_cols}"
        )

    return df[price_cols[0]].astype(float)


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
    df: pd.DataFrame,
    window: int,
    weights: Optional[Sequence[float]] = None,
    alpha: Optional[float] = None,
) -> pd.DataFrame:
    s = get_price_series(df)

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


def preprocess(raw_df: pd.DataFrame, features: Sequence[Feature]) -> pd.DataFrame:
    """
    Usage:
    >>> df = preprocess(
    ...     df,
    ...     [
    ...         lambda df: rolling_features(df, window=3),
    ...     ],
    ... )
    """

    if "timestamp" not in raw_df.columns:
        raise ValueError("Input DataFrame must contain 'timestamp' column")
    if not raw_df["timestamp"].is_unique:
        raise ValueError("Timestamps must be unique")

    base = raw_df.sort_values("timestamp").set_index("timestamp", drop=False)
    feat_dfs: list[pd.DataFrame] = []

    for fn in features:
        fdf = fn(base)
        if len(fdf) != len(base):
            raise ValueError(
                f"Feature {getattr(fn, '__name__', fn)} returned DataFrame "
                f"of wrong length: {len(fdf)} != {len(base)}"
            )

        if not fdf.index.equals(base.index):
            fdf = fdf.copy()
            fdf.index = base.index

        feat_dfs.append(fdf)

    dataset = pd.concat([base, *feat_dfs], axis=1)
    dataset = dataset.loc[:, ~dataset.columns.duplicated(keep="first")]
    return dataset.reset_index(drop=True)
