import numpy as np
import pandas as pd
from typing import Sequence, Optional, Callable
from functools import reduce

Feature = Callable[[pd.DataFrame, str], pd.DataFrame]


def get_price_series(df: pd.DataFrame, ticker_name: str) -> pd.Series:
    assert "timestamp" in df.columns, "DataFrame must contain 'timestamp' column"

    if ticker_name not in df.columns:
        raise ValueError(
            f"DataFrame must contain price column {ticker_name}, "
            f"available columns: {list(df.columns)}"
        )

    return df[ticker_name].astype(float)


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
    ticker_name: str,
    window: int,
    weights: Optional[Sequence[float]] = None,
    alpha: Optional[float] = None,
) -> pd.DataFrame:
    s = get_price_series(df, ticker_name)

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
        lambda x: reduce(lambda s_, v: alpha * v + (1.0 - alpha) * s_, x[1:], x[0]),
        raw=True,
    )
    features["median"] = roll.median()
    features["min"] = roll.min()
    features["max"] = roll.max()
    features["std"] = roll.std()
    return features


def preprocess(
    raw_df: pd.DataFrame, ticker_name: str, features: Sequence[Feature]
) -> pd.DataFrame:
    """
    Usage:
    >>> df = preprocess(
    ...     df,
    ...     ticker_name="SBER",
    ...     features=[
    ...         lambda df_, col: rolling_features(df_, col, window=3),
    ...     ],
    ... )
    """

    if "timestamp" not in raw_df.columns:
        raise ValueError("Input DataFrame must contain 'timestamp' column")
    if not raw_df["timestamp"].is_unique:
        raise ValueError("Timestamps must be unique")
    if ticker_name not in raw_df.columns:
        raise ValueError(
            f"Input DataFrame must contain price column {ticker_name}, "
            f"available columns: {list(raw_df.columns)}"
        )

    base = raw_df.sort_values("timestamp").set_index("timestamp", drop=False)
    feat_dfs: list[pd.DataFrame] = []

    for fn in features:
        fdf = fn(base, ticker_name)
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
