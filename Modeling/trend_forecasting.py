from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable

from Modeling.lin_model_trend import (
    fit_exponential_model,
    fit_linear_model,
    fit_logistic_model,
)


def _select_training_window(trend: pd.Series, train_window: int | None) -> pd.Series:
    """
    Возвращает очищенный ряд, ограниченный последними train_window точками.
    """
    if train_window is not None and train_window <= 0:
        raise ValueError("train_window должен быть положительным или None.")
    trend_clean = trend.dropna()
    if train_window is None:
        return trend_clean
    return trend_clean.tail(train_window)


def _anchor_forecast_to_last(
    trend: pd.Series, forecast_values: np.ndarray
) -> np.ndarray:
    """
    Сдвигает прогноз тренда так, чтобы первая точка совпала с последним наблюдением.
    Это убирает скачок уровня при экстраполяции глобальной регрессии.
    """
    if len(forecast_values) == 0:
        return forecast_values

    trend_clean = trend.dropna()
    if len(trend_clean) == 0:
        return forecast_values

    shift = trend_clean.iloc[-1] - forecast_values[0]
    return forecast_values + shift


def _trend_forecast_last_slope(
    trend: pd.Series, steps: int, train_window: int | None = None
) -> pd.Series:
    """
    Линейная экстраполяция по последнему среднему наклону тренда.
    """
    trend_clean = _select_training_window(trend, train_window)
    if len(trend_clean) == 0:
        raise ValueError("Нет данных тренда для прогноза.")

    if len(trend_clean) > 1:
        slope = trend_clean.diff().tail(min(5, len(trend_clean) - 1)).mean()
    else:
        slope = 0.0
    start = trend_clean.iloc[-1]

    values = [start + slope * (i + 1) for i in range(steps)]
    return pd.Series(values, name="trend_forecast")


def _trend_forecast_linear_reg(
    trend: pd.Series, steps: int, train_window: int | None = None
) -> pd.Series:
    """
    Линейная регрессия из lin_model_trend (polyfit), экстраполяция вперед.
    """
    trend_clean = _select_training_window(trend, train_window)
    if len(trend_clean) < 2:
        return _trend_forecast_last_slope(trend, steps, train_window=train_window)

    t = np.linspace(0.0, 1.0, len(trend_clean))
    y = trend_clean.to_numpy(dtype=float)
    try:
        a, b, _, _ = fit_linear_model(t, y)
        step = 1.0 / max(len(trend_clean) - 1, 1)
        future_t = np.linspace(1.0 + step, 1.0 + step * steps, steps)
        future_pred = a * future_t + b
        future_pred = _anchor_forecast_to_last(trend_clean, future_pred)
        return pd.Series(future_pred, name="trend_forecast")
    except Exception:
        return _trend_forecast_last_slope(trend, steps, train_window=train_window)


def _trend_forecast_exponential(
    trend: pd.Series, steps: int, train_window: int | None = None
) -> pd.Series:
    """Экспоненциальная подгонка + прогноз из lin_model_trend."""
    trend_clean = _select_training_window(trend, train_window)
    if len(trend_clean) < 3:
        return _trend_forecast_last_slope(trend, steps, train_window=train_window)

    t = np.linspace(0.0, 1.0, len(trend_clean))
    y = trend_clean.to_numpy(dtype=float)
    try:
        a, b, _, _ = fit_exponential_model(t, y)
        step = 1.0 / max(len(trend_clean) - 1, 1)
        future_t = np.linspace(1.0 + step, 1.0 + step * steps, steps)
        future_pred = a * np.exp(b * future_t)
        future_pred = _anchor_forecast_to_last(trend_clean, future_pred)
        return pd.Series(future_pred, name="trend_forecast")
    except Exception:
        return trend_forecast_last_slope(trend, steps, train_window=train_window)


def _trend_forecast_logistic(
    trend: pd.Series, steps: int, train_window: int | None = None
) -> pd.Series:
    """
    Логистическая кривая из lin_model_trend. При ошибке откат на last_slope.
    """
    trend_clean = _select_training_window(trend, train_window)
    if len(trend_clean) < 5:
        return trend_forecast_last_slope(trend, steps, train_window=train_window)

    t = np.linspace(0.0, 1.0, len(trend_clean))
    y = trend_clean.to_numpy(dtype=float)

    try:
        L, k, t0, _, _ = fit_logistic_model(t, y)
        step = 1.0 / max(len(trend_clean) - 1, 1)
        future_t = np.linspace(1.0 + step, 1.0 + step * steps, steps)
        future_pred = L / (1 + np.exp(-k * (future_t - t0)))
        future_pred = _anchor_forecast_to_last(trend_clean, future_pred)
        return pd.Series(future_pred, name="trend_forecast")
    except Exception:
        return _trend_forecast_last_slope(trend, steps, train_window=train_window)


def _bind_train_window(
    forecaster: Callable[[pd.Series, int, int | None], pd.Series],
    train_window: int | None,
) -> Callable[[pd.Series, int, int | None], pd.Series]:
    """Фиксирует train_window, но позволяет его переопределить при вызове."""
    if train_window is not None and train_window <= 0:
        raise ValueError("train_window должен быть положительным или None.")

    def _wrapped(
        trend: pd.Series, steps: int, train_window_override: int | None = None
    ) -> pd.Series:
        window = (
            train_window if train_window_override is None else train_window_override
        )
        return forecaster(trend, steps, train_window=window)

    return _wrapped


def trend_forecast_last_slope(
    train_window: int | None = None,
) -> Callable[[pd.Series, int, int | None], pd.Series]:
    return _bind_train_window(_trend_forecast_last_slope, train_window)


def trend_forecast_linear_reg(
    train_window: int | None = None,
) -> Callable[[pd.Series, int, int | None], pd.Series]:
    return _bind_train_window(_trend_forecast_linear_reg, train_window)


def trend_forecast_exponential(
    train_window: int | None = None,
) -> Callable[[pd.Series, int, int | None], pd.Series]:
    return _bind_train_window(_trend_forecast_exponential, train_window)


def trend_forecast_logistic(
    train_window: int | None = None,
) -> Callable[[pd.Series, int, int | None], pd.Series]:
    return _bind_train_window(_trend_forecast_logistic, train_window)
