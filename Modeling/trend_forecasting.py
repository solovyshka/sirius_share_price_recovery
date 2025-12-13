from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable

from Modeling.lin_model_trend import (
    fit_exponential_model,
    fit_linear_model,
    fit_logistic_model,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuralforecast import NeuralForecast  # noqa: F401
    from neuralforecast.models import PatchTST  # noqa: F401


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
        return _trend_forecast_last_slope(trend, steps, train_window=train_window)


def _trend_forecast_logistic(
    trend: pd.Series, steps: int, train_window: int | None = None
) -> pd.Series:
    """
    Логистическая кривая из lin_model_trend. При ошибке откат на last_slope.
    """
    trend_clean = _select_training_window(trend, train_window)
    if len(trend_clean) < 5:
        return _trend_forecast_last_slope(trend, steps, train_window=train_window)

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


def _trend_forecast_patchtst(
    trend: pd.Series, steps: int, train_window: int | None = None
) -> pd.Series:
    """
    Прогноз тренда с помощью PatchTST из библиотеки neuralforecast.
    """
    from neuralforecast import NeuralForecast
    from neuralforecast.models import PatchTST

    # Очищаем ряд и ограничиваем окно обучения
    trend_clean = _select_training_window(trend, train_window)
    if len(trend_clean) == 0:
        raise ValueError("Нет данных тренда для прогноза.")

    if steps <= 0:
        raise ValueError("steps должен быть положительным.")

    n = len(trend_clean)

    # Если данных мало, нет смысла гонять нейросетку — откат к last_slope
    MIN_TRAIN_LEN = 64
    if n <= steps or n < MIN_TRAIN_LEN:
        print(
            "PatchTST fallback: недостаточно данных для нейросети "
            f"(n={n}, steps={steps}, MIN_TRAIN_LEN={MIN_TRAIN_LEN}). "
            "Переключаемся на last slope."
        )
        return _trend_forecast_last_slope(trend, steps, train_window=train_window)

    # ---- Подбор окна для PatchTST ----
    # Требование: должно быть минимум одно обучающее окно длиной (input_size + h)
    MAX_INPUT_SIZE = 256
    max_feasible_input = max(n - steps, 1)
    if max_feasible_input < 8:
        # слишком мало точек для нормального обучения PatchTST
        print(
            "PatchTST fallback: слишком мало точек для нормального обучения "
            f"PatchTST (max_feasible_input={max_feasible_input}). "
            "Переключаемся на last slope."
        )
        return _trend_forecast_last_slope(trend, steps, train_window=train_window)

    input_size = min(MAX_INPUT_SIZE, max_feasible_input)
    patch_len = min(16, input_size)
    stride = max(1, patch_len // 2)

    # ---- Подготовка данных под NeuralForecast ----
    # NeuralForecast ожидает long-формат: [unique_id, ds, y].
    # Внутри используем ds = 0..n-1 — нам важен только порядок точек.
    y_values = trend_clean.to_numpy(dtype=float)
    train_df = pd.DataFrame(
        {
            "unique_id": "trend",
            "ds": np.arange(n, dtype=int),
            "y": y_values,
        }
    )

    batch_size = min(32, max(4, n // 4))
    val_size = min(steps, max(n // 5, 1))

    try:
        model = PatchTST(
            h=steps,
            input_size=input_size,
            patch_len=patch_len,
            stride=stride,
            encoder_layers=2,
            n_heads=4,
            hidden_size=64,
            linear_hidden_size=128,
            dropout=0.1,
            fc_dropout=0.1,
            head_dropout=0.0,
            attn_dropout=0.0,
            revin=True,
            revin_affine=False,
            revin_subtract_last=True,
            scaler_type="standard",
            batch_size=batch_size,
            max_steps=200,
            val_check_steps=50,
            early_stop_patience_steps=5,
            random_seed=1,
            alias="PatchTST_trend",
            enable_checkpointing=False,
            enable_progress_bar=True,
            logger=True,
        )

        # freq=1, т.к. ds — просто целочисленный индекс с шагом 1
        nf = NeuralForecast(models=[model], freq=1)
        nf.fit(df=train_df, val_size=val_size)

        forecasts_df = nf.predict()
        y_hat = forecasts_df[model.alias].to_numpy(dtype=float)

        if len(y_hat) < steps:
            # если почему-то предсказаний меньше, докидываем последнее значение
            last_val = y_hat[-1] if len(y_hat) > 0 else float(trend_clean.iloc[-1])
            y_hat = np.concatenate(
                [y_hat, np.full(steps - len(y_hat), last_val, dtype=float)]
            )
        elif len(y_hat) > steps:
            y_hat = y_hat[:steps]

        return pd.Series(y_hat, name="trend_forecast")

    except Exception as e:
        # Любая ошибка внутри нейросетки — тихий откат на last_slope,
        # чтобы не ронять весь пайплайн.
        print(
            "PatchTST fallback: ошибка внутри нейросети. " "Переход на last slope.\n",
            str(e),
        )
        return _trend_forecast_last_slope(trend, steps, train_window=train_window)


def trend_forecast_last_slope(
    train_window: int | None = None,
) -> Callable[[pd.Series, int, int | None], pd.Series]:
    """
    :param train_window: количество последних точек тренда, используемых для оценки наклона
    """
    return _bind_train_window(_trend_forecast_last_slope, train_window)


def trend_forecast_linear_reg(
    train_window: int | None = None,
) -> Callable[[pd.Series, int, int | None], pd.Series]:
    """
    :param train_window: количество точек тренда для подгонки линейной регрессии
    """
    return _bind_train_window(_trend_forecast_linear_reg, train_window)


def trend_forecast_exponential(
    train_window: int | None = None,
) -> Callable[[pd.Series, int, int | None], pd.Series]:
    """
    :param train_window: окно точек для подгонки экспоненциальной модели тренда
    """
    return _bind_train_window(_trend_forecast_exponential, train_window)


def trend_forecast_logistic(
    train_window: int | None = None,
) -> Callable[[pd.Series, int, int | None], pd.Series]:
    """
    :param train_window: окно точек для подгонки логистической кривой тренда
    """
    return _bind_train_window(_trend_forecast_logistic, train_window)


def trend_forecast_patchtst(
    train_window: int | None = None,
) -> Callable[[pd.Series, int, int | None], pd.Series]:
    """
    Прогноз тренда с помощью PatchTST из neuralforecast.

    :param train_window: количество последних точек тренда, используемых для обучения
    """
    if train_window is not None and train_window <= 0:
        raise ValueError("train_window должен быть положительным или None.")

    def _wrapped(
        trend: pd.Series,
        steps: int,
        train_window_override: int | None = None,
    ) -> pd.Series:
        window = (
            train_window if train_window_override is None else train_window_override
        )
        return _trend_forecast_patchtst(trend, steps, train_window=window)

    return _wrapped
