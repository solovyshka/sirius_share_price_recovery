from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Tuple

TrendForecaster = Callable[[pd.Series, int, int | None], pd.Series]


class ExogForecastModel(Protocol):
    """Интерфейс для моделей прогноза экзогенных признаков."""

    def fit(self, exog: pd.DataFrame) -> ExogForecastModel:
        """
        :param exog: исторические экзогенные признаки для обучения модели
        """
        ...

    def predict(self, steps: int) -> pd.DataFrame:
        """
        :param steps: горизонт прогноза для экзогенных признаков
        """
        ...


class ForecastModel(Protocol):
    """Интерфейс для моделей прогноза остатка."""

    def fit(self, y: pd.Series, exog: Optional[pd.DataFrame] = None) -> ForecastModel:
        """
        :param y: ряд остатка для обучения
        :param exog: экзогенные признаки, совпадающие по индексу с y
        """
        ...

    def predict(
        self, steps: int, exog_future: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        :param steps: длина прогноза
        :param exog_future: будущие значения экзогенных признаков
        """
        ...


@dataclass
class DecompositionResult:
    trend: pd.Series
    seasonal: pd.Series
    residual: pd.Series
    log_smooth: bool
    seasonal_periods: Optional[int] = None
    trend_forecaster: Optional[TrendForecaster] = None

    def forecast_components(self, steps: int) -> Tuple[pd.Series, pd.Series]:
        """
        Простейшая экстраполяция тренда и сезонности вперед.

        :param steps: горизонт, на который надо экстраполировать тренд и сезонность
        """
        if steps <= 0:
            raise ValueError("steps должен быть > 0")

        # Импорт здесь, чтобы избежать циклической зависимости.
        from Modeling.trend_forecasting import trend_forecast_last_slope

        trend_fn = self.trend_forecaster or trend_forecast_last_slope()
        trend_forecast = trend_fn(self.trend, steps)

        seasonal_clean = self.seasonal.dropna()
        if len(seasonal_clean) == 0:
            seasonal_forecast = pd.Series(
                np.zeros(steps, dtype=float), name="seasonal_forecast"
            )
        else:
            period = self.seasonal_periods or len(seasonal_clean)
            period = max(1, min(period, len(seasonal_clean)))
            values = seasonal_clean.iloc[-period:].to_numpy(dtype=float)
            repeats = int(np.ceil(steps / len(values)))
            seasonal_values = np.tile(values, repeats)[:steps]
            seasonal_forecast = pd.Series(seasonal_values, name="seasonal_forecast")

        return trend_forecast, seasonal_forecast


# Тип для функций разложения временного ряда.
Decomposition = Callable[[pd.Series], DecompositionResult]


@dataclass
class ForecastResult:
    forecast: pd.Series
    trend_forecast: pd.Series
    seasonal_forecast: pd.Series
    residual_forecast: pd.Series
    decomposition: DecompositionResult
    prepared_dataset: pd.DataFrame
    stationarity_tests: Optional[dict] = None
