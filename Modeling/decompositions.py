from __future__ import annotations

import pandas as pd
from typing import Optional

from Modeling.forecast_types import DecompositionResult, TrendForecaster
from Modeling.stationarize import HolterWintersModel, STLModel


def _with_ordinal_index(series: pd.Series) -> tuple[pd.Series, pd.Index]:
    orig_index = series.index
    s = series.copy()
    s.index = pd.RangeIndex(len(s))
    return s, orig_index


def stl_decomposition(
    series: pd.Series,
    log_smooth: bool = True,
    seasonal_smooth: int = 23,
    seasonal_periods: int = 15,
    trend_forecaster: Optional[TrendForecaster] = None,
) -> DecompositionResult:
    """
    :param series: целевой временной ряд
    :param log_smooth: сглаживать ли логарифмом перед STL
    :param seasonal_smooth: окно сглаживания сезонности
    :param seasonal_periods: длина сезона
    :param trend_forecaster: функция для экстраполяции тренда вперёд
    """
    model = STLModel(
        data=series,
        log_smooth=log_smooth,
        seasonal_smooth=seasonal_smooth,
        seasonal_periods=seasonal_periods,
    ).fit()
    if model is None:
        raise RuntimeError("STLModel.fit() вернул None.")

    components = model.get_components(make_plot=False)
    return DecompositionResult(
        trend=components["trend"],
        seasonal=components["seasonal"],
        residual=components["residual"],
        log_smooth=log_smooth,
        seasonal_periods=seasonal_periods,
        trend_forecaster=trend_forecaster,
    )


def holt_winters_decomposition(
    series: pd.Series,
    log_smooth: bool = True,
    seasonal_periods: int = 15,
    trend_mode: str = "add",
    seasonal_mode: str = "add",
    trend_forecaster: Optional[TrendForecaster] = None,
) -> DecompositionResult:
    """
    :param series: целевой временной ряд
    :param log_smooth: сглаживать ли логарифмом перед Holt-Winters
    :param seasonal_periods: длина сезона
    :param trend_mode: тип тренда ("add"/"mul")
    :param seasonal_mode: тип сезонности ("add"/"mul")
    :param trend_forecaster: функция экстраполяции тренда
    """
    series_ordinal, orig_index = _with_ordinal_index(series)
    model = HolterWintersModel(
        data=series_ordinal,
        log_smooth=log_smooth,
        seasonal_periods=seasonal_periods,
        trend_mode=trend_mode,
        seasonal_mode=seasonal_mode,
    ).fit()
    components = model.get_components(make_plot=False)
    components = components.copy()
    components.index = orig_index
    return DecompositionResult(
        trend=components["trend"],
        seasonal=components["seasonal"],
        residual=components["residual"],
        log_smooth=log_smooth,
        seasonal_periods=seasonal_periods,
        trend_forecaster=trend_forecaster,
    )
