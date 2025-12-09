from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

from Modeling.forecast_types import ForecastModel


class ArimaResidualModel:
    """Обычный ARIMA для прогноза остатка."""

    def __init__(self, order: tuple[int, int, int] = (1, 0, 0)):
        """
        :param order: параметры (p, d, q) ARIMA для моделирования остатка
        """
        self.order = order
        self._fitted = None

    def fit(
        self, y: pd.Series, exog: Optional[pd.DataFrame] = None
    ) -> ArimaResidualModel:
        """
        :param y: ряд остатка для обучения
        :param exog: экзогенные признаки, синхронные с рядом y
        """
        y_fit = y.reset_index(drop=True)
        exog_fit = exog.reset_index(drop=True) if exog is not None else None
        self._fitted = ARIMA(y_fit, order=self.order, exog=exog_fit).fit()
        return self

    def predict(
        self, steps: int, exog_future: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        :param steps: длина прогноза остатка
        :param exog_future: будущие значения экзогенных признаков
        """
        if self._fitted is None:
            raise RuntimeError("Сначала вызовите fit.")

        exog_pred = (
            exog_future.reset_index(drop=True) if exog_future is not None else None
        )
        forecast = self._fitted.forecast(steps=steps, exog=exog_pred)
        return pd.Series(forecast, name="residual_forecast")


class AutoArimaResidualModel:
    """Автовыбор параметров ARIMA через pmdarima.auto_arima для прогноза остатка."""

    def __init__(
        self,
        seasonal: bool = False,
        m: int = 1,
        max_p: int = 3,
        max_d: int = 2,
        max_q: int = 3,
        **kwargs,
    ):
        """
        :param seasonal: использовать ли сезонную ARIMA
        :param m: сезонный период
        :param max_p/max_d/max_q: верхние границы поиска по p,d,q
        :param kwargs: доп. параметры auto_arima
        """
        self.seasonal = seasonal
        self.m = m
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.kwargs = kwargs
        self._fitted = None

    def fit(
        self, y: pd.Series, exog: Optional[pd.DataFrame] = None
    ) -> AutoArimaResidualModel:
        """
        :param y: ряд остатка для обучения
        :param exog: экзогенные признаки для авто-ARIMA
        """
        y_fit = y.reset_index(drop=True)
        exog_fit = exog.reset_index(drop=True) if exog is not None else None
        self._fitted = auto_arima(
            y_fit,
            exogenous=exog_fit,
            seasonal=self.seasonal,
            m=self.m,
            max_p=self.max_p,
            max_d=self.max_d,
            max_q=self.max_q,
            **self.kwargs,
        )
        return self

    def predict(
        self, steps: int, exog_future: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        :param steps: длина прогноза остатка
        :param exog_future: будущие экзогенные значения для прогноза
        """
        if self._fitted is None:
            raise RuntimeError("Сначала вызовите fit.")

        exog_pred = (
            exog_future.reset_index(drop=True) if exog_future is not None else None
        )
        forecast = self._fitted.predict(n_periods=steps, X=exog_pred)
        return pd.Series(forecast, name="residual_forecast")
