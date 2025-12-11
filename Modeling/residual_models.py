from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Any
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from catboost import CatBoostRegressor

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


class CatBoostResidualModel:
    """
    Модель прогноза остатка на основе градиентного бустинга (CatBoost).

    Особенности:
    - использует лаги ряда остатков (n_lags);
    - опционально использует экзогенные признаки (exog);
    - многошаговый прогноз делается рекурсивно (каждый следующий шаг
      использует уже предсказанные значения как лаги).
    """

    def __init__(
        self,
        n_lags: int = 24,
        *,
        loss_function: str = "RMSE",
        depth: int = 6,
        learning_rate: float = 0.1,
        n_estimators: int = 500,
        random_seed: int = 42,
        verbose: bool = False,
        **catboost_params: Any,
    ):
        """
        Инициализация модели прогноза остатка на базе CatBoostRegressor.

        :param n_lags:
            Сколько лагов остатка использовать в качестве признаков.
            Должен быть > 0. Фактически используемое число лагов может быть
            уменьшено, если длина ряда y меньше n_lags + 1.

        :param loss_function:
            Функция потерь для CatBoostRegressor (например, "RMSE", "MAE" и т.д.).

        :param depth:
            Глубина деревьев в бустинге.

        :param learning_rate:
            Шаг обучения (eta) для градиентного бустинга.

        :param n_estimators:
            Количество деревьев (итераций бустинга).

        :param random_seed:
            Значение seed для воспроизводимости обучения CatBoost.

        :param verbose:
            Флаг вывода логов CatBoost во время обучения.
            Если False — обучение проходит без лишнего вывода.

        :param catboost_params:
            Любые дополнительные именованные параметры, которые будут
            напрямую переданы в CatBoostRegressor в момент инициализации.
            Могут переопределять значения параметров по умолчанию выше.
        """
        if n_lags <= 0:
            raise ValueError("n_lags должен быть > 0")

        self.n_lags = int(n_lags)

        self.model_params: dict[str, Any] = dict(
            loss_function=loss_function,
            depth=depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            random_seed=random_seed,
            verbose=verbose,
        )
        # Любые дополнительные параметры CatBoostRegressor
        self.model_params.update(catboost_params)

        self._model: Optional[CatBoostRegressor] = None
        self._fitted: bool = False

        self._n_lags_effective: Optional[int] = None
        self._feature_columns: Optional[list[str]] = None
        self._exog_columns: Optional[list[str]] = None
        self._last_values: Optional[np.ndarray] = None

    def _build_lag_features(
        self, y: pd.Series, exog: Optional[pd.DataFrame]
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Построение обучающей выборки: лаги y + (опционально) exog.

        Возвращает X, y_target, уже обрезанные по n_lags (без NaN).
        """
        y_fit = y.reset_index(drop=True).astype(float)
        n = len(y_fit)

        if n < 2:
            raise ValueError(
                "Слишком мало наблюдений для обучения CatBoostResidualModel (len(y) < 2)."
            )

        # Если данных мало, фактическое число лагов уменьшаем.
        self._n_lags_effective = min(self.n_lags, n - 1)

        # Лаговые признаки
        lag_data = {}
        for lag in range(1, self._n_lags_effective + 1):
            lag_data[f"lag_{lag}"] = y_fit.shift(lag)

        X_lags = pd.DataFrame(lag_data)

        if exog is not None:
            exog_fit = exog.reset_index(drop=True)
            if len(exog_fit) != n:
                raise ValueError(
                    "Длины y и exog должны совпадать: "
                    f"len(y)={n}, len(exog)={len(exog_fit)}."
                )
            # Экзогенные признаки привязываем к моменту цели (y_t)
            X_full = pd.concat([X_lags, exog_fit], axis=1)
            self._exog_columns = list(exog_fit.columns)
        else:
            X_full = X_lags
            self._exog_columns = None

        # Отбрасываем первые n_lags строк с NaN
        X_train = X_full.iloc[self._n_lags_effective :].reset_index(drop=True)
        y_train = y_fit.iloc[self._n_lags_effective :].reset_index(drop=True)

        self._feature_columns = list(X_train.columns)
        # Последние n_lags значений ряда для старта рекурсивного прогноза
        self._last_values = y_fit.iloc[-self._n_lags_effective :].to_numpy(dtype=float)

        return X_train, y_train

    def fit(
        self, y: pd.Series, exog: Optional[pd.DataFrame] = None
    ) -> CatBoostResidualModel:
        """
        :param y: ряд остатка для обучения
        :param exog: экзогенные признаки, совпадающие по индексу с y
        """
        X_train, y_train = self._build_lag_features(y, exog)

        self._model = CatBoostRegressor(**self.model_params)
        self._model.fit(X_train, y_train)

        self._fitted = True
        return self

    def predict(
        self, steps: int, exog_future: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        :param steps: длина прогноза
        :param exog_future: будущие значения экзогенных признаков.
                            Должны содержать те же столбцы, что и exog при fit.
        """
        if not self._fitted or self._model is None:
            raise RuntimeError("Сначала вызовите fit для CatBoostResidualModel.")

        if steps <= 0:
            raise ValueError("steps должен быть > 0")

        if self._n_lags_effective is None or self._last_values is None:
            raise RuntimeError(
                "Внутреннее состояние модели некорректно. " "Вызовите fit ещё раз."
            )

        # Подготовка будущих экзогенных признаков (если они использовались при обучении)
        if self._exog_columns is not None:
            if exog_future is None:
                raise ValueError(
                    "Модель обучалась с экзогенными признаками, "
                    "но exog_future не передан."
                )
            exog_future_fit = exog_future.reset_index(drop=True)
            if len(exog_future_fit) < steps:
                raise ValueError(
                    "exog_future должен иметь не меньше строк, чем steps: "
                    f"len(exog_future)={len(exog_future_fit)}, steps={steps}."
                )

            missing_cols = [
                c for c in self._exog_columns if c not in exog_future_fit.columns
            ]
            if missing_cols:
                raise ValueError(
                    "exog_future не содержит некоторые столбцы, использованные при обучении: "
                    f"{missing_cols}"
                )

            # Оставляем только нужные столбцы и обрезаем до steps
            exog_future_used = (
                exog_future_fit.loc[:, self._exog_columns]
                .iloc[:steps]
                .reset_index(drop=True)
            )
        else:
            # Экзогенные признаки при обучении не использовались – просто игнорируем вход exog_future.
            exog_future_used = None

        history = list(self._last_values.astype(float))
        preds: list[float] = []

        for i in range(steps):
            # Лаги из history: lag_1 = последнее значение, lag_2 = предпоследнее и т.д.
            feature_dict: dict[str, float] = {
                f"lag_{lag}": history[-lag]
                for lag in range(1, self._n_lags_effective + 1)
            }

            # Добавляем экзогенные признаки, если они используются
            if exog_future_used is not None:
                ex_row = exog_future_used.iloc[i].to_dict()
                feature_dict.update(ex_row)

            # Гарантируем тот же порядок признаков, что и при обучении
            X_row = pd.DataFrame([feature_dict], columns=self._feature_columns)

            pred = float(self._model.predict(X_row)[0])
            preds.append(pred)
            history.append(pred)

        return pd.Series(preds, name="residual_forecast")
