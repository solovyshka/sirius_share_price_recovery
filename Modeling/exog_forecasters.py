from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Callable
import matplotlib

try:
    import matplotlib.backends.backend_gtk4agg  # noqa: F401
except Exception:
    matplotlib.use("TkAgg")
from pmdarima import auto_arima

from Modeling.forecast_types import ExogForecastModel


class LastValueExogForecaster:
    """
    Простейший вариант прогноза экзогенных признаков:
    повторяем последнюю строку exog на весь горизонт.
    """

    def __init__(self):
        self._last: Optional[pd.Series] = None

    def fit(self, exog: pd.DataFrame) -> LastValueExogForecaster:
        """
        :param exog: история экзогенных признаков, из которой берётся последняя строка
        """
        if exog is None or exog.empty:
            raise ValueError("exog пуст, нечего обучать в LastValueExogForecaster.")
        self._last = exog.iloc[-1]
        return self

    def predict(self, steps: int) -> pd.DataFrame:
        """
        :param steps: сколько будущих шагов повторять последнюю строку exog
        """
        if self._last is None:
            raise RuntimeError("Сначала вызовите fit().")
        if steps <= 0:
            raise ValueError("steps должен быть > 0.")

        data = np.tile(self._last.to_numpy(dtype=float), (steps, 1))
        return pd.DataFrame(data, columns=self._last.index)


@dataclass
class PerColumnAutoArimaExogForecaster:
    """
    Строит auto_arima по каждому экзогенному признаку отдельно
    и прогнозирует их вперёд по одному и тому же горизонту.
    """

    seasonal: bool = False
    m: int = 1
    max_p: int = 3
    max_d: int = 2
    max_q: int = 3
    kwargs: Dict[str, Any] = field(default_factory=dict)

    _models: Dict[str, Any] = field(init=False, default_factory=dict)
    _columns: list[str] = field(init=False, default_factory=list)
    _last_values: Dict[str, float] = field(init=False, default_factory=dict)

    def fit(self, exog: pd.DataFrame) -> PerColumnAutoArimaExogForecaster:
        """
        :param exog: исторические экзогенные признаки для подбора ARIMA по каждому столбцу
        """
        if exog is None or exog.empty:
            raise ValueError(
                "exog пуст, нечего обучать в PerColumnAutoArimaExogForecaster."
            )

        exog_clean = exog.astype(float)
        self._columns = list(exog_clean.columns)
        self._models = {}
        self._last_values = {}

        for col in self._columns:
            series = exog_clean[col]
            clean_series = series.dropna()
            if not clean_series.empty:
                self._last_values[col] = float(clean_series.iloc[-1])
            else:
                self._last_values[col] = 0.0

            if series.nunique(dropna=True) <= 1:
                continue

            model = auto_arima(
                series,
                seasonal=self.seasonal,
                m=self.m,
                max_p=self.max_p,
                max_d=self.max_d,
                max_q=self.max_q,
                **self.kwargs,
            )
            self._models[col] = model

        return self

    def predict(self, steps: int) -> pd.DataFrame:
        """
        :param steps: сколько шагов прогнозируем для каждого экзогенного столбца
        """
        if steps <= 0:
            raise ValueError("steps должен быть > 0.")
        if not self._columns:
            raise RuntimeError("Сначала вызовите fit().")

        data: Dict[str, np.ndarray] = {}

        for col in self._columns:
            model = self._models.get(col)
            if model is None:
                last_val = self._last_values.get(col, 0.0)
                data[col] = np.full(steps, last_val, dtype=float)
            else:
                fc = model.predict(n_periods=steps)
                data[col] = np.asarray(fc, dtype=float)

        df = pd.DataFrame(data)
        df = df[self._columns]
        return df


class SeasonalNaiveExogForecaster:
    """
    Сезонный наивный прогноз экзогенных признаков:
    повторяем последнюю наблюдённую сезонную фигуру длиной `period`.
    """

    def __init__(self, period: int):
        """
        :param period: длина повторяющегося сезонного паттерна экзогенных признаков
        """
        if period <= 0:
            raise ValueError("period должен быть > 0")
        self.period = period
        self._pattern: Optional[pd.DataFrame] = None

    def fit(self, exog: pd.DataFrame) -> SeasonalNaiveExogForecaster:
        """
        :param exog: исторические экзогенные значения для извлечения последнего сезонного паттерна
        """
        if exog is None or exog.empty:
            raise ValueError("exog пуст, нечего обучать в SeasonalNaiveExogForecaster.")

        exog_numeric = exog.astype(float)
        window = min(self.period, len(exog_numeric))
        self._pattern = exog_numeric.iloc[-window:].reset_index(drop=True)
        return self

    def predict(self, steps: int) -> pd.DataFrame:
        """
        :param steps: горизонт, на который повторяется сохранённый сезонный паттерн
        """
        if self._pattern is None:
            raise RuntimeError("Сначала вызовите fit().")
        if steps <= 0:
            raise ValueError("steps должен быть > 0.")

        repeats = int(np.ceil(steps / len(self._pattern)))
        pattern_rep = pd.concat([self._pattern] * repeats, ignore_index=True)
        return pattern_rep.iloc[:steps].reset_index(drop=True)


@dataclass
class PerColumnLastSlopeExogForecaster:
    """
    Локальный линейный тренд экзогенных признаков:
    для каждого столбца берём средний наклон по последним `window` приростам
    и экстраполируем вперёд.
    """

    window: int = 5

    _last_values: Dict[str, float] = field(init=False, default_factory=dict)
    _slopes: Dict[str, float] = field(init=False, default_factory=dict)
    _columns: list[str] = field(init=False, default_factory=list)

    def fit(self, exog: pd.DataFrame) -> PerColumnLastSlopeExogForecaster:
        """
        :param exog: исторические экзогенные значения для оценки последнего значения и наклона по каждому столбцу
        """
        if exog is None or exog.empty:
            raise ValueError(
                "exog пуст, нечего обучать в PerColumnLastSlopeExogForecaster."
            )

        exog_numeric = exog.astype(float)
        self._columns = list(exog_numeric.columns)
        self._last_values = {}
        self._slopes = {}

        for col in self._columns:
            series = exog_numeric[col].dropna()
            if series.empty:
                self._last_values[col] = 0.0
                self._slopes[col] = 0.0
                continue

            self._last_values[col] = float(series.iloc[-1])
            if len(series) > 1:
                diffs = series.diff().dropna().tail(min(self.window, len(series) - 1))
                slope = float(diffs.mean())
            else:
                slope = 0.0
            self._slopes[col] = slope

        return self

    def predict(self, steps: int) -> pd.DataFrame:
        """
        :param steps: на сколько периодов вперёд экстраполировать каждый экзогенный столбец
        """
        if steps <= 0:
            raise ValueError("steps должен быть > 0.")
        if not self._columns:
            raise RuntimeError("Сначала вызовите fit().")

        data: Dict[str, np.ndarray] = {}
        steps = np.arange(1, steps + 1, dtype=float)

        for col in self._columns:
            last = self._last_values.get(col, 0.0)
            slope = self._slopes.get(col, 0.0)
            data[col] = last + slope * steps

        return pd.DataFrame(data)


@dataclass
class PerColumnLinearRegExogForecaster:
    """
    Линейная регрессия exog(t) ~ a*t + b для каждого признака.
    В качестве времени используется числовой индекс 0..N-1.
    """

    _coeffs: Dict[str, tuple[float, float]] = field(init=False, default_factory=dict)
    _n_obs: int = field(init=False, default=0)
    _columns: list[str] = field(init=False, default_factory=list)

    def fit(self, exog: pd.DataFrame) -> PerColumnLinearRegExogForecaster:
        """
        :param exog: экзогенный датасет, по которому строится линейная регрессия индекс->значение
        """
        if exog is None or exog.empty:
            raise ValueError(
                "exog пуст, нечего обучать в PerColumnLinearRegExogForecaster."
            )

        exog_numeric = exog.astype(float)
        self._n_obs = len(exog_numeric)
        self._columns = list(exog_numeric.columns)
        self._coeffs = {}

        t = np.arange(self._n_obs, dtype=float)

        for col in self._columns:
            y = exog_numeric[col].to_numpy(dtype=float)
            mask = np.isfinite(y)
            t_c = t[mask]
            y_c = y[mask]

            if y_c.size == 0:
                self._coeffs[col] = (0.0, 0.0)
                continue
            if y_c.size == 1:
                self._coeffs[col] = (0.0, float(y_c[0]))
                continue

            a, b = np.polyfit(t_c, y_c, 1)
            last_t = t_c[-1]
            last_y = y_c[-1]
            last_hat = a * last_t + b
            b = b + (last_y - last_hat)
            self._coeffs[col] = (float(a), float(b))

        return self

    def predict(self, steps: int) -> pd.DataFrame:
        """
        :param steps: горизонт, на который продолжается линейная регрессия по столбцам
        """
        if steps <= 0:
            raise ValueError("steps должен быть > 0.")
        if not self._columns:
            raise RuntimeError("Сначала вызовите fit().")

        start = self._n_obs
        t_future = np.arange(start, start + steps, dtype=float)
        data: Dict[str, np.ndarray] = {}

        for col in self._columns:
            a, b = self._coeffs.get(col, (0.0, 0.0))
            data[col] = a * t_future + b

        return pd.DataFrame(data)


@dataclass
class VarExogForecaster:
    """
    Мультивариантный VAR по всем экзогенным признакам.
    Учитывает перекрёстные зависимости между фичами.
    """

    maxlags: int = 5
    ic: Optional[str] = "aic"

    _model: Any = field(init=False, default=None)
    _last_obs: Optional[np.ndarray] = field(init=False, default=None)
    _columns: list[str] = field(init=False, default_factory=list)

    def fit(self, exog: pd.DataFrame) -> VarExogForecaster:
        """
        :param exog: многомерный ряд экзогенных признаков, по которому обучается VAR
        """
        if exog is None or exog.empty:
            raise ValueError("exog пуст, нечего обучать в VarExogForecaster.")

        exog_numeric = exog.astype(float).dropna()
        if len(exog_numeric) <= 2:
            raise ValueError("Слишком мало наблюдений для VAR.")

        self._columns = list(exog_numeric.columns)

        from statsmodels.tsa.api import VAR

        model = VAR(exog_numeric)

        if self.ic is not None:
            try:
                order_res = model.select_order(maxlags=self.maxlags)
                selected_lag = getattr(order_res, self.ic)
                if selected_lag is None or selected_lag < 1:
                    selected_lag = 1
            except Exception:
                selected_lag = min(1, self.maxlags)
        else:
            selected_lag = self.maxlags

        self._model = model.fit(selected_lag)
        k_ar = getattr(self._model, "k_ar", selected_lag)
        endog_source = getattr(self._model, "y", None)
        if endog_source is None and hasattr(self._model, "model"):
            endog_source = getattr(self._model.model, "endog", None)
        if endog_source is None:
            endog_source = exog_numeric.to_numpy()
        endog_source = np.asarray(endog_source, dtype=float)
        if len(endog_source) < k_ar:
            raise ValueError("Недостаточно точек для формирования стартового окна VAR.")
        self._last_obs = endog_source[-k_ar:]
        return self

    def predict(self, steps: int) -> pd.DataFrame:
        """
        :param steps: сколько будущих наблюдений сгенерировать из обученной VAR
        """
        if steps <= 0:
            raise ValueError("steps должен быть > 0.")
        if self._model is None:
            raise RuntimeError("Сначала вызовите fit().")
        if self._last_obs is None:
            raise RuntimeError("Нет стартовых наблюдений для прогноза VAR.")

        fc = self._model.forecast(y=self._last_obs, steps=steps)
        df = pd.DataFrame(fc, columns=self._columns)
        return df


@dataclass
class PerColumnExpSmoothingExogForecaster:
    """
    Экспоненциальное сглаживание (SimpleExpSmoothing) для каждого экзогенного признака.
    """

    smoothing_level: Optional[float] = None
    optimized: bool = True

    _models: Dict[str, Any] = field(init=False, default_factory=dict)
    _columns: list[str] = field(init=False, default_factory=list)

    def fit(self, exog: pd.DataFrame) -> PerColumnExpSmoothingExogForecaster:
        """
        :param exog: исторические экзогенные данные, по которым обучается SimpleExpSmoothing для каждого столбца
        """
        if exog is None or exog.empty:
            raise ValueError(
                "exog пуст, нечего обучать в PerColumnExpSmoothingExogForecaster."
            )

        from statsmodels.tsa.holtwinters import SimpleExpSmoothing

        exog_numeric = exog.astype(float)
        self._columns = list(exog_numeric.columns)
        self._models = {}

        for col in self._columns:
            series = exog_numeric[col].dropna()
            if len(series) == 0:
                continue
            if len(series) == 1:
                self._models[col] = float(series.iloc[0])
                continue

            try:
                model = SimpleExpSmoothing(series).fit(
                    smoothing_level=self.smoothing_level,
                    optimized=self.optimized,
                )
                self._models[col] = model
            except Exception:
                self._models[col] = float(series.iloc[-1])

        return self

    def predict(self, steps: int) -> pd.DataFrame:
        """
        :param steps: сколько шагов вперёд прогнозировать с помощью обученных моделей сглаживания
        """
        if steps <= 0:
            raise ValueError("steps должен быть > 0.")
        if not self._columns:
            raise RuntimeError("Сначала вызовите fit().")

        data: Dict[str, np.ndarray] = {}
        for col in self._columns:
            model_or_val = self._models.get(col)
            if model_or_val is None:
                data[col] = np.zeros(steps, dtype=float)
            elif hasattr(model_or_val, "forecast"):
                fc = model_or_val.forecast(steps)
                data[col] = np.asarray(fc, dtype=float)
            else:
                data[col] = np.full(steps, float(model_or_val), dtype=float)

        return pd.DataFrame(data)


@dataclass
class EnsembleExogForecaster:
    """
    Ансамбль из нескольких моделей прогноза экзогенных признаков.
    Усредняет (по столбцам) результаты нескольких ExogForecastModel.
    """

    forecaster_factories: Sequence[Callable[[], ExogForecastModel]]
    weights: Optional[Sequence[float]] = None

    _models: list[ExogForecastModel] = field(init=False, default_factory=list)
    _columns: Optional[list[str]] = field(init=False, default=None)
    _weights: list[float] = field(init=False, default_factory=list)

    def fit(self, exog: pd.DataFrame) -> EnsembleExogForecaster:
        """
        :param exog: обучающий экзогенный датасет, передаваемый каждому базовому прогнозисту в ансамбле
        """
        if not self.forecaster_factories:
            raise ValueError("Список forecaster_factories пуст.")

        self._models = []
        for factory in self.forecaster_factories:
            model = factory()
            model.fit(exog)
            self._models.append(model)

        sample = self._models[0].predict(steps=1)
        self._columns = list(sample.columns)

        for m in self._models[1:]:
            cols = list(m.predict(steps=1).columns)
            if cols != self._columns:
                raise ValueError(
                    "Модели в ансамбле возвращают разные наборы колонок exog."
                )

        if self.weights is not None:
            if len(self.weights) != len(self._models):
                raise ValueError("Длина weights должна совпадать с числом моделей.")
            w = np.asarray(self.weights, dtype=float)
            w_sum = w.sum()
            if w_sum <= 0:
                raise ValueError("Сумма weights должна быть > 0.")
            self._weights = list(w / w_sum)
        else:
            self._weights = [1.0 / len(self._models)] * len(self._models)

        return self

    def predict(self, steps: int) -> pd.DataFrame:
        """
        :param steps: горизонт, на который усредняются прогнозы участников ансамбля
        """
        if steps <= 0:
            raise ValueError("steps должен быть > 0.")
        if not self._models:
            raise RuntimeError("Сначала вызовите fit().")

        preds = [m.predict(steps=steps) for m in self._models]
        preds = [df[self._columns].astype(float).reset_index(drop=True) for df in preds]

        stacked = np.stack([df.values for df in preds], axis=0)
        w = np.asarray(self._weights, dtype=float).reshape(-1, 1, 1)
        avg = (stacked * w).sum(axis=0)

        result = pd.DataFrame(avg, columns=self._columns)
        return result
