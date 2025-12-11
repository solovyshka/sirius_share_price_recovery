from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Callable
import matplotlib
from catboost import CatBoostRegressor

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


class CatBoostExogForecaster:
    """
    Модель прогноза экзогенных признаков на основе CatBoostRegressor.

    * Для каждого столбца exog обучается отдельная модель.
    * В качестве признаков используются лаги всех экзогенных столбцов.
    * Прогноз на несколько шагов вперёд строится рекурсивно.
    * Дополнительно (если индекс можно привести к DatetimeIndex) используются
      простые календарные признаки: час, день недели, день месяца, месяц и т.п.
    """

    def __init__(
        self,
        n_lags: int = 24,
        *,
        loss_function: str = "RMSE",
        depth: int = 6,
        learning_rate: float = 0.03,
        iterations: int = 500,
        verbose: bool = False,
        random_seed: Optional[int] = None,
        **catboost_params: Any,
    ):
        """
        :param n_lags: сколько лагов учитывать при построении признаков
        :param loss_function: функция потерь CatBoostRegressor
        :param depth: глубина деревьев
        :param learning_rate: шаг обучения
        :param iterations: число итераций бустинга
        :param verbose: выводить ли лог обучения CatBoost
        :param random_seed: seed для воспроизводимости, если нужен
        :param catboost_params: дополнительные параметры CatBoostRegressor
        """
        if n_lags <= 0:
            raise ValueError("n_lags должен быть > 0.")

        self.n_lags = n_lags
        self.model_params: dict[str, Any] = {
            "loss_function": loss_function,
            "depth": depth,
            "learning_rate": learning_rate,
            "iterations": iterations,
            "verbose": verbose,
        }
        if random_seed is not None:
            self.model_params["random_seed"] = random_seed
        if catboost_params:
            self.model_params.update(catboost_params)

        self._models: Dict[str, Any] = {}
        self._columns: list[str] = []
        self._fallback_last_values: Dict[str, float] = {}
        self._last_window: Optional[np.ndarray] = None
        self._used_lags: int = 0

        self._use_time_features: bool = False
        self._time_feature_names: list[str] = []
        self._last_timestamp: Optional[pd.Timestamp] = None
        self._time_delta: Optional[pd.Timedelta] = None

    def _extract_time_features(self, dt_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Строит простые календарные фичи из DatetimeIndex."""
        return pd.DataFrame(
            {
                "hour": dt_index.hour,
                "dayofweek": dt_index.dayofweek,
                "day": dt_index.day,
                "month": dt_index.month,
                "is_month_start": dt_index.is_month_start.astype(int),
                "is_month_end": dt_index.is_month_end.astype(int),
            },
            index=dt_index,
        )

    def fit(self, exog: pd.DataFrame) -> CatBoostExogForecaster:
        """
        :param exog: исторические значения экзогенных признаков (строки — моменты времени)
        """
        if exog is None or exog.empty:
            raise ValueError("exog пуст, нечего обучать в CatBoostExogForecaster.")

        exog_numeric = exog.astype(float)
        self._columns = list(exog_numeric.columns)
        n_samples = len(exog_numeric)

        if n_samples < 2:
            raise ValueError(
                "Слишком мало наблюдений для CatBoostExogForecaster (нужно >= 2 строк)."
            )

        # используем не больше n_lags, чем доступно наблюдений - 1
        self._used_lags = max(1, min(self.n_lags, n_samples - 1))

        # подготовка временных фич
        self._use_time_features = False
        time_features: Optional[pd.DataFrame] = None
        try:
            dt_index = pd.to_datetime(exog.index)
            if isinstance(dt_index, pd.DatetimeIndex):
                self._use_time_features = True
                time_features = self._extract_time_features(dt_index)
                self._time_feature_names = list(time_features.columns)

                # оценим шаг по времени (медиана разностей)
                diffs = dt_index.to_series().diff().dropna()
                if len(diffs) > 0:
                    self._time_delta = diffs.median()
                else:
                    self._time_delta = pd.Timedelta(hours=1)
                self._last_timestamp = dt_index[-1]
        except Exception:
            # если индекс не удаётся интерпретировать как даты — просто не используем time features
            self._use_time_features = False
            time_features = None
            self._time_feature_names = []
            self._last_timestamp = None
            self._time_delta = None

        values = exog_numeric.to_numpy(dtype=float)
        n_samples, n_features = values.shape

        # матрица лагов
        n_rows = n_samples - self._used_lags
        n_lag_features = n_features * self._used_lags
        n_time_features = (
            len(self._time_feature_names) if self._use_time_features else 0
        )

        X = np.zeros((n_rows, n_lag_features + n_time_features), dtype=float)

        for i in range(self._used_lags, n_samples):
            # окно последних self._used_lags наблюдений до момента i
            window = values[i - self._used_lags : i]  # (used_lags, n_features)
            # хотим порядок: [lag1_all_features, lag2_all_features, ...] где lag1 = t-1
            window_rev = window[::-1]  # от последнего к более старым
            lag_vec = window_rev.reshape(-1)

            if self._use_time_features and time_features is not None:
                tf_row = time_features.iloc[i].to_numpy(dtype=float)
                row = np.concatenate([lag_vec, tf_row])
            else:
                row = lag_vec

            X[i - self._used_lags] = row

        self._models = {}
        self._fallback_last_values = {}

        # запомним последнее окно значений для рекурсивного прогноза
        self._last_window = values[-self._used_lags :].copy()

        # обучаем отдельную модель для каждого столбца
        for col_idx, col_name in enumerate(self._columns):
            y_all = values[self._used_lags :, col_idx]

            # таргет должен быть конечным числом
            mask = np.isfinite(y_all)
            X_col = X[mask]
            y_col = y_all[mask]

            # запомним последнее наблюдение для фолбэка
            col_series = exog_numeric[col_name].dropna()
            if len(col_series) > 0:
                self._fallback_last_values[col_name] = float(col_series.iloc[-1])
            else:
                self._fallback_last_values[col_name] = 0.0

            if len(y_col) < 2:
                # слишком мало наблюдений — будет просто повторяться последнее значение
                self._models[col_name] = None
                continue

            model = CatBoostRegressor(**self.model_params)
            model.fit(X_col, y_col)
            self._models[col_name] = model

        return self

    def _make_time_feature_row(self, timestamp: pd.Timestamp) -> np.ndarray:
        """Временные фичи для одного будущего момента времени в том же порядке, что и при обучении."""
        if not self._use_time_features:
            return np.empty(0, dtype=float)

        data = {
            "hour": timestamp.hour,
            "dayofweek": timestamp.dayofweek,
            "day": timestamp.day,
            "month": timestamp.month,
            "is_month_start": int(timestamp.is_month_start),
            "is_month_end": int(timestamp.is_month_end),
        }
        return np.array(
            [float(data[name]) for name in self._time_feature_names],
            dtype=float,
        )

    def predict(self, steps: int) -> pd.DataFrame:
        """
        :param steps: горизонт, на который строится рекурсивный прогноз exog
        """
        if steps <= 0:
            raise ValueError("steps должен быть > 0.")
        if not self._columns:
            raise RuntimeError("Сначала вызовите fit().")

        # если ни один столбец не получил модель, просто повторим последние значения
        if all(m is None for m in self._models.values()):
            last_vals = np.array(
                [self._fallback_last_values.get(col, 0.0) for col in self._columns],
                dtype=float,
            )
            data = np.tile(last_vals, (steps, 1))
            return pd.DataFrame(data, columns=self._columns)

        if self._last_window is None or self._last_window.shape[0] < 1:
            raise RuntimeError("Нет последнего окна наблюдений для прогноза.")

        history = self._last_window.copy()  # (used_lags, n_features)
        n_features = history.shape[1]

        results = []

        current_timestamp = self._last_timestamp
        time_delta = (
            self._time_delta if self._time_delta is not None else pd.Timedelta(hours=1)
        )

        for _ in range(steps):
            # формируем вектор лаговых фич
            window_rev = history[::-1]
            lag_vec = window_rev.reshape(-1)

            if self._use_time_features and current_timestamp is not None:
                current_timestamp = current_timestamp + time_delta
                tf_row = self._make_time_feature_row(current_timestamp)
                x_row = np.concatenate([lag_vec, tf_row])
            else:
                x_row = lag_vec

            preds_row = np.zeros(n_features, dtype=float)
            for col_idx, col_name in enumerate(self._columns):
                model = self._models.get(col_name)
                if model is None:
                    preds_row[col_idx] = self._fallback_last_values.get(col_name, 0.0)
                else:
                    val = float(model.predict(x_row.reshape(1, -1))[0])
                    preds_row[col_idx] = val

            results.append(preds_row)
            # обновляем историю: добавляем новый ряд и отбрасываем самый старый
            history = np.vstack([history, preds_row])[1:, :]

        data = np.vstack(results)
        return pd.DataFrame(data, columns=self._columns)
