from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
from typing import Callable, Optional, Sequence, Tuple, Dict, Any
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)

try:
    import matplotlib.backends.backend_gtk4agg  # noqa: F401
except Exception:
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from Collected_data.MOEX_connector.preprocessing import Feature, preprocess
from Modeling.KPSS import StationarityTester
from Modeling.forecast_types import (
    Decomposition,
    DecompositionResult,
    ForecastModel,
    ForecastResult,
    ExogForecastModel,
)


class ForecastPipeline:
    """
    Простой пайплайн:
    1) сырые данные -> preprocess -> фичи;
    2) выбранное разложение -> тренд/сезон/остаток;
    3) выбранная модель -> прогноз остатка;
    4) сборка финального прогноза.
    """

    def __init__(
        self,
        ticker: str,
        decomposition_fn: Decomposition,
        model_factory: Callable[[], ForecastModel],
        feature_builders: Sequence[Feature] = (),
        timestamp_unit: str = "s",
        stationarity_alpha: float = 0.05,
        exog_forecast_factory: Optional[Callable[[], ExogForecastModel]] = None,
    ):
        """
        :param ticker: имя целевого столбца
        :param decomposition_fn: функция разложения ряда на компоненты
        :param model_factory: фабрика модели для прогноза остатка
        :param feature_builders: функции построения фич
        :param timestamp_unit: единица измерения timestamp
        :param stationarity_alpha: уровень значимости тестов стационарности
        :param exog_forecast_factory: фабрика прогноза экзогенных признаков для будущего периода
        """
        self.ticker = ticker
        self.decomposition_fn = decomposition_fn
        self.model_factory = model_factory
        self.feature_builders = feature_builders
        self.timestamp_unit = timestamp_unit
        self.stationarity_alpha = stationarity_alpha
        self.exog_forecast_factory = exog_forecast_factory

    def forecast(
        self,
        raw_df: pd.DataFrame,
        steps: int,
        future_exog: Optional[pd.DataFrame] = None,
        plot: bool = False,
        plot_last: Optional[int] = None,
    ) -> ForecastResult:
        """
        :param raw_df: исходный датафрейм с ценой и фичами
        :param steps: сколько шагов прогнозируем
        :param future_exog: будущие значения экзогенных признаков, если есть
        :param plot: рисовать ли график
        :param plot_last: сколько последних точек отображать
        """
        y, exog, prepared_dataset = self._prepare_dataset(raw_df)

        decomposition = self.decomposition_fn(y)
        residual = decomposition.residual.dropna()
        if len(residual) == 0:
            raise ValueError("Разложение не вернуло валидный остаток.")

        exog_train = (
            exog.loc[residual.index] if exog is not None and not exog.empty else None
        )

        model = self.model_factory()
        model.fit(residual, exog=exog_train)

        stationarity_tests = self._run_stationarity_tests(residual)

        exog_future = self._prepare_future_exog(
            exog_train=exog_train, future_exog=future_exog, steps=steps
        )

        residual_forecast = model.predict(steps=steps, exog_future=exog_future)
        trend_forecast, seasonal_forecast = decomposition.forecast_components(steps)

        future_index = self._make_future_index(y.index, steps)
        residual_forecast.index = future_index
        trend_forecast.index = future_index
        seasonal_forecast.index = future_index

        forecast = self._combine(
            trend_forecast, seasonal_forecast, residual_forecast, decomposition
        )

        if plot:
            self.plot_price_forecast(
                history=y,
                forecast=forecast,
                last_points=plot_last,
                title=f"{self.ticker} price forecast",
            )

        return ForecastResult(
            forecast=forecast,
            trend_forecast=trend_forecast,
            seasonal_forecast=seasonal_forecast,
            residual_forecast=residual_forecast,
            decomposition=decomposition,
            prepared_dataset=prepared_dataset,
            stationarity_tests=stationarity_tests,
        )

    def evaluate(
        self,
        raw_df: pd.DataFrame,
        test_size: int | float = 100,
        plot: bool = True,
        plot_last: Optional[int] = None,
    ) -> dict:
        """
        Простая валидация: train/test split по времени, обучение на train, прогноз на длину test.

        :param raw_df: исходный датафрейм с ценой и фичами
        :param test_size: количество точек (int) или доля (0..1) отложенной выборки
        :param plot: рисовать ли бэктест
        :param plot_last: сколько последних точек train показывать
        """
        y, exog, prepared_dataset = self._prepare_dataset(raw_df)
        if len(y) < 2:
            raise ValueError("Недостаточно данных для разбиения на train/test.")

        if isinstance(test_size, float):
            if not (0 < test_size < 1):
                raise ValueError("test_size как доля должен быть в (0, 1).")
            split_idx = int(len(y) * (1 - test_size))
        else:
            if not (0 < test_size < len(y)):
                raise ValueError("test_size должен быть в диапазоне (0, len(y)).")
            split_idx = len(y) - int(test_size)

        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        exog_train = exog.loc[y_train.index] if exog is not None else None
        exog_test = exog.loc[y_test.index] if exog is not None else None

        decomposition = self.decomposition_fn(y_train)
        residual = decomposition.residual.dropna()
        if len(residual) == 0:
            raise ValueError("Разложение не вернуло валидный остаток на train.")

        exog_train_aligned = (
            exog_train.loc[residual.index]
            if exog_train is not None and not exog_train.empty
            else None
        )

        model = self.model_factory()
        model.fit(residual, exog=exog_train_aligned)

        stationarity_tests = self._run_stationarity_tests(residual)

        steps = len(y_test)
        future_exog_input = (
            None if self.exog_forecast_factory is not None else exog_test
        )
        exog_future = self._prepare_future_exog(
            exog_train=exog_train_aligned,
            future_exog=future_exog_input,
            steps=steps,
        )

        residual_forecast = model.predict(steps=steps, exog_future=exog_future)
        trend_forecast, seasonal_forecast = decomposition.forecast_components(steps)

        residual_forecast.index = y_test.index
        trend_forecast.index = y_test.index
        seasonal_forecast.index = y_test.index

        forecast = self._combine(
            trend_forecast, seasonal_forecast, residual_forecast, decomposition
        )

        metrics = self._compute_metrics(y_test, forecast)

        if plot:
            self.plot_backtest(
                train=y_train,
                test=y_test,
                forecast=forecast,
                last_points=plot_last,
                title=f"{self.ticker} backtest (test={len(y_test)})",
            )

        # print(
        #     r2_score(
        #         np.log(y_test / y_test.shift(1))[1:],
        #         np.log(forecast / forecast.shift(1))[1:],
        #     )
        # )

        return {
            "forecast": forecast,
            "y_test": y_test,
            "y_train": y_train,
            "metrics": metrics,
            "decomposition": decomposition,
            "prepared_dataset": prepared_dataset,
            "stationarity_tests": stationarity_tests,
        }

    def _prepare_dataset(
        self, raw_df: pd.DataFrame
    ) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
        raw_columns = set(raw_df.columns)

        dataset = preprocess(
            raw_df=raw_df, ticker_name=self.ticker, features=self.feature_builders
        )

        if "timestamp" not in dataset.columns:
            raise ValueError("В исходных данных нет столбца timestamp.")

        feature_columns = [
            col
            for col in dataset.columns
            if col not in raw_columns and col != "timestamp"
        ]
        ordered_columns = ["timestamp", self.ticker, *feature_columns]
        dataset = dataset.loc[:, [c for c in ordered_columns if c in dataset.columns]]

        index = pd.to_datetime(dataset["timestamp"], unit=self.timestamp_unit)
        dataset = dataset.set_index(index)

        target = dataset[self.ticker].astype(float)
        exog = dataset.drop(columns=[self.ticker, "timestamp"], errors="ignore")

        aligned = pd.concat([target, exog], axis=1).dropna()
        target = aligned[self.ticker]
        exog = aligned.drop(columns=[self.ticker], errors="ignore")

        if target.empty:
            raise ValueError(
                "После очистки от NaN не осталось данных для модели. "
                "Убедитесь, что используемые фичи заполняются."
            )

        return target, exog, aligned

    @staticmethod
    def _combine(
        trend: pd.Series,
        seasonal: pd.Series,
        residual: pd.Series,
        decomposition: DecompositionResult,
    ) -> pd.Series:
        base = trend + seasonal + residual
        if decomposition.log_smooth:
            base = np.exp(base)
        return pd.Series(base.values, index=trend.index, name="forecast")

    @staticmethod
    def _make_future_index(index: pd.Index, steps: int) -> pd.Index:
        if steps <= 0:
            raise ValueError("steps должен быть > 0")

        if isinstance(index, pd.DatetimeIndex):
            freq = index.inferred_freq
            if freq is None and len(index) > 1:
                diffs = index.to_series().diff().dropna()
                if not diffs.empty:
                    freq = diffs.value_counts().idxmax()
            if freq is None:
                freq = pd.Timedelta(days=1)
            return pd.date_range(start=index[-1] + freq, periods=steps, freq=freq)

        if pd.api.types.is_numeric_dtype(index):
            if len(index) > 1:
                diffs = pd.Series(np.diff(index))
                step = diffs.value_counts().idxmax()
            else:
                step = 1

            last = index[-1]
            values = [last + step * (i + 1) for i in range(steps)]
            return pd.Index(values, name=index.name)

        return pd.RangeIndex(start=len(index), stop=len(index) + steps, name=index.name)

    def _prepare_future_exog(
        self,
        exog_train: Optional[pd.DataFrame],
        future_exog: Optional[pd.DataFrame],
        steps: int,
    ) -> Optional[pd.DataFrame]:
        """
        Приоритет такой:
        1) если явно передан future_exog — используем его;
        2) иначе, если задан exog_forecast_factory — строим модель и прогнозируем exog;
        3) иначе — повторяем последнюю строку (константный прогноз признаков).
        """
        if exog_train is None or exog_train.empty:
            return None

        if future_exog is not None:
            if len(future_exog) != steps:
                raise ValueError("future_exog должен иметь такую же длину, как steps.")

            missing_cols = set(exog_train.columns) - set(future_exog.columns)
            if missing_cols:
                raise ValueError(f"В future_exog нет колонок: {missing_cols}")

            return future_exog[exog_train.columns]

        if self.exog_forecast_factory is not None:
            forecaster = self.exog_forecast_factory()
            exog_numeric = exog_train.astype(float)

            forecaster.fit(exog_numeric)
            exog_future = forecaster.predict(steps=steps)

            missing_cols = set(exog_train.columns) - set(exog_future.columns)
            if missing_cols:
                raise ValueError(
                    f"Модель прогноза exog не вернула колонки: {missing_cols}"
                )

            exog_future = exog_future[exog_train.columns]
            exog_future = exog_future.reset_index(drop=True)
            return exog_future

        last_row = exog_train.iloc[[-1]]
        repeated = pd.concat([last_row] * steps, ignore_index=True)
        repeated.columns = exog_train.columns
        return repeated

    def _run_stationarity_tests(self, residual: pd.Series) -> dict:
        cleaned = residual.dropna()
        if cleaned.empty:
            return {"error": "Нет данных для стационарности (residual пуст)."}

        tester = StationarityTester(alpha=self.stationarity_alpha)

        results: dict = {}

        try:
            results["adf"] = tester.adf_test(cleaned)
        except Exception as exc:
            results["adf_error"] = str(exc)

        try:
            results["kpss"] = tester.kpss_test(cleaned)
        except Exception as exc:
            results["kpss_error"] = str(exc)

        try:
            results["pp"] = tester.pp_test(cleaned)
        except Exception as exc:
            results["pp_error"] = str(exc)

        try:
            results["summary"] = tester.get_stationarity_summary()
        except Exception as exc:
            results["summary_error"] = str(exc)

        return results

    @staticmethod
    def plot_price_forecast(
        history: pd.Series,
        forecast: pd.Series,
        last_points: Optional[int] = None,
        title: str = "Price forecast",
        figsize: tuple[int, int] = (12, 6),
    ):
        """
        Рисует историю + прогноз. Возвращает (fig, ax).

        :param history: исторический ценовой ряд
        :param forecast: прогнозируемый ряд
        :param last_points: длина хвоста истории для отображения
        :param title: заголовок графика
        :param figsize: размер фигуры matplotlib
        """
        history = history.sort_index()
        forecast = forecast.sort_index()

        if last_points is not None:
            history = history.iloc[-last_points:]

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(history.index, history.values, label="history")
        ax.plot(forecast.index, forecast.values, label="forecast", linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("time")
        ax.set_ylabel("price")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()
        return fig, ax

    @staticmethod
    def plot_backtest(
        train: pd.Series,
        test: pd.Series,
        forecast: pd.Series,
        last_points: Optional[int] = None,
        title: str = "Backtest",
        figsize: tuple[int, int] = (12, 6),
    ):
        """
        Рисует train/test/fc. Возвращает (fig, ax).

        :param train: обучающий сегмент
        :param test: фактические значения отложенной выборки
        :param forecast: прогноз на тестовом горизонте
        :param last_points: длина хвоста train для отображения
        :param title: заголовок графика
        :param figsize: размер фигуры matplotlib
        """
        train = train.sort_index()
        test = test.sort_index()
        forecast = forecast.sort_index()

        if last_points is not None:
            tail_idx = max(len(train) - last_points, 0)
            train = train.iloc[tail_idx:]

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(train.index, train.values, label="train")
        ax.plot(test.index, test.values, label="test (actual)")
        ax.plot(forecast.index, forecast.values, label="forecast", linestyle="--")
        ax.axvline(train.index[-1], color="gray", linestyle=":", alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("time")
        ax.set_ylabel("price")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()
        return fig, ax

    @staticmethod
    def _compute_metrics(actual: pd.Series, forecast: pd.Series) -> dict:
        actual, forecast = actual.align(forecast, join="inner")

        if len(actual) == 0:
            raise ValueError("Нет перекрытия индексов между actual и forecast.")

        nonzero = actual != 0
        if nonzero.any():
            mape = float(
                mean_absolute_percentage_error(
                    actual[nonzero],
                    forecast[nonzero],
                )
                * 100
            )
        else:
            mape = np.nan

        return {
            "mae": mean_absolute_error(actual, forecast),
            "rmse": root_mean_squared_error(actual, forecast),
            "mape": mape,
            "r2": r2_score(actual, forecast),
        }
