from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Sequence

from Collected_data.MOEX_connector.preprocessing import Feature
from Collected_data.MOEX_connector.add_technical import TechnicalIndicators
from Modeling.stationarize import FracDiffModel


def fracdiff_feature(
    diff_amt: float = 0.3,
    log_smooth: bool = True,
    column_name: str = "fracdiff",
    timestamp_unit: str = "s",
) -> Feature:
    """
    Обертка над FracDiffModel, чтобы добавить дробно-дифференцированный ряд как фичу.
    Используется как экзогенная переменная, а не как разложение для восстановления цены.
    """

    def build(df: pd.DataFrame, col: str) -> pd.DataFrame:
        if "timestamp" not in df.columns:
            raise ValueError("В DataFrame нет столбца 'timestamp' для fracdiff.")

        ts_index = pd.to_datetime(df["timestamp"], unit=timestamp_unit)
        series = pd.Series(df[col].astype(float).values, index=ts_index, name="close")

        model = FracDiffModel(
            data=series, log_smooth=log_smooth, diff_amt=diff_amt
        ).fit()
        if model is None:
            raise RuntimeError("FracDiffModel.fit() вернул None.")

        components = model.get_components(make_plot=False)
        fracdiff_series = components["residual"].rename(column_name)

        feature_df = fracdiff_series.to_frame()
        feature_df.index = df.index  # выравниваем с базовым df в preprocess
        return feature_df

    return build


def technical_indicator_feature(
    ticker: str, indicator_columns: Optional[Sequence[str]] = None
) -> Feature:
    """
    Обертка над TechnicalIndicators, чтобы добавить индикаторы как фичи для preprocess.
    Загружаются из DataBase/OPEN|HIGH|LOW|CLOSE|VOLUME.csv.
    """
    tech = TechnicalIndicators()

    def build(df: pd.DataFrame, col: str) -> pd.DataFrame:
        ind_df = tech.get_indicators(ticker)
        if ind_df is None:
            raise ValueError(f"Не удалось получить индикаторы для {ticker}")

        ind_df = ind_df.copy()

        if not np.issubdtype(ind_df.index.dtype, np.number):
            ind_df["timestamp"] = ind_df.index.view("int64") // 10**9
            ind_df = ind_df.set_index("timestamp")

        cols = indicator_columns or ind_df.columns
        missing = set(cols) - set(ind_df.columns)
        if missing:
            raise ValueError(f"Нет колонок {missing} в индикаторах для {ticker}")

        ind_df = ind_df[list(cols)]

        ind_df = ind_df.reindex(df.index)
        ind_df.index = df.index
        return ind_df

    return build
