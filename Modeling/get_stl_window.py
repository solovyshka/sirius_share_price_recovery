import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import STL
from tqdm import tqdm

def rolling_stl(data: pd.Series, seasonal_periods=15, window_size=240):
    """
    Скользящее STL-разложение.
    """

    trend = []
    seasonal = []
    resid = []

    index = data.index

    for t in tqdm(range(len(data))):
        if t < window_size:
            trend.append(np.nan)
            seasonal.append(np.nan)
            resid.append(np.nan)
            continue

        series_window = data.iloc[t-window_size:t]

        try:
            stl = STL(series_window, period=seasonal_periods, robust=True)
            result = stl.fit()

            trend.append(result.trend.iloc[-1])
            seasonal.append(result.seasonal.iloc[-1])
            resid.append(result.resid.iloc[-1])

        except Exception as e:
            trend.append(np.nan)
            seasonal.append(np.nan)
            resid.append(np.nan)

    return pd.DataFrame({
        "trend": trend,
        "seasonal": seasonal,
        "residual": resid
    }, index=index)


if __name__ == "__main__":
    data = pd.read_csv('DataBase/CLOSE.csv')
    data['datetime'] = pd.to_datetime(data['timestamp'], unit='s')

    data = pd.Series(np.array(data['SBER']), index=data['datetime'], name='close')

    components_stl = rolling_stl(data, seasonal_periods=15)

    components_stl.to_csv("components_stl.csv")