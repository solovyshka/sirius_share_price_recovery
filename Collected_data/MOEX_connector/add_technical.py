import pandas as pd
import numpy as np

# import ta
# from ta import add_all_ta_features
# from ta.utils import dropna
from tqdm import tqdm

# from datetime import datetime
# import os

import warnings

warnings.filterwarnings("ignore")


class TechnicalIndicators:
    def __init__(self, data_folder="DataBase/"):
        self.data_folder = data_folder
        self.indicators_data = {}

    def get_available_tickers(self):
        try:
            close_df = pd.read_csv(f"{self.data_folder}CLOSE.csv", index_col=0)
            return close_df.columns.tolist()[1:]
        except:
            print("Не удалось загрузить список тикеров")
            return []

    def init_ticker(self, ticker):
        data = self.load_data(ticker)
        self.indicators_data[ticker] = self.calculate_all_indicators(data, ticker)

    def init_all_tickers(self):
        tickers = self.get_available_tickers()
        for ticker in tqdm(tickers):
            if ticker not in self.indicators_data:
                data = self.load_data(ticker)
                if data is not None and len(data) > 50:
                    indicators_df = self.calculate_all_indicators(data, ticker)
                    if indicators_df is not None:
                        self.indicators_data[ticker] = indicators_df

    def load_data(self, ticker):
        try:
            close = pd.read_csv(
                f"{self.data_folder}CLOSE.csv", index_col=1, parse_dates=True
            )[ticker].rename("close")
            high = pd.read_csv(
                f"{self.data_folder}HIGH.csv", index_col=1, parse_dates=True
            )[ticker].rename("high")
            low = pd.read_csv(
                f"{self.data_folder}LOW.csv", index_col=1, parse_dates=True
            )[ticker].rename("low")
            open_ = pd.read_csv(
                f"{self.data_folder}OPEN.csv", index_col=1, parse_dates=True
            )[ticker].rename("open")
            volume = pd.read_csv(
                f"{self.data_folder}VOLUME.csv", index_col=1, parse_dates=True
            )[ticker].rename("volume")

            data = pd.concat([open_, high, low, close, volume], axis=1)
            data.columns = ["open", "high", "low", "close", "volume"]

            return data
        except Exception as e:
            print(f"Ошибка при загрузке данных для {ticker}: {e}")
            return None

    def calculate_all_indicators(self, data, ticker):
        if data is None or len(data) == 0:
            return None

        df = data.copy()

        # 1. Trend
        df["SMA_20"] = df["close"].shift(1).rolling(window=20, min_periods=1).mean()
        df["SMA_50"] = df["close"].shift(1).rolling(window=50, min_periods=1).mean()
        df["SMA_200"] = df["close"].shift(1).rolling(window=200, min_periods=1).mean()

        df["EMA_12"] = df["close"].shift(1).ewm(span=12, adjust=False).mean()
        df["EMA_26"] = df["close"].shift(1).ewm(span=26, adjust=False).mean()

        # MACD
        df["MACD"] = df["EMA_12"] - df["EMA_26"]
        df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_diff"] = df["MACD"] - df["MACD_signal"]

        # Parabolic SAR
        def calc_psar(
            high, low, af_start=0.02, af_step=0.02, af_max=0.2, epsilon=1e-10
        ):
            psar = np.full(len(high), np.nan, dtype=float)
            trend = np.full(len(high), 0, dtype=int)
            af = np.full(len(high), af_start, dtype=float)
            ep = np.full(len(high), np.nan, dtype=float)

            psar[0] = low.iloc[0] if not np.isnan(low.iloc[0]) else low.iloc[1]
            trend[0] = 1  # bull
            ep[0] = high.iloc[0] if not np.isnan(high.iloc[0]) else high.iloc[1]

            for i in range(1, len(high)):
                prev_psar = psar[i - 1]
                prev_trend = trend[i - 1]
                prev_af = af[i - 1]
                prev_ep = ep[i - 1]

                current_low = low.iloc[i]
                current_high = high.iloc[i]

                if pd.isna(current_low) or pd.isna(current_high):
                    psar[i] = prev_psar
                    trend[i] = prev_trend
                    af[i] = prev_af
                    ep[i] = prev_ep
                    continue

                if prev_trend == 1:  # bull
                    new_psar = prev_psar + prev_af * (prev_ep - prev_psar)

                    if abs(new_psar) > 1e100:
                        new_psar = np.sign(new_psar) * 1e100

                    if current_low <= new_psar:  # reverse
                        trend[i] = -1
                        psar[i] = max(prev_ep, current_high)
                        ep[i] = current_low
                        af[i] = af_start
                    else:  # continue
                        trend[i] = 1
                        psar[i] = new_psar

                        if current_high > prev_ep:
                            ep[i] = current_high
                            af[i] = min(prev_af + af_step, af_max)
                        else:
                            ep[i] = prev_ep
                            af[i] = prev_af

                else:  # bear
                    new_psar = prev_psar - prev_af * (prev_ep - prev_psar)

                    if abs(new_psar) > 1e100:
                        new_psar = np.sign(new_psar) * 1e100

                    if current_high >= new_psar:  # reverse
                        trend[i] = 1
                        psar[i] = min(prev_ep, current_low)
                        ep[i] = current_high
                        af[i] = af_start
                    else:  # continue
                        trend[i] = -1
                        psar[i] = new_psar

                        if current_low < prev_ep:
                            ep[i] = current_low
                            af[i] = min(prev_af + af_step, af_max)
                        else:
                            ep[i] = prev_ep
                            af[i] = prev_af

            psar_series = pd.Series(psar, index=high.index, name="PSAR")
            trend_series = pd.Series(trend, index=high.index, name="PSAR_Trend")
            ep_series = pd.Series(ep, index=high.index, name="PSAR_EP")

            return psar_series, trend_series, ep_series

        def safe_psar_distance(close, psar, epsilon=1e-10):
            close_safe = close.copy()
            psar_safe = psar.copy()

            psar_safe = psar_safe.replace([np.inf, -np.inf], np.nan)

            psar_safe = psar_safe.interpolate(method="linear", limit_direction="both")

            min_positive = (
                close_safe[close_safe > 0].min() if (close_safe > 0).any() else epsilon
            )

            close_safe = close_safe.where(close_safe > 0, min_positive)

            distance = (close_safe - psar_safe) / close_safe * 100

            distance = np.clip(distance, -1000, 1000)

            return distance

        psar, trend, ep = calc_psar(df["high"], df["low"])
        df["PSAR"] = psar
        df["PSAR_trend"] = trend
        df["PSAR_ep"] = ep

        df["PSAR_above_price"] = (df["PSAR"] > df["close"]).astype(int)
        df["PSAR_distance"] = safe_psar_distance(df["close"], psar)

        # ADX
        def calc_adx(high, low, close, window=14):  # [adx, +di, -di, dx]
            n = len(high)

            prev_close = close.shift(1)

            tr1 = high - low
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()

            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            prev_high = high.shift(1)
            prev_low = low.shift(1)

            up_move = high - prev_high
            down_move = prev_low - low

            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

            plus_dm_series = pd.Series(plus_dm, index=high.index)
            minus_dm_series = pd.Series(minus_dm, index=high.index)

            smoothed_tr = np.full(n, np.nan)
            smoothed_plus_dm = np.full(n, np.nan)
            smoothed_minus_dm = np.full(n, np.nan)

            for i in range(window, n):
                start_idx = i - window + 1

                smoothed_tr[i] = true_range.iloc[start_idx : i + 1].sum()
                smoothed_plus_dm[i] = plus_dm_series.iloc[start_idx : i + 1].sum()
                smoothed_minus_dm[i] = minus_dm_series.iloc[start_idx : i + 1].sum()

            for i in range(
                window + 1, n
            ):  # Smoothed = Previous Smoothed - (Previous Smoothed/window) + Current
                smoothed_tr[i] = (
                    smoothed_tr[i - 1]
                    - (smoothed_tr[i - 1] / window)
                    + true_range.iloc[i]
                )
                smoothed_plus_dm[i] = (
                    smoothed_plus_dm[i - 1]
                    - (smoothed_plus_dm[i - 1] / window)
                    + plus_dm_series.iloc[i]
                )
                smoothed_minus_dm[i] = (
                    smoothed_minus_dm[i - 1]
                    - (smoothed_minus_dm[i - 1] / window)
                    + minus_dm_series.iloc[i]
                )

            plus_di = np.full(n, np.nan)
            minus_di = np.full(n, np.nan)

            for i in range(window, n):
                if smoothed_tr[i] != 0:
                    plus_di[i] = 100 * (smoothed_plus_dm[i] / smoothed_tr[i])
                    minus_di[i] = 100 * (smoothed_minus_dm[i] / smoothed_tr[i])
                else:
                    plus_di[i] = 0
                    minus_di[i] = 0

            dx = np.full(n, np.nan)

            for i in range(window, n):
                di_sum = plus_di[i] + minus_di[i]
                di_diff = abs(plus_di[i] - minus_di[i])

                if di_sum != 0:
                    dx[i] = 100 * (di_diff / di_sum)
                else:
                    dx[i] = 0

            adx = np.full(n, np.nan)

            for i in range(window * 2 - 1, n):
                if i == window * 2 - 1:
                    start_idx = window
                    end_idx = i
                    adx[i] = np.nanmean(dx[start_idx : end_idx + 1])
                else:
                    adx[i] = (adx[i - 1] * (window - 1) + dx[i]) / window

            adx_series = pd.Series(adx, index=high.index, name="ADX")
            plus_di_series = pd.Series(plus_di, index=high.index, name="Plus_DI")
            minus_di_series = pd.Series(minus_di, index=high.index, name="Minus_DI")
            dx_series = pd.Series(dx, index=high.index, name="DX")

            return adx_series, plus_di_series, minus_di_series, dx_series

        adx, plus_di, minus_di, dx = calc_adx(df["high"], df["low"], df["close"])
        df["ADX"] = adx
        df["PDI"] = plus_di
        df["MDI"] = minus_di
        df["DX"] = dx

        # 2. Momentum
        def calc_rsi(close, window=14):
            deltas = close.diff()
            gain = deltas.where(deltas > 0, 0)
            loss = -deltas.where(deltas < 0, 0)

            avg_gain = gain.shift(1).rolling(window=window, min_periods=1).mean()
            avg_loss = loss.shift(1).rolling(window=window, min_periods=1).mean()

            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            return rsi

        df["RSI"] = calc_rsi(df["close"], window=14)
        df["RSI_7"] = calc_rsi(df["close"], window=7)

        # Stochastic Oscillator
        def calc_stoch_osc(
            high, low, close, k_period=14, d_period=3, smooth_k=1, ma_type="sma"
        ):  # [%K, %D, raw_k, signal]
            n = len(close)

            raw_k = np.full(n, np.nan, dtype=float)
            percent_k = np.full(n, np.nan, dtype=float)
            percent_d = np.full(n, np.nan, dtype=float)
            stoch_signal = np.full(n, np.nan, dtype=float)

            for i in range(k_period - 1, n):
                start_idx = i - k_period + 1
                end_idx = i

                period_high = high.iloc[start_idx : end_idx + 1].max()
                period_low = low.iloc[start_idx : end_idx + 1].min()

                current_close = close.iloc[i]

                if period_high != period_low:
                    raw_k[i] = (
                        100 * (current_close - period_low) / (period_high - period_low)
                    )
                else:
                    raw_k[i] = 0.0

            if smooth_k > 1:
                for i in range(k_period + smooth_k - 2, n):
                    start_idx = i - smooth_k + 1
                    end_idx = i
                    percent_k[i] = np.mean(raw_k[start_idx : end_idx + 1])
            else:
                percent_k = raw_k.copy()

            for i in range(k_period + max(smooth_k, d_period) - 2, n):
                if ma_type == "sma":
                    start_idx = i - d_period + 1
                    end_idx = i
                    percent_d[i] = np.mean(percent_k[start_idx : end_idx + 1])

                elif ma_type == "ema":
                    alpha = 2 / (d_period + 1)
                    if i == k_period + max(smooth_k, d_period) - 2:
                        start_idx = i - d_period + 1
                        end_idx = i
                        percent_d[i] = np.mean(percent_k[start_idx : end_idx + 1])
                    else:
                        percent_d[i] = percent_k[i] * alpha + percent_d[i - 1] * (
                            1 - alpha
                        )

            for i in range(k_period + max(smooth_k, d_period * 2) - 3, n):
                start_idx = i - d_period + 1
                end_idx = i
                stoch_signal[i] = np.mean(percent_d[start_idx : end_idx + 1])

            percent_k_series = pd.Series(percent_k, index=close.index, name="Stoch_%K")
            percent_d_series = pd.Series(percent_d, index=close.index, name="Stoch_%D")
            raw_k_series = pd.Series(raw_k, index=close.index, name="Stoch_Raw_%K")
            signal_series = pd.Series(
                stoch_signal, index=close.index, name="Stoch_Signal"
            )

            return percent_k_series, percent_d_series, raw_k_series, signal_series

        pK, pD, raw_k, signal = calc_stoch_osc(df["high"], df["low"], df["close"])
        df["stoch_K"] = pK
        df["stoch_D"] = pD
        df["stoch_raw_K"] = raw_k
        df["stoch_signal"] = signal

        # Williams %R
        # df['Williams_%R'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])

        # CCI
        # df['CCI'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)

        # Ultimate Oscillator
        # df['Ultimate_Oscillator'] = ta.momentum.ultimate_oscillator(df['high'], df['low'], df['close'])

        # 3. Volatility
        df["BB_middle"] = df["close"].shift(1).rolling(window=20, min_periods=1).mean()
        bb_std = df["close"].shift(1).rolling(window=20, min_periods=1).std()
        df["BB_upper"] = df["BB_middle"] + (bb_std * 2)
        df["BB_lower"] = df["BB_middle"] - (bb_std * 2)
        df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_middle"]

        def calc_atr(high, low, close, window=14):
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.shift(1).rolling(window=window, min_periods=1).mean()
            return atr

        df["ATR"] = calc_atr(df["high"], df["low"], df["close"])

        # 4. Volume
        obv = (np.sign(df["close"].diff()) * df["volume"]).fillna(0)
        df["OBV"] = obv.shift(1).cumsum()
        # df['VWAP'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'])
        # df['CMF'] = ta.volume.chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'])
        # df['MFI'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'])

        # 5. Additional
        df["ROC"] = (
            (df["close"] - df["close"].shift(12)) / df["close"].shift(12)
        ) * 100
        df["Upper_Channel_20"] = (
            df["high"].shift(1).rolling(window=20, min_periods=1).max()
        )
        df["Lower_Channel_20"] = (
            df["low"].shift(1).rolling(window=20, min_periods=1).min()
        )

        df["ticker"] = ticker

        return df

    def get_indicators(
        self, ticker, timestamp=None, window=None
    ):  # (timestamp-window, timestamp]
        if ticker not in self.indicators_data:
            data = self.load_data(ticker)
            if data is not None and len(data) > 50:
                indicators_df = self.calculate_all_indicators(data, ticker)
                if indicators_df is not None:
                    self.indicators_data[ticker] = indicators_df
                else:
                    print(f"Не удалось рассчитать индикаторы для {ticker}")
                    return None
            else:
                print(f"Недостаточно данных для {ticker}")
                return None

        df = self.indicators_data[ticker]

        if timestamp is None:
            return df

        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)

        if timestamp not in df.index:
            available_dates = df.index[df.index <= timestamp]
            if len(available_dates) > 0:
                timestamp = available_dates[-1]
                print(
                    f"Указанный timestamp не найден. Используется ближайший доступный: {timestamp}"
                )
            else:
                print(f"Нет данных для тикера {ticker} до указанной даты {timestamp}")
                return None

        if window is not None:
            try:
                position = df.index.get_loc(timestamp)
                start_idx = max(0, position - window + 1)
                return df.iloc[start_idx : position + 1]
            except:
                print(f"Ошибка при получении окна данных для {ticker}")
                return df.loc[:timestamp].tail(window)
        else:
            return df.loc[[timestamp]]


if __name__ == "__main__":
    calc = TechnicalIndicators()
    calc.init_all_tickers()
    print(132)

    print(calc.get_indicators("SBER", "1674316000", 20))
