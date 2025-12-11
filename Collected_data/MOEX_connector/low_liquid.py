import pandas as pd


def make_yearly_mean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = pd.to_datetime(df["timestamp"], unit="s").dt.year
    tickers = list(df.columns.difference(["timestamp", "year"]))

    # Группировка по году и среднее по каждому тикеру
    # На выходе: index = год, columns = тикеры
    yearly = df.groupby("year")[tickers].mean()
    res = yearly.T
    res.columns.name = None
    return res


if __name__ == "__main__":
    df = pd.read_csv("DataBase/VOLUME.csv", index_col=0)
    pd.set_option("display.float_format", "{:.4f}".format)
    low_liquid_df = make_yearly_mean_df(df).sort_values(by=2020)
    print(low_liquid_df.head(10))
    # low_liquid_df.to_csv("DataBase/LOW_LIQUID.csv")
