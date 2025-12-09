import pandas as pd
import os
from tqdm import tqdm

path = "../../DataBase/"
print()

close_df = pd.DataFrame()
open_df = pd.DataFrame()
high_df = pd.DataFrame()
low_df = pd.DataFrame()
volume_df = pd.DataFrame()

base_ticker = "SBER.csv"
df_download = pd.read_csv(path + base_ticker)
close_df["timestamp"] = df_download["Date"]
open_df["timestamp"] = df_download["Date"]
high_df["timestamp"] = df_download["Date"]
low_df["timestamp"] = df_download["Date"]
volume_df["timestamp"] = df_download["Date"]

for file in tqdm(os.listdir(path)):
    df_step = pd.read_csv(path + file)
    name = file.split(".")[0]
    close_df[name] = df_step["Close"]
    open_df[name] = df_step["Open"]
    low_df[name] = df_step["Low"]
    high_df[name] = df_step["High"]
    volume_df[name] = df_step["Volume"]

close_df.to_csv(path + "CLOSE.csv")
open_df.to_csv(path + "OPEN.csv")
low_df.to_csv(path + "LOW.csv")
high_df.to_csv(path + "HIGH.csv")
volume_df.to_csv(path + "VOLUME.csv")
