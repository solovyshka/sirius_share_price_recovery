import pandas as pd
from getdatacandles import recieve_day_candle_MOEX
from tqdm import tqdm

df = pd.read_csv('list_of_tickers.csv')
for tick in tqdm(df['Tickers'].values):
    recieve_day_candle_MOEX(tick)