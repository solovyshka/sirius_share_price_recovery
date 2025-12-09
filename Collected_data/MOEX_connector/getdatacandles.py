import pandas as pd
import time
import requests
import globalvar as glv
from tqdm import tqdm


class all_data_stock(object):
    # --------------------------
    def __init__(self, ticker):
        self.s_ticker = ticker  # тикер
        self.df_candles = pd.DataFrame()  # датафрейм данных


# ------------------------------------------
# инициализация (создание) списка объектов класса "данные по акции"
# возвращает список объектов класса "данные по акции"
def init(list_tickers):
    dict_obj = {}
    for tcr in list_tickers:
        tmp_obj = all_data_stock(tcr[0])  # создание объекта по тикеру
        dict_obj[tcr[0]] = tmp_obj  # добавление в словарь
    return dict_obj


# ------------------------------------------
# получение данных с биржи MOEX. на входе  тикер, тип (shares или futures)
# на выходе DataFrame:'Date' 'High' 'Low' 'Open' 'Close' 'Volume'


def recieve_day_candle_MOEX(ticker, type="shares"):
    read_ok = True
    op = []
    cl = []
    hi = []
    lo = []
    vo = []
    tm = []
    yesterday = time.strftime("%Y-%m-%d", time.localtime(time.time() - 86400))
    # yesterday = '2024-05-07'
    dateto = int(time.mktime(time.strptime(yesterday, "%Y-%m-%d")))
    # table_exist = glv.dbcandle.checkexisttable(ticker)
    table_exist = False
    ## TODO
    if table_exist:  # в базе уже есть таблица с данными по этой акции
        # print(ticker)
        dtunix, dtstr = glv.dbcandle.getlastdate(
            ticker
        )  # получим конечную дату имеющихся в базе свечей
        if dtunix is None:
            datefrom = "2019-01-01"
        else:
            dtunix = dtunix + 86400  # это следующий день
            if (
                dtunix >= dateto
            ):  # если следующий день это завтра, значит свечи по сегодня включительно уже есть
                return
            datefrom = time.strftime("%Y-%m-%d", time.localtime(dtunix))
    else:
        datefrom = "2019-01-01"
    header = (
        "Mozilla/5.0 "
        "(Windows NT 10.0; WOW64) "
        "AppleWebKit/537.36"
        " (KHTML, like Gecko) "
        "Chrome/91.0.4472.135 "
        "Safari/537.36"
    )
    session = requests.Session()
    start = 0
    flagend = True

    while flagend:
        # print(start)
        if type == "shares":
            url = f"https://iss.moex.com/iss/engines/stock/markets/shares/securities/{ticker}/candles.json?from={datefrom}&till={yesterday}&interval=60&start={start}"
            # ДЛЯ ФЬЮЧЕРСА
        else:
            url = f"https://iss.moex.com/iss/engines/futures/markets/forts/securities/{ticker}/candles.json?from={datefrom}&till={yesterday}&interval=60&start={start}"
        try:
            response = session.get(url, headers={"User-Agent": header}, timeout=300)
        except Exception as e:
            print(f"MOEX response error: {ticker}")
            print(e)
            read_ok = False
            break
        if response.status_code == 200:
            try:
                count = 0
                array = response.json()  # array - это объект json???
                # print(url)
                for item in array["candles"]["data"]:
                    op.append(item[0])
                    cl.append(item[1])
                    hi.append(item[2])
                    lo.append(item[3])
                    vo.append(item[5])
                    tm.append(item[6])
                    count += 1
                msg = f"MOEX: ok read {ticker} {start} - {start + count}"
                print(msg)
                glv.queue_msg.append(msg)
                if count < 500:
                    flagend = False
                    read_ok = True
                else:
                    start += 500
                    continue
            except Exception as e:
                msg = f"MOEX error data in array: {ticker}"
                print(msg)
                glv.queue_msg.append(msg)
                print(e)
                read_ok = False
                break
        else:
            msg = f"MOEX error response ticker: {ticker}"
            print(msg)
            glv.queue_msg.append(msg)
            read_ok = False

    if not read_ok:
        return
    else:
        df = pd.DataFrame(
            {
                "Date": tm,
                "High": hi,
                "Low": lo,
                "Open": op,
                "Close": cl,
                "Volume": vo,
            }
        )
        df["Date"] = df["Date"].apply(
            lambda x: int(time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))
        )
        df["High"] = df["High"].apply(lambda x: f"{x}")
        df["Low"] = df["Low"].apply(lambda x: f"{x}")
        df["Open"] = df["Open"].apply(lambda x: f"{x}")
        df["Close"] = df["Close"].apply(lambda x: f"{x}")
        df["Volume"] = df["Volume"].apply(lambda x: f"{x}")

        df.to_csv(f"../../DataBase/{ticker}.csv")
    # print(df)
    return


# ------------------------------------------


# =====================
# =====================
if __name__ == "__main__":
    print("======= :-) =========")
    print("Это модуль getdatacandles")
    print("=====================")

    recieve_day_candle_MOEX("MOEX")
# =====================
# =====================
