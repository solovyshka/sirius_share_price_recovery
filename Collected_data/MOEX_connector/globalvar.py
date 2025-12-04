import time
from db_creator import cldbcandle

#== БЛОК ГЛОБАЛЬНЫХ ПЕРЕМЕННЫХ ===
#== списки тикеров ===
l_tickers               = []            # основной список
#l_tickers_report        = []            # список тикеров, у которых сегодня отчеты
#l_tickers_cancel        = []            # список тикеров, у которых отчеты на текущей неделе, но после текущего дня
l_rsimax                = []            # список тикеров, у которых RSI выше границы
l_rsimin                = []            # список тикеров, у которых RSI ниже границы
#== словари объектов с данными по акции ===
dict_obj_stocks         = {}            # словарь ВСЕХ  объектов
#dict_obj_stocks_rep     = {}            # словарь тех, у кого сегодня отчеты (в постмаркете вчера и премаркете сегодня)
#dict_obj_stocks_panel   = {}            # словарь индексов, ETF, акций панели приборов
#== флаги разные ====
fl_ispause              = True
fl_start                = True          # true при запуске программы для обхода первого таймера
#== разные переменные ===
timer1d                 = None           # суточный таймер
favorite                = None           # объект избранноеDa
#exchange                = None          # если 'MOEX' то работаем с МОЕХ
#*********************************
dbcandle                = cldbcandle()   # база sql со свечами
excelfile               = None           # файл excel с результатами
datefrom                = int(time.mktime(time.strptime('2018-01-01', '%Y-%m-%d'))) 
#*********************************
queue_msg               = []            # очередь сообщений программы
new_queue_count         = 0             # количество новых сообщений
#*********************************

import datetime
def getmoment():
    moment = datetime.datetime.now()
    return f'{moment:%d-%m-%Y %H:%M:%S:%f}'
