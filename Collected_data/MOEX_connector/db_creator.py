#import sqlite3


#def create_db(**data):
#    with sqlite3.connect('tcrdata.sqlite') as db:
#        cursor = db.cursor()
#        a = ['Date', 'High', 'Low', 'Open', 'Close', 'Volume']
#        with open('listtickers.tcr') as file:
#            tcrs = [i.replace('\n', '') for i in file.readlines()]
#
#        for i in range(len(tcrs)):
#            for j in a:
#                query = """CREATE TABLE IF NOT EXISTS "Tickers"(Ticker TEXT, Date integer[], High integer[], 
#                Low integer[], Open integer[], Close integer[], Volume integer[])"""
#
#                query2 = f""" INSERT INTO "Tickers" VALUES
#                 ("{tcrs[i]}", { {*data['Date']} }, { {*data['High']} }, { {*data['Low']} }, { {*data['Open']} }, { {*data['Close']} }, { {*data['Volume']} })"""
#                cursor.execute(query)
#                cursor.execute(query2)

"""
модуль работы с базой данных SQLite:

"""
import globalvar as glv
import pandas as pd
import os
from cmath import nan
import sqlite3
from datetime import datetime
import time


# ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
# функции для сохранения в базе свечей
# ===== (база данных) ======
class cldbcandle(object):
    connect     = None
    cursordb 	= None
    records     = None
    #--------------------------
    def __init__(self):
        self.connect = sqlite3.connect('allcandles.db')			    # устанавливаем соединение с БД
        self.cursordb = self.connect.cursor()                       # получаем курсор
    #-------------------------
    def adddatatable(self, name, df):
        self.createtable(name)
        try:
            df.to_sql(name, con=self.connect, if_exists="replace", index=False)
        except sqlite3.Error as error:
            print("ERROR ", error)
    #-------------------------
    def createtable(self, ticker):
        # создаем таблицу с именем ticker (если её ещё нет) 
        sqlstr = f'CREATE TABLE IF NOT EXISTS {ticker} ("Date" INTEGER, "High" TEXT, "Low" TEXT, "Open"  TEXT, "Close" TEXT, "Volume" TEXT)'
        try:
            self.cursordb.execute(sqlstr)
        except Exception as e:
            print(sqlstr)
        self.connect.commit()
    #---------------------------
    def getlisttable(self):
        self.cursordb.execute('SELECT name from sqlite_master where type= "table"')      
        result = self.cursordb.fetchall()
        return list(result)
    #---------------------------
    def loadcandles(self, name):
        sqlstr = f'SELECT * FROM "{name}"'
        try:
            df = pd.read_sql(sqlstr, self.connect)
        except sqlite3.Error as error:
            print("ERROR ", error)
            return None
        df = df.set_index('Date')
        try:
            df = df.astype(float)
            return df
        except Exception as e:
            return None
    #---------------------------
    def getlastdate(self, name):
        sqlstr = f'SELECT MAX(Date) FROM "{name}"'
        self.cursordb.execute(sqlstr)
        result = self.cursordb.fetchone()
        if result[0] != '' and result[0] != None:
            dt = time.strftime("%d-%m-%Y %H:%M", time.localtime(int(result[0])))
            return int(result[0]), dt
        else:
            return 0, ''
    #---------------------------
    def updatetable(self, name, df):
        try:
            df.to_sql(name, con=self.connect, if_exists="append", index=False)
        except sqlite3.Error as error:
            print("ERROR ", error)
        return
    #---------------------------
    def checkexisttable(self,name):
        self.cursordb.execute(''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name=? ''', (name,))
        if self.cursordb.fetchone()[0]==1:
            return True
        else:
            return False
    #---------------------------
