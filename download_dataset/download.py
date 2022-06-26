import os
import sys
import pandas as pd
import datetime as datetime
from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError


code_list=[]
stock_list = pd.read_excel('./data_j.xls')
for index, data in stock_list.iterrows():
    code_list.append(data['コード'])


def finance_info_csv(code, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    code_T = str(code)
    my_share = share.Share(code_T)

    save_path = save_dir+"/"+str(code)+".csv"
    try:
        symbol_data= my_share.get_historical(
            period_type=share.PERIOD_TYPE_YEAR,
            period=10, #期間
            frequency_type=share.FREQUENCY_TYPE_DAY,
            frequency=1, #足
        )
        df = pd.DataFrame(symbol_data)
        df['datetime'] = pd.to_datetime(df.timestamp, unit="ms")
        df["datetime_jst"] = df["datetime"] + datetime.timedelta(hours=9)

        # df = df.drop(labels="datetime", axis=1)
        # df = df.drop(labels="timestamp", axis=1)

        df = df.set_index("datetime_jst")
        df = df.asfreq(freq="1min")
        df = df.fillna(method="ffill")
        df = df.drop_duplicates()

        # if len(df) == 30:
        #     df.to_csv(save_path)
        #     print(code_T, 'save')
        # else:
        #     print(code_T,"loss the stock")

        df.to_csv(save_path)
        print(code_T, 'save')


    except YahooFinanceError as e:
        print(e.message)

    except AttributeError as e:
        print(e)


# for code in code_list:
finance_info_csv(code="^N225", save_dir="../dataset/NIKKEI/10years_1day")



