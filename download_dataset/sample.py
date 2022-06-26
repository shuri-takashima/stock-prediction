import sys
import pandas as pd
from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError

code = "9432T"

my_share = share.Share(code)

save_path = str(code)
try:
    symbol_data= my_share.get_historical(
        period_type=share.PERIOD_TYPE_DAY,
        period=1, #期間
        frequency_type=share.FREQUENCY_TYPE_MINUTE,
        frequency=1, #足
    )
except YahooFinanceError as e:
    print(e.message)
    sys.exit(1)

df = pd.DataFrame(symbol_data)
df['datatime'] = pd.to_datetime(df.timestamp, unit="ms")
df = df.fillna(method="ffill")
df.to_csv('{}.csv'.format(save_path))