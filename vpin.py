##套件#######
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import chardet
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import statsmodels.api as sm
from numpy.linalg import lstsq
from dateutil.relativedelta import relativedelta
import mplfinance as mpf
import seaborn as sns
from scipy.stats import pearsonr
from scipy.stats import ttest_1samp
from arch import arch_model
import statsmodels.formula.api as smf
### import data ####
def read_csv_auto_encoding(path):
    with open(path, 'rb') as f:
        raw_data = f.read(20000)
    detect_result = chardet.detect(raw_data)
    encoding = detect_result['encoding']
    print(f"{path} 偵測到編碼：{encoding}")
    return pd.read_csv(path, encoding=encoding)

path_data1  = r"D:/台新/pin/FTQ5_intraday_data_818.csv"
path_data2  = r"D:/台新/pin/FTQ5_intraday_data_819.csv"
path_data3  = r"D:/台新/pin/FTQ5_intraday_data_820.csv"
path_data4  = r"D:/台新/pin/FTQ5_intraday_data_813.csv"
path_data5  = r"D:/台新/pin/FTQ5_intraday_data_815.csv"
path_data6  = r"D:/台新/pin/FTQ5_intraday_data_814.csv"
path_data7  = r"D:/台新/pin/FTQ5_intraday_data_813.csv"
path_data8  = r"D:/台新/pin/FTQ5_intraday_data_811.csv"
path_data9  = r"D:/台新/pin/FTQ5_intraday_data_808.csv"
path_data10 = r"D:/台新/pin/FTQ5_intraday_data_807.csv"
path_data11 = r"D:/台新/pin/FTQ5_intraday_data_805.csv"
path_data12 = r"D:/台新/pin/FTQ5_intraday_data_804.csv"
path_data13 = r"D:/台新/pin/FTQ5_intraday_data_801.csv"

data_818 = read_csv_auto_encoding(path_data1)
data_819 = read_csv_auto_encoding(path_data2)
data_820 = read_csv_auto_encoding(path_data3)
data_813 = read_csv_auto_encoding(path_data4)
data_815 = read_csv_auto_encoding(path_data5)
data_814 = read_csv_auto_encoding(path_data6)
data_813 = read_csv_auto_encoding(path_data7)
data_811 = read_csv_auto_encoding(path_data8)
data_808 = read_csv_auto_encoding(path_data9)
data_807 = read_csv_auto_encoding(path_data10)
data_805 = read_csv_auto_encoding(path_data11)
data_804 = read_csv_auto_encoding(path_data12)
data_801 = read_csv_auto_encoding(path_data13)

####處理基本資料######
#(1) 2025/08/18
data_818 = data_818[data_818['Dates'] != 'Dates'].copy()
data_818['Dates'] = pd.to_datetime(
    data_818['Dates'],
    format='%Y/%m/%d %p %I:%M:%S'
)
data_818['date'] = data_818['Dates'].dt.date
data_818['time'] = data_818['Dates'].dt.time
data_818 = data_818.drop(columns=['Dates'])

#(2) 2025/08/19
data_819 = data_819[data_819['Dates'] != 'Dates'].copy()
data_819['Dates'] = pd.to_datetime(
    data_819['Dates'],
    format='%Y/%m/%d %p %I:%M:%S'
)
data_819['date'] = data_819['Dates'].dt.date
data_819['time'] = data_819['Dates'].dt.time
data_819 = data_819.drop(columns=['Dates'])

#(3) 2025/08/20
data_820 = data_820[data_820['Dates'] != 'Dates'].copy()
data_820['Dates'] = pd.to_datetime(
    data_820['Dates'],
    format='%Y/%m/%d %p %I:%M:%S'
)
data_820['date'] = data_820['Dates'].dt.date
data_820['time'] = data_820['Dates'].dt.time
data_820 = data_820.drop(columns=['Dates'])

#(5) 2025/08/15
data_815 = data_815[data_815['Dates'] != 'Dates'].copy()
data_815['Dates'] = pd.to_datetime(
    data_815['Dates'],
    format='%Y/%m/%d %p %I:%M:%S'
)
data_815['date'] = data_815['Dates'].dt.date
data_815['time'] = data_815['Dates'].dt.time
data_815 = data_815.drop(columns=['Dates'])


#(6) 2025/08/14
data_814 = data_814[data_814['Dates'] != 'Dates'].copy()
data_814['Dates'] = pd.to_datetime(
    data_814['Dates'],
    format='%Y/%m/%d %p %I:%M:%S'
)
data_814['date'] = data_814['Dates'].dt.date
data_814['time'] = data_814['Dates'].dt.time
data_814 = data_814.drop(columns=['Dates'])

#(7) 2025/08/13
data_813 = data_813[data_813['Dates'] != 'Dates'].copy()
data_813['Dates'] = pd.to_datetime(
    data_813['Dates'],
    format='%Y/%m/%d %p %I:%M:%S'
)
data_813['date'] = data_813['Dates'].dt.date
data_813['time'] = data_813['Dates'].dt.time
data_813 = data_813.drop(columns=['Dates'])

#(9) 2025/08/08
data_808 = data_808[data_808['Dates'] != 'Dates'].copy()
data_808['Dates'] = pd.to_datetime(
    data_808['Dates'],
    format='%Y/%m/%d %p %I:%M:%S'
)
data_808['date'] = data_808['Dates'].dt.date
data_808['time'] = data_808['Dates'].dt.time
data_808 = data_808.drop(columns=['Dates'])

#(11) 2025/08/05 
data_805 = data_805[data_805['Dates'] != 'Dates'].copy()
data_805['Dates'] = pd.to_datetime(
    data_805['Dates'],
    format='%Y/%m/%d %p %I:%M:%S'
)   
data_805['date'] = data_805['Dates'].dt.date
data_805['time'] = data_805['Dates'].dt.time
data_805 = data_805.drop(columns=['Dates'])

#print(data_820,data_818,data_819.head())

combine_data = pd.concat([data_805, data_808, data_813, data_814, data_815, data_818, data_819, data_820], ignore_index=True)
combine_data = combine_data.sort_values(by=['date', 'time']).reset_index(drop=True) 

##標記trade、bid、ask##
df = combine_data.copy()
df = df.sort_values(['date','time']).reset_index(drop=True)
df['Type']  = df['Type'].str.upper().str.strip()
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df['Size']  = pd.to_numeric(df['Size'],  errors='coerce')

bid_types = ['BID']
ask_types = ['ASK']

# 建立報價
df['bid_prev'] = np.where(df['Type'].isin(bid_types), df['Price'], np.nan)
df['ask_prev'] = np.where(df['Type'].isin(ask_types), df['Price'], np.nan)
df[['bid_prev','ask_prev']] = df[['bid_prev','ask_prev']].ffill()

#標記買賣方驅動
trade = df[df['Type'] == 'TRADE'].copy()
trade['side'] = np.where(trade['Price'] >= trade['ask_prev'], 'BUY',
                 np.where(trade['Price'] <= trade['bid_prev'], 'SELL', 'UNKNOWN'))
trade = trade.drop(trade[trade['side'] == 'UNKNOWN'].index)
print(trade.head())

##計算每日總成交量及日平均成交量##
##計算每日總成交量##
trade_data = trade[trade['Type'] == 'TRADE'].copy()
trade_data['Size'] = pd.to_numeric(trade_data['Size'], errors='coerce').fillna(0) #轉換為num
daily_volume = trade_data.groupby('date', as_index=False)['Size'].sum().rename(columns={'Size':'total_volume'})

##計算每日平均成交量(有扣除開盤搓合的那筆)##
n = 50
daily_volume['avg_volume'] = daily_volume['total_volume'].mean()
bucket_capacity = daily_volume['avg_volume'] / n
#print("日平均成交量 (ADV):", daily_volume['avg_volume'])
#print("每桶容量:", bucket_capacity)
bucket_capacity = round(bucket_capacity)
bucket_capacity = bucket_capacity.iloc[0]   # 取第一個
trade['Size'] = pd.to_numeric(trade['Size'], errors='coerce').fillna(0)
trade['cum'] = trade.groupby('date')['Size'].cumsum()

def assign_bucket_with_time(group, cap):
    buckets = []
    done_ts = []
    cum = 0
    b = 1
    for size, ts in zip(group['Size'], group['dt']):
        cum += size
        if cum >= cap:
            # 桶完成：標記這一筆時間戳
            buckets.append(b)
            done_ts.append(ts)
            b += 1
            cum = cum - cap  # 保留超過的部分，重新累積
        else:
            # 桶還沒滿
            buckets.append(b)
            done_ts.append(pd.NaT)
    group['bucket'] = buckets
    group['bucket_done_at'] = done_ts
    return group

#加入時間戳#
trade = trade.sort_values(['date','time']).copy()
trade['dt'] = pd.to_datetime(trade['date'].astype(str) + ' ' + trade['time'].astype(str),
                             errors='coerce')

trade = trade.groupby('date', group_keys=False).apply(lambda g: assign_bucket_with_time(g, bucket_capacity))


print(trade[['date','time','Size','bucket','cum']].head(30))

##################計算vpin###################
trade['buy_vol']  = np.where(trade['side'] == 'BUY',  trade['Size'], 0.0)
trade['sell_vol'] = np.where(trade['side'] == 'SELL', trade['Size'], 0.0)

bucket_stats = (
    trade.groupby(['date','bucket'], as_index=False)
         .agg(
             buy_vol=('buy_vol','sum'),
             sell_vol=('sell_vol','sum'),
             bucket_done_at=('bucket_done_at','max')  # 加上完成時間戳
         )
)

bucket_stats['sell_roll'] = bucket_stats.groupby('date')['sell_vol'].transform(lambda s: s.rolling(window=n, min_periods=n).sum())
bucket_stats['buy_roll']  = bucket_stats.groupby('date')['buy_vol'].transform(lambda s: s.rolling(window=n, min_periods=n).sum())

#bucket_stats['OI_roll'] = (bucket_stats['sell_roll'] - bucket_stats['buy_roll']).abs()
bucket_stats['OI_roll'] = (bucket_stats['sell_roll'] - bucket_stats['buy_roll'])
bucket_stats['VPIN_alt'] = bucket_stats['OI_roll'] / (n * bucket_capacity)

bucket_stats.to_excel('D:/台新/pin/vpin_result.xlsx', index=False)

###畫累積分布####
# 篩出 8/13 的 VPIN 值
vpin_813 = bucket_stats.loc[bucket_stats['date'] == pd.to_datetime("2025-08-13").date(), 'VPIN_alt'].dropna()

# 排序後計算累積比例
values = np.sort(vpin_813)
cdf = np.arange(1, len(values)+1) / len(values)

# 畫圖
plt.figure(figsize=(8,5))
plt.plot(values, cdf, marker='.', linestyle='-')
plt.xlabel("VPIN")
plt.ylabel("CDF")
plt.title("CDF of VPIN (2025-08-13)")
plt.grid(True)
plt.show()

####觀察是否有相關###
# 先過濾 8/13 的資料
focus = bucket_stats[bucket_stats['VPIN_alt'].notna()].copy()
focus['target_time'] = focus['bucket_done_at'] + pd.Timedelta(minutes=5)
focus = focus[focus['target_time'].notna()].copy()
print(focus.head())
results = []
trade_all = trade.copy()

for _, row in focus.iterrows():
    done_time = row['bucket_done_at']
    target_time = row['target_time']

    # 找「桶完成時」的成交價
    price_done = trade_all.loc[trade_all['dt'] <= done_time, 'Price'].iloc[-1]

    # 找「target_time 附近」最近的一筆成交價
    nearest = trade_all.loc[trade_all['dt'] >= target_time].iloc[0]
    price_target = nearest['Price']

    # 算報酬率
    ret = price_target / price_done - 1

    results.append({
        "bucket": row['bucket'],
        "bucket_done_at": done_time,
        "VPIN_alt": row['VPIN_alt'],
        "target_time": target_time,
        "price_done": price_done,
        "price_target": price_target,
        "return": ret
    })

returns_df = pd.DataFrame(results)
#print(returns_df)

X = returns_df['VPIN_alt']
y = returns_df['return']

X = sm.add_constant(X) 
model = sm.OLS(y, X).fit()

print(model.summary()) 

##把abs.刪掉的vpin與報酬呈現負相關，若賣方驅動越大，賣壓出來，報酬會下降，完全合邏輯笑死，但不顯著^^
##用正常的vpin(絕對值)與一分鐘後的報酬成正相關，合邏輯，畢竟資訊不對稱出來，會要求較高的報酬率，但不顯著。
##但我想要知道這個vpin出來後，是否是

