import pandas as pd

df = pd.read_csv(f'Data/SH1A0001/SH1A0001_1m.csv')
# df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df.drop(['Time'], inplace=True, axis=1)
df.set_index('Date', inplace=True)
df.to_csv(f'Data/SH1A0001/SH1A0001_1m.csv')

# Tổng hợp dữ liệu theo phút
df_5m = df.resample('5min').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
})

df_5m.dropna(inplace=True)
df_5m.to_csv(f'Data/SH1A0001/SH1A0001.csv')

# Tổng hợp dữ liệu theo phút
df_30m = df.resample('30min').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
})

df_30m.dropna(inplace=True)
df_30m.to_csv(f'Data/SH1A0001/SH1A0001_30m.csv')

df_2h = df.resample('2h').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
})

df_2h.dropna(inplace=True)
df_2h.to_csv(f'Data/SH1A0001/SH1A0001_2h.csv')

# df_1d = df.resample('1D').agg({
#     'open': 'first',
#     'high': 'max',
#     'low': 'min',
#     'close': 'last',
#     'volume': 'sum'
# })
#
# df_1d.dropna(inplace=True)
# df_1d.to_csv(f'Data/SH1A0001/SH1A0001.csv')

