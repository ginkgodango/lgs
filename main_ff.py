import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

df_ff = pd.read_csv(
    'U:/CIO/#Data/input/ff/F-F_Research_Data_5_Factors_2x3.csv',
    skiprows=[0, 1, 2]
)

df_ff = df_ff.rename(columns={'Unnamed: 0': 'Date'})

for i in range(0, len(df_ff)):
    if df_ff['Date'][i] == ' Annual Factors: January-December ':
        split_index = i

df_ff_monthly = df_ff[:split_index].reset_index(drop=True)

df_ff_yearly = df_ff[split_index+2:].reset_index(drop=True)



#df_ff_monthly['Date'] = [str(df_ff_monthly['Date'][i]) for i in range(0, len(df_ff_monthly))]
#df_ff_monthly['Date'] = pd.to_datetime(df_ff_monthly['Date'], format='%Y%m', errors='coerce')


# df_ff_monthly = df_ff_monthly.set_index('Date')
#
# df_ff_yearly = df_ff_yearly.set_index('Date')

for column in df_ff_monthly.columns:
    df_ff_monthly[column] = pd.to_numeric(df_ff_monthly[column])

for column in df_ff_yearly.columns:
    df_ff_yearly[column] = pd.to_numeric(df_ff_yearly[column])

# df_ff_monthly.plot(figsize=(12.80, 7.2))
# plt.axhline(y=0, linestyle=':', linewidth=1, color='k', )

# df_ff_yearly.plot(figsize=(12.80, 7.2))
# plt.axhline(y=0, linestyle=':', linewidth=1, color='k', )

# i = 0
# for column in df_ff_yearly.columns:
#     fig = df_ff_yearly[[column]].plot()
#     fig.set_title('12 Month Return on ' + column + ' portfolio')
#     fig.set_ylabel(column + ' Return (%)')
#     plt.axhline(y=0, linestyle=':', linewidth=1, color='k', )
#     plt.tight_layout()
#     plt.savefig('U:/CIO/#Data/output/ff/charts/' + str(i) + '. 12 Month Return on ' + column + '.png')
#     i += 1


horizon_to_period_dict = {
        '1_': 1,
        '3_': 3,
        '12_': 12,
        '36_': 36,
        '60_': 60,
        '84_': 84
}

# Calculates the holding period returns and annualises for periods greater than 12 months.
for horizon, period in horizon_to_period_dict.items():

    for column in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']:

        column_name = horizon + column

        if period <= 12:
            df_ff_monthly[column_name] = (
                df_ff_monthly
                [column]
                .rolling(period)
                .apply(lambda r: np.prod(1+r)-1, raw=False)
                .reset_index(drop=False)[column]
            )

        elif period > 12:
            df_ff_monthly[column_name] = (
                df_ff_monthly
                [column]
                .rolling(period)
                .apply(lambda r: (np.prod(1+r)**(12/period))-1, raw=False)
                .reset_index(drop=False)[column]
            )

