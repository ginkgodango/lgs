import pandas as pd
import numpy as np
import datetime as dt

relative_file_path = 'C:/Users/Mnguyen/LGSS/Investments Team - SandPits - SandPits/documents/lgs/reports/sri/'
rba_cpi_file_path = relative_file_path + 'g1-data_20210228.csv'
FYTD = 8

df_cpi = pd.read_csv(
    rba_cpi_file_path,
    header=10,
    usecols=['Series ID', 'GCPIAGQP'],
    parse_dates=['Series ID'],
    index_col=['Series ID']
)

# Converts quarterly inflation into monthly
df_cpi = df_cpi.rename(columns={'GCPIAGQP': 'Inflation'})
df_cpi = df_cpi[df_cpi.index >= dt.datetime(1999, 1, 1)]
df_cpi['Inflation'] = df_cpi['Inflation']/100
df_cpi['Inflation'] = ((1 + df_cpi['Inflation']) ** (1/3)) - 1
df_cpi = df_cpi.resample('M').pad()
df_cpi['Inflation'] = df_cpi['Inflation'].shift(-2).ffill()
df_cpi = df_cpi.reset_index(drop=False)
df_cpi = df_cpi.rename(columns={'Series ID': 'Date'})

df_cpi.to_csv(relative_file_path + 'inflation_output.csv', index=False)

horizon_to_period_dict = {
        '1_': 1,
        '3_': 3,
        str(FYTD) + '_': FYTD,
        '12_': 12,
        '24_': 24,
        '36_': 36,
        '60_': 60,
        '84_': 84,
        '120_': 120,
}

for horizon, period in horizon_to_period_dict.items():

    for return_type in ['Inflation']:

        column_name = str(period) + '_' + return_type

        if period <= 12:
            df_cpi[column_name] = (
                df_cpi
                [return_type]
                .rolling(period)
                .apply(lambda r: np.prod(1+r)-1, raw=False)
                .reset_index(drop=False)[return_type]
            )

        elif period > 12:
            df_cpi[column_name] = (
                df_cpi
                [return_type]
                .rolling(period)
                .apply(lambda r: (np.prod(1+r)**(12/period))-1, raw=False)
                .reset_index(drop=False)[return_type]
            )

df_cpi.to_csv(relative_file_path + 'inflation_output.csv', index=False)
