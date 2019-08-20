import pandas as pd
import numpy as np

jpm_filepath = 'U:/CIO/#Investment_Report/Data/input/returns/20190731 Historical Time Series.xlsx'
FYTD = 1

jpm_xlsx = pd.ExcelFile(jpm_filepath)
use_managerid = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11]
use_accountid = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14]
footnote_rows = 28

df_jpm = pd.read_excel(
        jpm_xlsx,
        sheet_name='Sheet1',
        skiprows=use_accountid,
        skipfooter=footnote_rows,
        header=1
        )

df_jpm = df_jpm.rename(columns={'Unnamed: 0': 'Date'})
df_jpm = df_jpm.set_index('Date')
df_jpm = df_jpm.transpose()
df_jpm = df_jpm.reset_index(drop=False)
df_jpm = df_jpm.rename(columns={'index': 'Manager'})
df_jpm = pd.melt(df_jpm, id_vars=['Manager'], value_name='Return_JPM')
df_jpm = df_jpm.sort_values(['Manager', 'Date'])
df_jpm = df_jpm.reset_index(drop=True)

df_jpm = df_jpm.replace('-', np.NaN)
df_jpm['Return_JPM'] = df_jpm['Return_JPM']/100

column_to_month_dict = {
    '1_re': 1,
    '3_re': 3,
    'FYTD_re': FYTD,
    '12_re': 12,
    '36_re': 36,
    '60_re': 60,
    '84_re': 84
}

for column, month in column_to_month_dict.items():

    if month <= 12:
        df_jpm[column] = (
            df_jpm
            .groupby(['Manager'])['Return_JPM']
            .rolling(month)
            .apply(lambda r: np.prod(1+r)-1, raw=False)
            .reset_index(drop=False)['Return_JPM']
        )

    elif month > 12:
        df_jpm[column] = (
            df_jpm
            .groupby(['Manager'])['Return_JPM']
            .rolling(month)
            .apply(lambda r: (np.prod(1+r)**(12/month))-1, raw=False)
            .reset_index(drop=False)['Return_JPM']
        )

