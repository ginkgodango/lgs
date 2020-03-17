import os
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import statsmodels.api as sm

# START USER INPUT DATA
jpm_main_filepath = 'U:/CIO/#Data/input/jpm/performance/2020/02/Historical Time Series - Monthly - Main Returns and Benchmarks.xlsx'
jpm_alts_filepath = 'U:/CIO/#Data/input/jpm/performance/2020/02/Historical Time Series - Monthly - Alternatives Returns and Benchmarks.xlsx'
jpm_mv_filepath = 'U:/CIO/#Data/input/jpm/performance/2020/02/Historical Time Series - Monthly - Main Market Values.xlsx'
jpm_mv_alts_filepath = 'U:/CIO/#Data/input/jpm/performance/2020/02/Historical Time Series - Monthly - Alternatives Market Values.xlsx'
lgs_dictionary_filepath = 'U:/CIO/#Data/input/lgs/dictionary/2020/02/New Dictionary_v6.xlsx'
FYTD = 8
report_date = dt.datetime(2020, 2, 29)
# END USER INPUT DATA

# Imports the JPM time-series.
use_managerid = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 14, 15]
use_accountid = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15]
footnote_rows = 28

df_jpm_main = pd.read_excel(
        pd.ExcelFile(jpm_main_filepath),
        sheet_name='Sheet1',
        skiprows=use_accountid,
        skipfooter=footnote_rows,
        header=1
        )

# Reshapes the time-series into a panel.
df_jpm_main = df_jpm_main.rename(columns={'Unnamed: 0': 'Date'})
df_jpm_main = df_jpm_main.set_index('Date')
df_jpm_main = df_jpm_main.transpose()
df_jpm_main = df_jpm_main.reset_index(drop=False)
df_jpm_main = df_jpm_main.rename(columns={'index': 'Manager'})
df_jpm_main = pd.melt(df_jpm_main, id_vars=['Manager'], value_name='JPM_Return')
df_jpm_main = df_jpm_main.sort_values(['Manager', 'Date'])
df_jpm_main = df_jpm_main.reset_index(drop=True)
df_jpm_main = df_jpm_main.replace('-', np.NaN)

df_jpm_alts = pd.read_excel(
        pd.ExcelFile(jpm_alts_filepath),
        sheet_name='Sheet1',
        skiprows=use_accountid,
        skipfooter=footnote_rows,
        header=1
        )

df_jpm_alts = df_jpm_alts.rename(columns={'Unnamed: 0': 'Date'})
df_jpm_alts = df_jpm_alts.set_index('Date')
df_jpm_alts = df_jpm_alts.transpose()
df_jpm_alts = df_jpm_alts.reset_index(drop=False)
df_jpm_alts = df_jpm_alts.rename(columns={'index': 'Manager'})
df_jpm_alts = pd.melt(df_jpm_alts, id_vars=['Manager'], value_name='JPM_Return')
df_jpm_alts = df_jpm_alts.sort_values(['Manager', 'Date'])
df_jpm_alts = df_jpm_alts.reset_index(drop=True)
df_jpm_alts = df_jpm_alts.replace('-', np.NaN)

df_jpm = pd.concat([df_jpm_alts, df_jpm_main], axis=0).reset_index(drop=True)


# Imports the JPM time-series of market values.
df_jpm_mv = pd.read_excel(
        pd.ExcelFile(jpm_mv_filepath),
        sheet_name='Sheet1',
        skiprows=use_accountid,
        skipfooter=footnote_rows,
        header=1
        )

# Reshapes the time-series into a panel.
df_jpm_mv = df_jpm_mv.rename(columns={'Unnamed: 0': 'Date'})
df_jpm_mv = df_jpm_mv.set_index('Date')
df_jpm_mv = df_jpm_mv.transpose()
df_jpm_mv = df_jpm_mv.reset_index(drop=False)
df_jpm_mv = df_jpm_mv.rename(columns={'index': 'Manager'})
df_jpm_mv = pd.melt(df_jpm_mv, id_vars=['Manager'], value_name='Market Value')
df_jpm_mv = df_jpm_mv.sort_values(['Manager', 'Date'])
df_jpm_mv = df_jpm_mv.reset_index(drop=True)
df_jpm_mv = df_jpm_mv.replace('-', np.NaN)

# Imports the JPM time-series of market values.
df_jpm_mv_alts = pd.read_excel(
        pd.ExcelFile(jpm_mv_alts_filepath),
        sheet_name='Sheet1',
        skiprows=use_accountid,
        skipfooter=footnote_rows,
        header=1
        )

# Reshapes the time-series into a panel.
df_jpm_mv_alts = df_jpm_mv_alts.rename(columns={'Unnamed: 0': 'Date'})
df_jpm_mv_alts = df_jpm_mv_alts.set_index('Date')
df_jpm_mv_alts = df_jpm_mv_alts.transpose()
df_jpm_mv_alts = df_jpm_mv_alts.reset_index(drop=False)
df_jpm_mv_alts = df_jpm_mv_alts.rename(columns={'index': 'Manager'})
df_jpm_mv_alts = pd.melt(df_jpm_mv_alts, id_vars=['Manager'], value_name='Market Value')
df_jpm_mv_alts = df_jpm_mv_alts.sort_values(['Manager', 'Date'])
df_jpm_mv_alts = df_jpm_mv_alts.reset_index(drop=True)
df_jpm_mv_alts = df_jpm_mv_alts.replace('-', np.NaN)

df_jpm_mv = pd.concat([df_jpm_mv, df_jpm_mv_alts]).reset_index(drop=True)

df_jpm_main = pd\
    .merge(
        left=df_jpm_mv,
        right=df_jpm,
        left_on=['Manager', 'Date'],
        right_on=['Manager', 'Date'],
        how='right'
    )\
    .sort_values(['Manager', 'Date'])\
    .reset_index(drop=True)

# Sort values VERY IMPORTANT for groupby merging
df_jpm_main = df_jpm_main.sort_values(['Manager', 'Date'], ascending=[True, True]).reset_index(drop=True)

df_lgs = pd.read_excel(
    pd.ExcelFile(lgs_dictionary_filepath),
    sheet_name='Sheet1',
    header=0
)
df_lgs = df_lgs.rename(columns={'JPM Account Id': 'Manager'})

df_jpm_main = pd.merge(
        left=df_jpm_main,
        right=df_lgs,
        left_on=['Manager'],
        right_on=['Manager'],
        how='inner'
)

# Keep only open accounts
df_jpm_main = df_jpm_main[df_jpm_main['LGS Open'] == 1].reset_index(drop=True)
df_jpm_main = df_jpm_main.drop(columns=['LGS Open'], axis=1)

# Keep only reported items
df_jpm_not_reported = df_jpm_main[df_jpm_main['JPM ReportName'].isin([np.nan])]
df_jpm_main = df_jpm_main[~df_jpm_main['JPM ReportName'].isin([np.nan])].reset_index(drop=True)

# Sort values VERY IMPORTANT for groupby merging
df_jpm_main = df_jpm_main.sort_values(['Manager', 'Date'], ascending=[True, True]).reset_index(drop=True)

