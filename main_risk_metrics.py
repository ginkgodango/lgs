import os
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import statsmodels.api as sm

# START USER INPUT DATA
jpm_main_returns_filepath = 'C:/Users/mnguyen/LGSS/Investments Team - SandPits - SandPits/data/input/vendors/jpm/markets/performance/2021/01/Historical Time Series - Monthly - Main Returns.xlsx'
jpm_alts_returns_filepath = 'C:/Users/mnguyen/LGSS/Investments Team - SandPits - SandPits/data/input/vendors/jpm/markets/performance/2021/01/Historical Time Series - Monthly - Alts Returns.xlsx'
jpm_main_benchmarks_filepath = 'C:/Users/mnguyen/LGSS/Investments Team - SandPits - SandPits/data/input/vendors/jpm/markets/performance/2021/01/Historical Time Series - Monthly - Main Benchmarks.xlsx'
jpm_alts_benchmarks_filepath = 'C:/Users/mnguyen/LGSS/Investments Team - SandPits - SandPits/data/input/vendors/jpm/markets/performance/2021/01/Historical Time Series - Monthly - Alts Benchmarks.xlsx'
jpm_main_market_values_filepath = 'C:/Users/mnguyen/LGSS/Investments Team - SandPits - SandPits/data/input/vendors/jpm/markets/performance/2021/01/Historical Time Series - Monthly - Main Market Values.xlsx'
jpm_alts_market_values_filepath = 'C:/Users/mnguyen/LGSS/Investments Team - SandPits - SandPits/data/input/vendors/jpm/markets/performance/2021/01/Historical Time Series - Monthly - Alts Market Values.xlsx'
lgs_dictionary_filepath = 'C:/Users/mnguyen/LGSS/Investments Team - SandPits - SandPits/data/input/lgs/dictionary/2021/01/New Dictionary_v17.xlsx'
FYTD = 7
report_date = dt.datetime(2021, 1, 31)
# END USER INPUT DATA

use_managerid = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12]
use_accountid = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
footnote_rows = 28

# # Sets rows to ignore when importing the JPM time-series. Before March 2020
# use_managerid = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 14, 15]
# use_accountid = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15]
# footnote_rows = 28


def ts_to_panel(df_ts, date, set_index_column_name, set_value_column_name):
    """
    Converts JPM time series from time-series shape to panel shape.
    :param df_ts: JPM time series
    :param date: Name of date column in JPM time series
    :param set_index_column_name: Name you give the index column
    :param set_value_column_name: Name you give the values column
    :return: dataframe in panel form of JPM time series
    """
    df_panel = df_ts.rename(columns={date: 'Date'})
    df_panel = df_panel.set_index('Date')
    df_panel = df_panel.transpose()
    df_panel = df_panel.reset_index(drop=False)
    df_panel = df_panel.rename(columns={'index': set_index_column_name})
    df_panel = pd.melt(df_panel, id_vars=[set_index_column_name], value_name=set_value_column_name)
    df_panel = df_panel.sort_values([set_index_column_name, 'Date'])
    df_panel = df_panel.reset_index(drop=True)
    df_panel = df_panel.replace('-', np.NaN)

    return df_panel


# Reads in the jpm main returns historical time series
df_jpm_main_returns = pd.read_excel(
    pd.ExcelFile(jpm_main_returns_filepath),
    sheet_name='Sheet1',
    skiprows=use_accountid,
    skipfooter=footnote_rows,
    header=1
)

# Reshapes the time-series into a panel.
df_jpm_main_returns = ts_to_panel(
    df_ts=df_jpm_main_returns,
    date='Unnamed: 0',
    set_index_column_name='Manager',
    set_value_column_name='JPM_Return'
)

# Reads in the jpm alts returns historical time series
df_jpm_alts_returns = pd.read_excel(
    pd.ExcelFile(jpm_alts_returns_filepath),
    sheet_name='Sheet1',
    skiprows=use_accountid,
    skipfooter=footnote_rows,
    header=1
)

# Reshapes the time-series into a panel.
df_jpm_alts_returns = ts_to_panel(
    df_ts=df_jpm_alts_returns,
    date='Unnamed: 0',
    set_index_column_name='Manager',
    set_value_column_name='JPM_Return'
)

# Joins the jpm main returns and jpm alts returns together
df_jpm_returns = pd.concat([df_jpm_alts_returns, df_jpm_main_returns], axis=0).reset_index(drop=True)

# Reads in the jpm main benchmarks historical time series
df_jpm_main_benchmarks = pd.read_excel(
    pd.ExcelFile(jpm_main_benchmarks_filepath),
    sheet_name='Sheet1',
    skiprows=use_accountid,
    skipfooter=footnote_rows,
    header=1
)

# Reshapes the time-series into a panel.
df_jpm_main_benchmarks = ts_to_panel(
    df_ts=df_jpm_main_benchmarks,
    date='Unnamed: 0',
    set_index_column_name='Manager',
    set_value_column_name='JPM_Benchmark'
)

# Reads in the jpm alts benchmarks historical time series
df_jpm_alts_benchmarks = pd.read_excel(
    pd.ExcelFile(jpm_alts_benchmarks_filepath),
    sheet_name='Sheet1',
    skiprows=use_accountid,
    skipfooter=footnote_rows,
    header=1
)

# Reshapes the time-series into a panel.
df_jpm_alts_benchmarks = ts_to_panel(
    df_ts=df_jpm_alts_benchmarks,
    date='Unnamed: 0',
    set_index_column_name='Manager',
    set_value_column_name='JPM_Benchmark'
)

# Joins the jpm main benchmarks and jpm alts benchmarks together
df_jpm_benchmarks = pd.concat([df_jpm_alts_benchmarks, df_jpm_main_benchmarks], axis=0).reset_index(drop=True)

# Imports the JPM main market values time-series.
df_jpm_main_mv = pd.read_excel(
    pd.ExcelFile(jpm_main_market_values_filepath),
    sheet_name='Sheet1',
    skiprows=use_accountid,
    skipfooter=footnote_rows,
    header=1
)

# Reshapes the time-series into a panel.
df_jpm_main_mv = ts_to_panel(
    df_ts=df_jpm_main_mv,
    date='Unnamed: 0',
    set_index_column_name='Manager',
    set_value_column_name='Market Value'
)

# Imports the JPM alternatives market values time-series.
df_jpm_alts_mv = pd.read_excel(
    pd.ExcelFile(jpm_alts_market_values_filepath),
    sheet_name='Sheet1',
    skiprows=use_accountid,
    skipfooter=footnote_rows,
    header=1
)

# Reshapes the time-series into a panel.
df_jpm_alts_mv = ts_to_panel(
    df_ts=df_jpm_alts_mv,
    date='Unnamed: 0',
    set_index_column_name='Manager',
    set_value_column_name='Market Value'
)

# Joins the jpm main mv and jpm alts mv together
df_jpm_mv = pd.concat([df_jpm_main_mv, df_jpm_alts_mv]).reset_index(drop=True)

# Checks for str/int errors before converting returns to percentages.
string_error_list = []
for i in range(0, len(df_jpm_returns)):
    if isinstance(df_jpm_returns['JPM_Return'][i], str):
        string_error_list.append(df_jpm_returns['Manager'][i])

# Converts the returns to percentage.
df_jpm_returns['JPM_Return'] = df_jpm_returns['JPM_Return']/100
df_jpm_benchmarks['JPM_Benchmark'] = df_jpm_benchmarks['JPM_Benchmark']/100
df_jpm_returns = df_jpm_returns[~df_jpm_returns['JPM_Return'].isin([np.nan])].reset_index(drop=True)
df_jpm_benchmarks = df_jpm_benchmarks[~df_jpm_benchmarks['JPM_Benchmark'].isin([np.nan])].reset_index(drop=True)

# Creates Rf from Cash Aggregate Benchmark
df_jpm_rf = df_jpm_benchmarks[df_jpm_benchmarks['Manager'].isin(['CLFACASH', 'Cash Aggregate'])].reset_index(drop=True)
df_jpm_rf = df_jpm_rf.rename(columns={'JPM_Benchmark': 'JPM_Rf'})

# Infers the risk-free rate from the Cash +0.2% benchmark, the +0.2% benchmark started November 2019.
rf_values = []
new_cash_benchmark_date = dt.datetime(2019, 11, 30)
for i in range(0, len(df_jpm_rf)):
    if df_jpm_rf['Date'][i] >= new_cash_benchmark_date:
        rf_values.append(df_jpm_rf['JPM_Rf'][i] - (((1+0.002)**(1/12))-1))
    else:
        rf_values.append(df_jpm_rf['JPM_Rf'][i])
df_jpm_rf['JPM_Rf'] = rf_values
df_jpm_rf = df_jpm_rf.drop(columns=['Manager'], axis=1)

# Create ASX300 for regression
df_jpm_asx300 = df_jpm_benchmarks[df_jpm_benchmarks['Manager'].isin(['CEIAETOT', 'Australian Equities Aggregate'])].reset_index(drop=True)
df_jpm_asx300 = df_jpm_asx300.rename(columns={'JPM_Benchmark': 'JPM_ASX300'})
df_jpm_asx300 = df_jpm_asx300.drop(columns=['Manager'], axis=1)

# Merges returns and benchmarks
df_jpm = pd.merge(
    left=df_jpm_returns,
    right=df_jpm_benchmarks,
    left_on=['Manager', 'Date'],
    right_on=['Manager', 'Date'],
    how='inner'
)

# Merges market value, and returns and benchmarks
df_jpm = pd.merge(
    left=df_jpm_mv,
    right=df_jpm,
    left_on=['Manager', 'Date'],
    right_on=['Manager', 'Date'],
    how='right'
    )

# Merges returns, benchmarks, Rf, ASX300
df_jpm = pd.merge(
    left=df_jpm,
    right=df_jpm_rf,
    left_on=['Date'],
    right_on=['Date'],
    how='inner'
)

df_jpm = pd.merge(
    left=df_jpm,
    right=df_jpm_asx300,
    left_on=['Date'],
    right_on=['Date'],
    how='inner'
)


# Reads LGS's dictionary
df_lgs = pd.read_excel(
    pd.ExcelFile(lgs_dictionary_filepath),
    sheet_name='Sheet1',
    header=0
)
df_lgs = df_lgs.rename(columns={'JPM Account Id': 'Manager'})

df_jpm = pd.merge(
        left=df_jpm,
        right=df_lgs,
        left_on=['Manager'],
        right_on=['Manager'],
        how='inner'
)

# Keep only open accounts
df_jpm = df_jpm[df_jpm['LGS Open'] == 1].reset_index(drop=True)
df_jpm = df_jpm.drop(columns=['LGS Open'], axis=1)

# Keep only reported items
df_jpm_not_reported = df_jpm[df_jpm['JPM ReportName'].isin([np.nan])]
df_jpm = df_jpm[~df_jpm['JPM ReportName'].isin([np.nan])].reset_index(drop=True)

# Sort values VERY IMPORTANT for groupby merging
df_jpm = df_jpm.sort_values(['Manager', 'Date'], ascending=[True, True]).reset_index(drop=True)

# Sets the dictionary for the holding period returns.
horizon_to_period_dict = {
        '1_': 1,
        '3_': 3,
        'FYTD_': FYTD,
        '12_': 12,
        '36_': 36,
        '60_': 60,
        '84_': 84
}


# Calculates the holding period returns and annualises for periods greater than 12 months.
for horizon, period in horizon_to_period_dict.items():

    for column in ['Return', 'Benchmark', 'Rf']:

        column_name = horizon + column
        return_type = 'JPM_' + column

        if period <= 12:
            df_jpm[column_name] = (
                df_jpm
                .groupby(['Manager'])[return_type]
                .rolling(period)
                .apply(lambda r: np.prod(1+r)-1, raw=False)
                .reset_index(drop=False)[return_type]
            )

        elif period > 12:
            df_jpm[column_name] = (
                df_jpm
                .groupby(['Manager'])[return_type]
                .rolling(period)
                .apply(lambda r: (np.prod(1+r)**(12/period))-1, raw=False)
                .reset_index(drop=False)[return_type]
            )

    df_jpm[horizon + 'Excess'] = df_jpm[horizon + 'Return'] - df_jpm[horizon + 'Benchmark']

indices_problem = []
for i in range(0, len(df_jpm)):
    if abs(df_jpm['JPM_Return'][i] - df_jpm['1_Return'][i]) > 0.01:
        indices_problem.append(i)


# Calculates volatility
df_jpm['60_Volatility'] = (
    df_jpm
    .groupby(['Manager'])['1_Return']
    .rolling(60)
    .apply(lambda r: np.std(r, ddof=1)*np.sqrt(12), raw=False)
    .reset_index(drop=False)['1_Return']
)

# Calculates tracking error
df_jpm['60_Tracking_Error'] = (
    df_jpm
    .groupby(['Manager'])['1_Excess']
    .rolling(60)
    .apply(lambda r: np.std(r, ddof=1)*np.sqrt(12), raw=False)
    .reset_index(drop=False)['1_Excess']
)

# Calculates sharpe ratio
df_jpm['60_Sharpe'] = (df_jpm['60_Return'] - df_jpm['60_Rf']) / df_jpm['60_Volatility']

# Calculates information ratio
df_jpm['60_Information'] = df_jpm['60_Excess'] / df_jpm['60_Tracking_Error']

df_sectors_today = df_jpm[df_jpm['Date'].isin([report_date]) & df_jpm['LGS Sector Aggregate'].isin([1])].sort_values(['LGS Asset Class Order'])

df_sectors_today.to_csv('C:/Users/mnguyen/LGSS/Investments Team - SandPits - SandPits/data/output/lgs/reports/active/active.csv', index=False)
