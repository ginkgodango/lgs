import os
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import statsmodels.api as sm

# START USER INPUT DATA
jpm_main_returns_filepath = 'U:/Shared/Investment Operations/Performance/IC Paper Performance Review/#Data/input/jpm/performance/2020/08/Historical Time Series - Monthly - Main Returns.xlsx'
jpm_alts_returns_filepath = 'U:/Shared/Investment Operations/Performance/IC Paper Performance Review/#Data/input/jpm/performance/2020/08/Historical Time Series - Monthly - Alts Returns.xlsx'
jpm_main_benchmarks_filepath = 'U:/Shared/Investment Operations/Performance/IC Paper Performance Review/#Data/input/jpm/performance/2020/08/Historical Time Series - Monthly - Main Benchmarks.xlsx'
jpm_alts_benchmarks_filepath = 'U:/Shared/Investment Operations/Performance/IC Paper Performance Review/#Data/input/jpm/performance/2020/08/Historical Time Series - Monthly - Alts Benchmarks.xlsx'
jpm_main_market_values_filepath = 'U:/Shared/Investment Operations/Performance/IC Paper Performance Review/#Data/input/jpm/performance/2020/08/Historical Time Series - Monthly - Main Market Values.xlsx'
jpm_alts_market_values_filepath = 'U:/Shared/Investment Operations/Performance/IC Paper Performance Review/#Data/input/jpm/performance/2020/08/Historical Time Series - Monthly - Alts Market Values.xlsx'
lgs_dictionary_filepath = 'U:/Shared/Investment Operations/Performance/IC Paper Performance Review/#Data/input/lgs/dictionary/2020/08/New Dictionary_v11.xlsx'
FYTD = 2
report_date = dt.datetime(2020, 8, 31)
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
df_jpm['36_Volatility'] = (
    df_jpm
    .groupby(['Manager'])['1_Return']
    .rolling(36)
    .apply(lambda r: np.std(r, ddof=1)*np.sqrt(12), raw=False)
    .reset_index(drop=False)['1_Return']
)

# Calculates tracking error
df_jpm['36_Tracking_Error'] = (
    df_jpm
    .groupby(['Manager'])['1_Excess']
    .rolling(36)
    .apply(lambda r: np.std(r, ddof=1)*np.sqrt(12), raw=False)
    .reset_index(drop=False)['1_Excess']
)

# Calculates sharpe ratio
df_jpm['36_Sharpe'] = (df_jpm['36_Return'] - df_jpm['36_Rf']) / df_jpm['36_Volatility']

# Calculates information ratio
df_jpm['36_Information'] = df_jpm['36_Excess'] / df_jpm['36_Tracking_Error']

# Calculates rolling betas
def rolling_ols(indices, result, ycol, xcols):
    roll_df = df.loc[indices] # get relevant data frame subset
    result[indices[-1]] = (sm.OLS(roll_df[ycol], sm.add_constant(roll_df[xcols]), hasconst=True).fit().params)[-1]
    return 0


# Creates container and writes results of regression beta to result: idx: beta
kwargs = {
    "xcols": ['JPM_ASX300'],
    "ycol": '1_Return',
    "result": {}
}

# iterate id's sub data frames and call rolling_ols for rolling windows
df = df_jpm.copy()
df["identifier"] = df.index
for idx, sub_df in df.groupby("Manager"):
    sub_df["identifier"].rolling(36).apply(rolling_ols, kwargs=kwargs, raw=True)

# write results back to original df
df_jpm['36_Beta'] = pd.Series(kwargs["result"])
# End calculation of rolling beta

# Creates a copy for df_jpm
df_jpm_combined = df_jpm.copy()

# Collects and deletes duplicates (Due to Sector Aggregate Problem) REMOVE THIS AFTER JPM FIXES THE DATA
df_jpm_combined_duplicated = df_jpm_combined[df_jpm_combined.duplicated(['LGS Name', 'Date'])]
df_jpm_combined = df_jpm_combined[~df_jpm_combined.duplicated(['LGS Name', 'Date'])].reset_index(drop=True)

# ACTIVE CONTRIBUTION
remove_double_count = []
df_jpm_combined['12_Average_Market_Value'] = (
    df_jpm_combined[~df_jpm_combined['Manager'].isin([remove_double_count])]
    .groupby(['Manager'])['Market Value']
    .rolling(12)
    .mean()
    .reset_index(drop=True)
)


# df_test = df_jpm_combined[['Date', 'Manager', 'Market Value', 'LGS Sector Aggregate']]
# df_test = df_test[df_test['LGS Sector Aggregate'].isin([0])]
# df_test['Total Market Value'] = (
#     df_test
#     .groupby(['Date'])['Market Value']
#     .sum()
#     .reset_index(drop=True)
# )
# df_test = df_test[['Date', 'Total Market Value']]
# df_test = df_test[~df_test['Total Market Value'].isin([np.nan])]
#
# df_jpm_combined_test = pd.merge(
#     left=df_jpm_combined,
#     right=df_test,
#     left_on=['Date'],
#     right_on=['Date'],
#     how='outer'
# )


# df_jpm_main['Total Market_Value'] = (
#     df_jpm_main[~df_jpm_main['LGS Sector Aggregate'].isin([1])]
#     .groupby(['Date'])['Market Value']
#     .transform('sum')
#     .reset_index(drop=True)
# )


# # Counts only fund managers
# df_jpm_combined_total_market_value_fund_managers = (
#     df_jpm_combined[df_jpm_combined['LGS Sector Aggregate'].isin([0])]
#     .groupby(['Date'])['Market Value']
#     .sum()
# )


# Calculates total market value as average of total asset class market value and fund manager market value
df_jpm_combined['Total Market_Value'] = (
    df_jpm_combined
    .groupby(['Date'])['Market Value']
    .transform('sum')
    .reset_index(drop=True)/2
)

df_jpm_combined['12_Active_Contribution'] = (
        (df_jpm_combined['12_Average_Market_Value'] / df_jpm_combined['Total Market_Value']) * (df_jpm_combined['12_Excess'])
)


# SUSTAINABILITY
df_jpm_combined['Sustainability Value'] = df_jpm_combined['LGS Sustainability Weight'] * df_jpm_combined['Market Value']
df_jpm_combined['Sustainability Total Value'] = df_jpm_combined.groupby(['Date'])['Sustainability Value'].transform('sum').reset_index(drop=True)
df_jpm_combined['Sustainability Total Weight'] = df_jpm_combined['Sustainability Value']/df_jpm_combined['Sustainability Total Value']
df_jpm_combined['Sustainability Total Total Weight'] = df_jpm_combined.groupby(['Date'])['Sustainability Total Weight'].transform('sum').reset_index(drop=True)

# SWITCHES MANAGER WITH LGS NAME, ALSO RENAMES RISK METRICS
df_jpm_combined = df_jpm_combined.drop(['Manager'], axis=1)
df_jpm_combined = df_jpm_combined.rename(
        columns={
                'LGS Name': 'Manager',
                '36_Tracking_Error': 'Tracking Error',
                '36_Volatility': 'Volatility',
                '36_Information': 'Information',
                '36_Sharpe': 'Sharpe',
                '36_Beta': 'Beta'
        }
)

df_jpm_combined = df_jpm_combined.sort_values(['LGS Asset Class Order', 'LGS Manager Order']).reset_index(drop=True)

# Deletes duplicates
# df_jpm_main_duplicated = df_jpm_main[df_jpm_main.duplicated(['Manager', 'Date'])]
# df_jpm_main = df_jpm_main[~df_jpm_main.duplicated(['Manager', 'Date'])].reset_index(drop=True)


# CREATES LATEX TABLES AND CHARTS
writer = pd.ExcelWriter('U:/CIO/#Data/output/investment/checker/investment_container.xlsx', engine='xlsxwriter')

# Selects rows as at report date and filters liquidity accounts
df_jpm_table = df_jpm_combined[(df_jpm_combined['Date'] == report_date) & (df_jpm_combined['LGS Liquidity'] == 0)].reset_index(drop=True)

# Sets list of columns for each table
columns_lead = ['Manager', 'Market Value']
columns_indicators = ['LGS Asset Class Level 2', 'LGS Sector Aggregate', 'JPM ReportName']
columns_performance = []
for horizon, period in horizon_to_period_dict.items():
    for column in ['Return', 'Excess']:
        columns_performance.append(horizon + column)
columns_risk = ['Tracking Error', 'Volatility', 'Information', 'Sharpe', 'Beta']
columns_active_contribution = ['12_Active_Contribution']
columns_millions = ['Market Value']
columns_decimal = columns_performance + columns_risk[:2] + columns_active_contribution
columns_round = columns_millions + columns_decimal + columns_risk + columns_active_contribution

# Selects columns for Latex Tables
df_jpm_table = df_jpm_table[
        columns_lead +
        columns_performance +
        columns_risk +
        columns_active_contribution +
        columns_indicators
]

# Converts market value into millions and decimal into percentage
df_jpm_table[columns_millions] = df_jpm_table[columns_millions] / 1000000
df_jpm_table[columns_decimal] = df_jpm_table[columns_decimal] * 100
df_jpm_table[columns_round] = df_jpm_table[columns_round].round(2)

# Creates column hierarchy for performance table
columns_performance_multilevel1 = pd.MultiIndex.from_product([[''], ['Manager', 'LGS Sector Aggregate', 'LGS Asset Class Level 2']])
columns_performance_multilevel2 = pd.MultiIndex.from_product([['Market Value'], ['$Mills']])
columns_performance_multilevel3 = pd.MultiIndex.from_product([['1 Month', '3 Month', 'FYTD', '1 Year', '3 Year', '5 Year', '7 Year'], ['LGS', 'Active']])

# Creates performance tables
df_jpm_table_performance1 = df_jpm_table[['Manager', 'LGS Sector Aggregate', 'LGS Asset Class Level 2']]
df_jpm_table_performance2 = df_jpm_table[['Market Value']]
df_jpm_table_performance3 = df_jpm_table[columns_performance]
df_jpm_table_performance1.columns = columns_performance_multilevel1
df_jpm_table_performance2.columns = columns_performance_multilevel2
df_jpm_table_performance3.columns = columns_performance_multilevel3
df_jpm_table_performance = pd.concat(
    [
        df_jpm_table_performance1,
        df_jpm_table_performance2,
        df_jpm_table_performance3
        ],
    axis=1
)

# del df_jpm_table_performance1
# del df_jpm_table_performance2
# del df_jpm_table_performance3

df_jpm_table_performance_sector = df_jpm_table_performance[df_jpm_table_performance[('', 'LGS Sector Aggregate')].isin([1])].reset_index(drop=True)
df_jpm_table_performance_sector = df_jpm_table_performance_sector.drop(('', 'LGS Sector Aggregate'), axis=1)
df_jpm_table_performance_sector = df_jpm_table_performance_sector.drop(('', 'LGS Asset Class Level 2'), axis=1)

with open('U:/CIO/#Data/output/investment/returns/Sector.tex', 'w') as tf:
    latex_string_temp = df_jpm_table_performance_sector.to_latex(index=False, na_rep='', multicolumn_format='c', column_format='lRRRRRRRRRRRRRRRRR')
    tf.write(latex_string_temp)

df_jpm_table_performance_sector.to_excel(writer, sheet_name='sector_returns')

asset_class_to_performance_dict = dict(list(df_jpm_table_performance.groupby([('', 'LGS Asset Class Level 2')])))
for asset_class, df_temp in asset_class_to_performance_dict.items():
    df_temp = df_temp.drop(('', 'LGS Asset Class Level 2'), axis=1)
    df_temp = df_temp.drop(('', 'LGS Sector Aggregate'), axis=1)

    # Removes managers from PE, OA, and DA tables
    if asset_class in ['PE', 'OA', 'DA']:
        df_temp = df_temp[df_temp[('', 'Manager')].isin(['Private Equity', 'Opportunistic Alternatives', 'Attunga', 'Defensive Alternatives'])]

    with open('U:/CIO/#Data/output/investment/returns/' + str(asset_class) + '.tex', 'w') as tf:
        latex_string_temp = (
                df_temp
                .to_latex(index=False, na_rep='', multicolumn_format='c', column_format='lRRRRRRRRRRRRRRRRR')
                .replace('-0.00', '0.00')
        )
        tf.write(latex_string_temp)

    df_temp.to_excel(writer, sheet_name=asset_class + '_returns')

# Creates risk table
df_jpm_table_risk = df_jpm_table[columns_lead + columns_risk + ['LGS Sector Aggregate', 'LGS Asset Class Level 2']]

df_jpm_table_risk_sector = df_jpm_table_risk[df_jpm_table_risk['LGS Sector Aggregate'].isin([1])].reset_index(drop=True)
df_jpm_table_risk_sector = df_jpm_table_risk_sector.drop(['LGS Sector Aggregate', 'LGS Asset Class Level 2'], axis=1)

with open('U:/CIO/#Data/output/investment/risk/Sector.tex', 'w') as tf:
    latex_string_temp = df_jpm_table_risk_sector.to_latex(index=False, na_rep='', multicolumn_format='c', column_format='lRRRRRRRRRRRRRRRRR')
    tf.write(latex_string_temp)

df_jpm_table_risk_sector.to_excel(writer, sheet_name='sector_risk')

asset_class_to_risk_dict = dict(list(df_jpm_table_risk.groupby(['LGS Asset Class Level 2'])))
for asset_class, df_temp in asset_class_to_risk_dict.items():
    df_temp = df_temp.drop(['LGS Sector Aggregate', 'LGS Asset Class Level 2'], axis=1)

    # Removes SRI from risk tables
    if asset_class in ['AE', 'IE']:
        df_temp = df_temp[~df_temp['Manager'].isin(['Domestic SRI', 'International SRI'])]

    with open('U:/CIO/#Data/output/investment/risk/' + str(asset_class) + '.tex', 'w') as tf:
        latex_string_temp = (
                df_temp
                .to_latex(index=False, na_rep='', multicolumn_format='c', column_format='lRRRRRR')
                .replace('-0.00', '0.00')
        )
        tf.write(latex_string_temp)

    df_temp.to_excel(writer, sheet_name=asset_class + '_risk')

# Creates active contribution table
df_jpm_table_active_contribution = df_jpm_table[columns_lead[:1] + columns_active_contribution + ['LGS Sector Aggregate']]
df_jpm_table_active_contribution = df_jpm_table_active_contribution[~df_jpm_table_active_contribution['LGS Sector Aggregate'].isin([1])].reset_index(drop=True)
df_jpm_table_active_contribution = df_jpm_table_active_contribution.drop(columns=['LGS Sector Aggregate'], axis=0)
df_jpm_table_active_contribution = df_jpm_table_active_contribution.sort_values(['12_Active_Contribution'], ascending=False).reset_index(drop=True)
df_jpm_table_active_contribution_missing = df_jpm_table_active_contribution[df_jpm_table_active_contribution['12_Active_Contribution'].isin([np.nan])]
df_jpm_table_active_contribution = df_jpm_table_active_contribution[~df_jpm_table_active_contribution['12_Active_Contribution'].isin([np.nan])]

df_jpm_table_active_contribution_top = df_jpm_table_active_contribution[:10].reset_index(drop=True)
df_jpm_table_active_contribution_top = df_jpm_table_active_contribution_top.rename(columns={'12_Active_Contribution': 'Contribution'})
df_jpm_table_active_contribution_bottom = df_jpm_table_active_contribution[-10:].reset_index(drop=True)
df_jpm_table_active_contribution_bottom = df_jpm_table_active_contribution_bottom.rename(columns={'12_Active_Contribution': 'Detraction'})

df_jpm_table_active_contribution_combined = pd.concat([df_jpm_table_active_contribution_top, df_jpm_table_active_contribution_bottom], axis=1, sort=False)
df_jpm_table_active_contribution_combined.to_csv('U:/CIO/#Data/output/investment/contributors/contributors.csv', index=False)
with open('U:/CIO/#Data/output/investment/contributors/contributors.tex', 'w') as tf:
    tf.write(df_jpm_table_active_contribution_combined.to_latex(index=False, na_rep='', column_format='lRlR'))

df_jpm_table_active_contribution_combined.to_excel(writer, sheet_name='active contribution')

# Creates charts
df_jpm_chart_12_excess = df_jpm_combined[['Manager', 'Date', '12_Excess', 'LGS Sector Aggregate', 'LGS Asset Class Level 2', 'LGS Liquidity']]
df_jpm_chart_12_excess = df_jpm_chart_12_excess[
    (df_jpm_chart_12_excess['LGS Sector Aggregate'] == 0) |
    (df_jpm_chart_12_excess['Manager'].isin(['Absolute Return', 'Bonds Aggregate', 'Legacy Private Equity']))
].reset_index(drop=True)
df_jpm_chart_12_excess = df_jpm_chart_12_excess[df_jpm_chart_12_excess['LGS Liquidity'] == 0].reset_index(drop=True)
df_jpm_chart_12_excess = df_jpm_chart_12_excess.drop(columns=['LGS Sector Aggregate'], axis=1)
df_jpm_chart_12_excess['12_Excess'] = df_jpm_chart_12_excess['12_Excess']*100
asset_class_to_chart_12_dict = dict(list(df_jpm_chart_12_excess.groupby(['LGS Asset Class Level 2'])))

df_jpm_chart_60_excess = df_jpm_combined[['Manager', 'Date', '60_Excess', 'LGS Sector Aggregate', 'LGS Asset Class Level 2', 'LGS Liquidity']]
df_jpm_chart_60_excess = df_jpm_chart_60_excess[
    (df_jpm_chart_60_excess['LGS Sector Aggregate'] == 1) |
    (df_jpm_chart_60_excess['Manager'] == 'FX Overlay')
].reset_index(drop=True)
df_jpm_chart_60_excess = df_jpm_chart_60_excess.drop(columns=['LGS Sector Aggregate'], axis=1)
df_jpm_chart_60_excess['60_Excess'] = df_jpm_chart_60_excess['60_Excess']*100
asset_class_to_chart_60_dict = dict(list(df_jpm_chart_60_excess.groupby(['LGS Asset Class Level 2'])))

df_jpm_chart_mv = df_jpm_combined[['Manager', 'Date', 'Market Value', 'LGS Sector Aggregate', 'LGS Asset Class Level 2', 'LGS Liquidity']]
df_jpm_chart_mv = df_jpm_chart_mv[df_jpm_chart_mv['Date'] == report_date].reset_index(drop=True)
df_jpm_chart_mv = df_jpm_chart_mv[
    (df_jpm_chart_mv['LGS Sector Aggregate'] == 0) |
    (df_jpm_chart_mv['Manager'].isin(['Absolute Return', 'Bonds Aggregate', 'Legacy Private Equity']))
].reset_index(drop=True)
df_jpm_chart_mv = df_jpm_chart_mv[df_jpm_chart_mv['LGS Liquidity'] == 0].reset_index(drop=True)
df_jpm_chart_mv = df_jpm_chart_mv.drop(columns=['LGS Sector Aggregate'], axis=1)
asset_class_to_chart_mv_dict = dict(list(df_jpm_chart_mv.groupby(['LGS Asset Class Level 2'])))

asset_classes1 = list(asset_class_to_chart_12_dict.keys())
asset_classes2 = list(asset_class_to_chart_60_dict.keys())
# assert asset_classes1 == asset_classes2
asset_classes = asset_classes2
for asset_class in asset_classes:

    fig, axes = plt.subplots(
            nrows=1,
            ncols=3,
            sharex=False,
            sharey=False,
            gridspec_kw={'width_ratios': [2, 2, 1]},
            figsize=(16.8, 3.6)
            )

    df_chart_12_temp = asset_class_to_chart_12_dict[asset_class]
    df_chart_12_temp = df_chart_12_temp.pivot(index='Date', columns='Manager', values='12_Excess')[-60:]
    df_chart_12_temp.plot(ax=axes[0], linewidth=1)
    axes[0].set_title('Manager 1-yr Rolling Active Returns')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('Active Return %')
    axes[0].margins(x=0)
    axes[0].axhline(y=0, linestyle=':', linewidth=1, color='k',)
    axes[0].legend(loc='lower left', title='')

    df_chart_60_temp = asset_class_to_chart_60_dict[asset_class]
    df_chart_60_temp = df_chart_60_temp.pivot(index='Date', columns='Manager', values='60_Excess')[-60:]
    df_chart_60_temp.plot(ax=axes[1], linewidth=1)
    axes[1].set_title('Sector Annualized 5-yr Rolling Active Returns')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('Active Return %')
    axes[1].margins(x=0)
    axes[1].axhline(y=0, linestyle=':', linewidth=1, color='k',)
    axes[1].legend(loc='lower left', title='')

    df_chart_mv_temp = asset_class_to_chart_mv_dict[asset_class]
    df_chart_mv_temp = df_chart_mv_temp.set_index('Manager').sort_index()

    try:
        axes[2] = df_chart_mv_temp.plot(kind='pie', ax=axes[2], y='Market Value', autopct='%1.0f%%', legend=None, title='Holdings')
    except ValueError:
        print(asset_class, ' has negative values! Using absolute market values for pie chart.')
        df_chart_mv_temp['Market Value'] = abs(df_chart_mv_temp['Market Value'])
        axes[2] = df_chart_mv_temp.plot(kind='pie', ax=axes[2], y='Market Value', autopct='%1.0f%%', legend=None, title='Holdings')

    my_circle = plt.Circle((0, 0), 0.75, color='white')
    axes[2].add_artist(my_circle)
    axes[2].set_ylabel(None)

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    fig.savefig('U:/CIO/#Data/output/investment/charts/' + str(asset_class) + '.png', dpi=300)


# Creates Manager Allocations table
df_jpm_manager_allocations = df_jpm_combined[df_jpm_combined['Date'] == report_date].reset_index(drop=True)
df_jpm_manager_allocations = df_jpm_manager_allocations[['Manager', 'Market Value', 'LGS Target Weight', 'LGS Asset Class Level 2']]
df_jpm_manager_allocations = df_jpm_manager_allocations[~df_jpm_manager_allocations['LGS Target Weight'].isin([np.nan])].reset_index(drop=True)

df_jpm_manager_allocations['Asset Class Market Value'] = df_jpm_manager_allocations.groupby(['LGS Asset Class Level 2'])['Market Value'].transform('sum').reset_index(drop=True)
df_jpm_manager_allocations['Weight'] = df_jpm_manager_allocations['Market Value']/df_jpm_manager_allocations['Asset Class Market Value']
df_jpm_manager_allocations['Deviation'] = df_jpm_manager_allocations['Weight'] - df_jpm_manager_allocations['LGS Target Weight']
df_jpm_manager_allocations = df_jpm_manager_allocations[['Manager', 'Weight', 'LGS Target Weight', 'Deviation']]
df_jpm_manager_allocations[['Weight', 'LGS Target Weight', 'Deviation']] = (df_jpm_manager_allocations[['Weight', 'LGS Target Weight', 'Deviation']]*100).round(1)
df_jpm_manager_allocations = df_jpm_manager_allocations.rename(columns={'LGS Target Weight': 'Target'})
df_jpm_manager_allocations1 = df_jpm_manager_allocations[:25].reset_index(drop=True)
df_jpm_manager_allocations2 = df_jpm_manager_allocations[25:].reset_index(drop=True)
df_jpm_manager_allocations3 = pd.concat([df_jpm_manager_allocations1, df_jpm_manager_allocations2], axis=1)

df_jpm_manager_allocations3.to_latex('U:/CIO/#Data/output/investment/manager/manager_allocations.tex', index=False, na_rep='', column_format='lRRRlRRR')
df_jpm_manager_allocations3.to_excel(writer, sheet_name='manager_allocations', index=False)

# Creates sustainability esg table
df_jpm_combined_esg = df_jpm_combined[
    [
        'Date',
        'Manager',
        'Market Value',
        'LGS Sustainability Weight',
        'Sustainability Value',
        'Sustainability Total Weight',
        '1_Return',
        '1_Excess',
        '3_Return',
        '3_Excess',
        '12_Return',
        '12_Excess',
        '36_Return',
        '36_Excess'
    ]
]

df_jpm_combined_esg = df_jpm_combined_esg[(df_jpm_combined_esg['LGS Sustainability Weight'] != 0) & (df_jpm_combined_esg['Date'] == report_date)].reset_index(drop=True)

# Create function to create totals row in sustainability table
def esg_total_row(data):
    d = dict()
    d['Date'] = np.max(data['Date'])
    d['Manager'] = 'Total'
    d['Market Value'] = np.sum(data['Market Value'])
    d['LGS Sustainability Weight'] = np.nan
    d['Sustainability Value'] = np.sum(data['Sustainability Value'])
    d['Sustainability Total Weight'] = np.sum(data['Sustainability Total Weight'])
    d['1_Return'] = np.sum(data['1_Return'] * data['Sustainability Total Weight'])
    d['1_Excess'] = np.sum(data['1_Excess'] * data['Sustainability Total Weight'])
    d['3_Return'] = np.sum(data['3_Return'] * data['Sustainability Total Weight'])
    d['3_Excess'] = np.sum(data['3_Excess'] * data['Sustainability Total Weight'])
    d['12_Return'] = np.sum(data['12_Return'] * data['Sustainability Total Weight'])
    d['12_Excess'] = np.sum(data['12_Excess'] * data['Sustainability Total Weight'])
    d['36_Return'] = np.sum(data['36_Return'] * data['Sustainability Total Weight'])
    d['36_Excess'] = np.sum(data['36_Excess'] * data['Sustainability Total Weight'])
    return pd.Series(d)


# Calculate the total row by applying the esg_total_row function
df_jpm_combined_esg_total = df_jpm_combined_esg.groupby('Date').apply(esg_total_row).reset_index(drop=True)

# Sorts by manager name
df_jpm_combined_esg = df_jpm_combined_esg.sort_values(['Manager'])

# Joins the total row to the esg table
df_jpm_combined_esg = pd.concat([df_jpm_combined_esg, df_jpm_combined_esg_total])

# Drops date column
df_jpm_combined_esg = df_jpm_combined_esg.drop(columns=['Date'])

# Formats sustainability table
df_jpm_combined_esg[['Market Value', 'Sustainability Value']] = df_jpm_combined_esg[['Market Value', 'Sustainability Value']]/1000000

# Rounds to 2 decimal places
columns_decimal_esg = [
        'LGS Sustainability Weight',
        'Sustainability Total Weight',
        '1_Return',
        '1_Excess',
        '3_Return',
        '3_Excess',
        '12_Return',
        '12_Excess',
        '36_Return',
        '36_Excess'
    ]
df_jpm_combined_esg[columns_decimal_esg] = df_jpm_combined_esg[columns_decimal_esg]*100
df_jpm_combined_esg = df_jpm_combined_esg.round(2)

# Creates column hierarchy for performance table
columns_esg_multilevel1 = pd.MultiIndex.from_product([[''], ['Manager']])
columns_esg_multilevel2 = pd.MultiIndex.from_product([['Market Value'], ['($Mills)']])
columns_esg_multilevel3 = pd.MultiIndex.from_product([['Sustainability'], ['Weight']])
columns_esg_multilevel4 = pd.MultiIndex.from_product([['Sustainability'], ['Value ($Mills)']])
columns_esg_multilevel5 = pd.MultiIndex.from_product([['Total'], ['Weight']])
columns_esg_multilevel6 = pd.MultiIndex.from_product([['1 Month', '3 Month', '1 Year', '3 Year'], ['LGS', 'Active']])

# Creates performance tables
df_jpm_table_esg1 = df_jpm_combined_esg[['Manager']]
df_jpm_table_esg2 = df_jpm_combined_esg[['Market Value']]
df_jpm_table_esg3 = df_jpm_combined_esg[['LGS Sustainability Weight']]
df_jpm_table_esg4 = df_jpm_combined_esg[['Sustainability Value']]
df_jpm_table_esg5 = df_jpm_combined_esg[['Sustainability Total Weight']]
df_jpm_table_esg6 = df_jpm_combined_esg[['1_Return', '1_Excess', '3_Return', '3_Excess', '12_Return', '12_Excess', '36_Return', '36_Excess']]
df_jpm_table_esg1.columns = columns_esg_multilevel1
df_jpm_table_esg2.columns = columns_esg_multilevel2
df_jpm_table_esg3.columns = columns_esg_multilevel3
df_jpm_table_esg4.columns = columns_esg_multilevel4
df_jpm_table_esg5.columns = columns_esg_multilevel5
df_jpm_table_esg6.columns = columns_esg_multilevel6

# Joins the 6 tables together to create the esg table
df_jpm_table_esg = pd.concat(
    [
        df_jpm_table_esg1,
        df_jpm_table_esg2,
        df_jpm_table_esg3,
        df_jpm_table_esg4,
        df_jpm_table_esg5,
        df_jpm_table_esg6,
        ],
    axis=1
)

# Outputs the esg table
with open('U:/CIO/#Data/output/investment/sustainability/sustainability.tex', 'w') as tf:
    tf.write(df_jpm_table_esg.to_latex(index=False, na_rep='', multicolumn_format='c', column_format='lRRRRRRRRRRRR'))

df_jpm_table_esg.to_excel(writer, sheet_name='sustainability')
writer.save()

# Outputs the tables for checking
df_jpm_table.to_csv('U:/CIO/#Data/output/investment/checker/lgs_table.csv', index=False)
df_jpm_combined.to_csv('U:/CIO/#Data/output/investment/checker/lgs_combined.csv', index=False)
