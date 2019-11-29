import os
import pandas as pd
import numpy as np
import datetime as dt

jpm_filepath = 'U:/CIO/#Investment_Report/Data/input/testing/20191031 JPM Historical Time Series.xlsx'
jpm_iap_filepath = 'U:/CIO/#Investment_Report/Data/input/testing/jpm_iap/'
lgs_dictionary_filepath = 'U:/CIO/#Investment_Report/Data/input/testing/20191031 New Dictionary.xlsx'
FYTD = 12
report_date = dt.datetime(2019, 6, 30)

# Imports the JPM time-series.
jpm_xlsx = pd.ExcelFile(jpm_filepath)
use_managerid = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 14, 15]
use_accountid = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15]
footnote_rows = 28

df_jpm = pd.read_excel(
        jpm_xlsx,
        sheet_name='Sheet1',
        skiprows=use_accountid,
        skipfooter=footnote_rows,
        header=1
        )

# Reshapes the time-series into a panel.
df_jpm = df_jpm.rename(columns={'Unnamed: 0': 'Date'})
df_jpm = df_jpm.set_index('Date')
df_jpm = df_jpm.transpose()
df_jpm = df_jpm.reset_index(drop=False)
df_jpm = df_jpm.rename(columns={'index': 'Manager'})
df_jpm = pd.melt(df_jpm, id_vars=['Manager'], value_name='Return_JPM')
df_jpm = df_jpm.sort_values(['Manager', 'Date'])
df_jpm = df_jpm.reset_index(drop=True)

# Cleans the data and converts the returns to percentage.
df_jpm = df_jpm.replace('-', np.NaN)
df_jpm['Return_JPM'] = df_jpm['Return_JPM']/100

# Splits df_jpm into a returns and benchmarks
df_jpm_returns = df_jpm[~df_jpm.Manager.str.endswith('.1')].reset_index(drop=True)
df_jpm_benchmarks = df_jpm[df_jpm.Manager.str.endswith('.1')].reset_index(drop=True)
df_jpm_benchmarks['Manager'] = [df_jpm_benchmarks['Manager'][i][:-2] for i in range(0, len(df_jpm_benchmarks))]
df_jpm_benchmarks = df_jpm_benchmarks.rename(columns={'Return_JPM': 'Benchmark_JPM'})

# Merges returns and benchmarks
df_jpm = pd.merge(
        left=df_jpm_returns,
        right=df_jpm_benchmarks,
        left_on=['Manager', 'Date'],
        right_on=['Manager', 'Date'],
        how='inner'
)

# Deletes the redundant dataframes.
del df_jpm_returns
del df_jpm_benchmarks

# Reads LGS's dictionary
lgs_xlsx = pd.ExcelFile(lgs_dictionary_filepath)
df_lgs = pd.read_excel(
        lgs_xlsx,
        sheet_name='Sheet1',
        header=0
        )
df_lgs = df_lgs.rename(
        columns={
                'JPM BookName': 'BookName',
                'JPM Account Id': 'Manager'
        }
)

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

# Sets the dictionary for the holding period returns.
horizon_to_period_dict = {
        '1_': 1,
        '3_': 3,
        'FYTD_': FYTD,
        '12_': 12,
        '24_': 24,
        '36_': 36,
        '60_': 60,
        '84_': 84
}

# Calculates the holding period returns and annualises for periods greater than 12 months.
for horizon, period in horizon_to_period_dict.items():

    for column in ['Return', 'Benchmark']:

        column_name = horizon + column
        return_type = column + '_JPM'

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

# Calculates volatility, tracking error, sharpe ratio, information ratio
df_jpm['36_Volatility'] = (
    df_jpm
    .groupby(['Manager'])['1_Return']
    .rolling(36)
    .apply(lambda r: np.std(r, ddof=1)*np.sqrt(12), raw=False)
    .reset_index(drop=False)['1_Return']
)

df_jpm['36_Tracking_Error'] = (
    df_jpm
    .groupby(['Manager'])['1_Benchmark']
    .rolling(36)
    .apply(lambda r: np.std(r, ddof=1)*np.sqrt(12), raw=False)
    .reset_index(drop=False)['1_Benchmark']
)

df_jpm['36_Sharpe_Ratio'] = np.nan
# df_jpm['36_Sharpe_Ratio'] = (df_jpm['36_Return'] - df_jpm['Rf']) / df_jpm['36_Volatility']

df_jpm['36_Information_Ratio'] = df_jpm['36_Excess'] / df_jpm['36_Tracking_Error']

#df_jpm.to_csv('U:/CIO/#Investment_Report/Data/output/verification/jpm_calculate.csv', index=False)

# Import JPM_IAP, Accounts; By ID; Include Closed Accounts; Select All; Mode: Portfolio Only
jpm_iap_filenames = sorted(os.listdir(jpm_iap_filepath))
df_jpm_iap = pd.DataFrame()
for filename in jpm_iap_filenames:
    jpm_iap_xlsx = pd.ExcelFile(jpm_iap_filepath + filename)
    df_jpm_iap_temp = pd.read_excel(
        jpm_iap_xlsx,
        sheet_name='Sheet1',
        skiprows=[0, 1],
        header=0
    )
    df_jpm_iap_temp['Date'] = dt.datetime(int(filename[:4]), int(filename[4:6]), int(filename[6:8]))
    df_jpm_iap = pd.concat([df_jpm_iap, df_jpm_iap_temp], sort=False)

df_jpm_iap = df_jpm_iap.rename(columns={'Account Id': 'Manager'}).reset_index(drop=True)
df_jpm_iap = df_jpm_iap[['Manager', 'Date', 'Market Value']]

# Merges the market values from JPM IAP with JPM HTS
df_jpm_main = pd\
    .merge(
        left=df_jpm_iap,
        right=df_jpm,
        left_on=['Manager', 'Date'],
        right_on=['Manager', 'Date'],
        how='right'
    )\
    .sort_values(['Manager', 'Date'])\
    .reset_index(drop=True)


# ACTIVE CONTRIBUTION
remove_double_count = ['']
df_jpm_main['12_Average_Market_Value'] = (
    df_jpm_main[~df_jpm_main['Manager'].isin([remove_double_count])]
    .groupby(['Manager'])['Market Value']
    .rolling(12)
    .mean()
    .reset_index(drop=True)
)

df_jpm_main['Total Market_Value'] = (
    df_jpm_main[~df_jpm_main['LGS Sector Aggregate'].isin([1])]
    .groupby(['Date'])['Market Value']
    .transform('sum')
    .reset_index(drop=True)
)

df_jpm_main['12_Active_Contribution'] = (
        (df_jpm_main['12_Average_Market_Value'] / df_jpm_main['Total Market_Value']) * (df_jpm_main['12_Excess'])
)


# SWITCHES MANAGER WITH LGS NAME, ALSO RENAMES RISK METRICS
df_jpm_main = df_jpm_main.drop(['Manager'], axis=1)
df_jpm_main = df_jpm_main.rename(
        columns={
                'LGS Name': 'Manager',
                '36_Tracking_Error': 'Tracking Error',
                '36_Volatility': 'Volatility',
                '36_Information_Ratio': 'Information Ratio',
                '36_Sharpe_Ratio': 'Sharpe Ratio'
        }
)

# Deletes duplicates
df_jpm_main_duplicated = df_jpm_main[df_jpm_main.duplicated(['Manager', 'Date'])]
df_jpm_main = df_jpm_main[~df_jpm_main.duplicated(['Manager', 'Date'])].reset_index(drop=True)

# CREATES LATEX TABLES AND CHARTS
# Selects rows as at report date
df_jpm_table = df_jpm_main[df_jpm_main['Date'] == report_date].reset_index(drop=True)

# Sets list of columns for each table
columns_lead = ['Manager', 'Market Value']
columns_indicators = ['LGS Asset Class', 'LGS Sector Aggregate']
columns_performance = []
for horizon, period in horizon_to_period_dict.items():
    for column in ['Return', 'Excess']:
        columns_performance.append(horizon + column)
columns_risk = ['Tracking Error', 'Volatility', 'Information Ratio', 'Sharpe Ratio']
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
columns_performance_lead_multilevel = pd.MultiIndex.from_product([[''], ['Manager', 'Market Value', 'LGS Asset Class']], names=['horizon', 'type'])
columns_performance_performance_multilevel = pd.MultiIndex.from_product(
    [['1 Month', '3 Month', 'FYTD', '1 Year', '2 Year', '3 Year', '5 Year', '7Year'], ['LGS', 'Active']],
    names=['horizon', 'type']
)

# Creates performance tables
df_jpm_table_performance_lead = df_jpm_table[columns_lead + ['LGS Asset Class']]
df_jpm_table_performance_lead.columns = columns_performance_lead_multilevel
df_jpm_table_performance_performance = df_jpm_table[columns_performance]
df_jpm_table_performance_performance.columns = columns_performance_performance_multilevel
df_jpm_table_performance = pd.concat([df_jpm_table_performance_lead, df_jpm_table_performance_performance], axis=1)
del df_jpm_table_performance_lead
del df_jpm_table_performance_performance

asset_class_to_performance_dict = dict(list(df_jpm_table_performance.groupby([('', 'LGS Asset Class')])))
for asset_class, df_temp in asset_class_to_performance_dict.items():
    df_temp = df_temp.drop(("", 'LGS Asset Class'), axis=1)
    df_temp.to_csv('U:/CIO/#Investment_Report/Data/output/testing/returns/' + str(asset_class) + '.csv', index=False)

    with open('U:/CIO/#Investment_Report/Data/output/testing/returns/' + str(asset_class) + '.tex', 'w') as tf:
        latex_string_temp = (
                df_temp
                .to_latex(index=False, na_rep='', multicolumn_format='c')
                .replace('-0.00', '0.00')
        )
        tf.write(latex_string_temp)

# Creates risk table
df_jpm_table_risk = df_jpm_table[columns_lead + columns_risk + ['LGS Asset Class']]

asset_class_to_risk_dict = dict(list(df_jpm_table_risk.groupby(['LGS Asset Class'])))
for asset_class, df_temp in asset_class_to_risk_dict.items():
    df_temp = df_temp.drop('LGS Asset Class', axis=1)
    df_temp.to_csv('U:/CIO/#Investment_Report/Data/output/testing/risk/' + str(asset_class) + '.csv', index=False)

    with open('U:/CIO/#Investment_Report/Data/output/testing/risk/' + str(asset_class) + '.tex', 'w') as tf:
        latex_string_temp = (
                df_temp
                .to_latex(index=False, na_rep='', multicolumn_format='c')
                .replace('-0.00', '0.00')
        )
        tf.write(latex_string_temp)


# Creates active contribution table
df_jpm_table_active_contribution = df_jpm_table[columns_lead[:1] + columns_active_contribution]
df_jpm_table_active_contribution = df_jpm_table_active_contribution.sort_values(['12_Active_Contribution'], ascending=False).reset_index(drop=True)
df_jpm_table_active_contribution_missing = df_jpm_table_active_contribution[df_jpm_table_active_contribution['12_Active_Contribution'].isin([np.nan])]
df_jpm_table_active_contribution = df_jpm_table_active_contribution[~df_jpm_table_active_contribution['12_Active_Contribution'].isin([np.nan])]

df_jpm_table_active_contribution_top = df_jpm_table_active_contribution[:10].reset_index(drop=True)
df_jpm_table_active_contribution_top = df_jpm_table_active_contribution_top.rename(columns={'12_Active_Contribution': 'Contribution'})
df_jpm_table_active_contribution_bottom = df_jpm_table_active_contribution[-10:].reset_index(drop=True)
df_jpm_table_active_contribution_bottom = df_jpm_table_active_contribution_bottom.rename(columns={'12_Active_Contribution': 'Detraction'})

df_jpm_table_active_contribution_combined = pd.concat([df_jpm_table_active_contribution_top, df_jpm_table_active_contribution_bottom], axis=1, sort=False)
df_jpm_table_active_contribution_combined.to_csv('U:/CIO/#Investment_Report/Data/output/testing/contributors/contributors.csv', index=False)
with open('U:/CIO/#Investment_Report/Data/output/testing/contributors/contributors.tex', 'w') as tf:
    tf.write(df_jpm_table_active_contribution_combined.to_latex(index=False, na_rep=''))




"""
# Creates charts
df_jpm_chart_12_excess = df_jpm_main[['Manager', 'Date', '12_Excess', 'LGS Asset Class']]
df_jpm_chart_12_excess = df_jpm_chart_12_excess.pivot(index='Date', columns='Manager', values='12_Excess')[-60:]

df_jpm_chart_60_excess = df_jpm_main[['Manager', 'Date', '60_Excess']]
df_jpm_chart_60_excess = df_jpm_chart_60_excess.pivot(index='Date', columns='Manager', values='60_Excess')[-60:]

df_jpm_chart_market_value = df_jpm_main[['Manager', 'Date', 'Market Value']]
df_jpm_chart_market_value = df_jpm_chart_market_value.pivot(index='Date', columns='Manager', values='Market Value')[-60:]
"""
