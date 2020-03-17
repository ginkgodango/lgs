import pandas as pd
import numpy as np
import datetime as dt

# START USER INPUT DATA
jpm_filepath = 'U:/CIO/#Data/input/jpm/performance/2020/02/Historical Time Series - Monthly - Strategy Market Values Returns and Benchmarks_GOF.xlsx'
lgs_dictionary_filepath = 'U:/CIO/#Data/input/lgs/dictionary/2020/02/New Dictionary_v6.xlsx'
FYTD = 8
report_date = dt.datetime(2020, 2, 29)
# END USER INPUT DATA

# Imports the JPM time-series.
use_managerid = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 14, 15]
use_accountid = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15]
footnote_rows = 28

df_jpm = pd.read_excel(
        pd.ExcelFile(jpm_filepath),
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
df_jpm = pd.melt(df_jpm, id_vars=['Manager'], value_name='Values')
df_jpm = df_jpm.sort_values(['Manager', 'Date'])
df_jpm = df_jpm.reset_index(drop=True)
df_jpm = df_jpm.replace('-', np.NaN)

# Filters nan values
df_jpm = df_jpm[~df_jpm['Values'].isin([np.nan])].reset_index(drop=True)

# Splits df_jpm into a returns and benchmarks
df_jpm_market_values = df_jpm[~df_jpm.Manager.str.endswith(('.1', '.2'))].reset_index(drop=True)
df_jpm_returns = df_jpm[df_jpm.Manager.str.endswith('.1')].reset_index(drop=True)
df_jpm_benchmarks = df_jpm[df_jpm.Manager.str.endswith('.2')].reset_index(drop=True)

df_jpm_returns['Manager'] = [df_jpm_benchmarks['Manager'][i][:-2] for i in range(0, len(df_jpm_benchmarks))]
df_jpm_benchmarks['Manager'] = [df_jpm_benchmarks['Manager'][i][:-2] for i in range(0, len(df_jpm_benchmarks))]

df_jpm_market_values = df_jpm_market_values.rename(columns={'Values': 'JPM_Market_Value'})
df_jpm_returns = df_jpm_returns.rename(columns={'Values': 'JPM_Return'})
df_jpm_benchmarks = df_jpm_benchmarks.rename(columns={'Values': 'JPM_Benchmark'})

# Converts the returns to percentage.
df_jpm_returns['JPM_Return'] = df_jpm_returns['JPM_Return']/100
df_jpm_benchmarks['JPM_Benchmark'] = df_jpm_benchmarks['JPM_Benchmark']/100

# Merges returns and benchmarks
df_jpm = pd.merge(
        left=df_jpm_market_values,
        right=df_jpm_returns,
        left_on=['Manager', 'Date'],
        right_on=['Manager', 'Date'],
        how='inner'
)

df_jpm = pd.merge(
        left=df_jpm,
        right=df_jpm_benchmarks,
        left_on=['Manager', 'Date'],
        right_on=['Manager', 'Date'],
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

    for column in ['Return', 'Benchmark']:

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


# Calculates tracking error
df_jpm['60_Tracking_Error'] = (
    df_jpm
    .groupby(['Manager'])['1_Excess']
    .rolling(60)
    .apply(lambda r: np.std(r, ddof=1)*np.sqrt(12), raw=False)
    .reset_index(drop=False)['1_Excess']
)

# Selects only strategy aggregates
df_jpm = df_jpm[df_jpm['LGS Strategy Aggregate'].isin([1])].reset_index(drop=True)

# Selects only reporting date entries
df_jpm_table = df_jpm[df_jpm['Date'] == report_date].reset_index(drop=True)
df_jpm_table = df_jpm_table.sort_values(['LGS Strategy Order']).reset_index(drop=True)

df_jpm_table = df_jpm_table[[
    'LGS Name',
    'JPM_Market_Value',
    '1_Return',
    '1_Excess',
    '3_Return',
    '3_Excess',
    'FYTD_Return',
    'FYTD_Excess',
    '12_Return',
    '12_Excess',
    '36_Return',
    '36_Excess',
    '60_Return',
    '60_Excess',
    '84_Return',
    '84_Excess',
    '60_Tracking_Error'
]]

df_jpm_table['JPM_Market_Value'] = (df_jpm_table['JPM_Market_Value']/1000000).round(2)

decimal_to_percentage_list = [
    '1_Return',
    '1_Excess',
    '3_Return',
    '3_Excess',
    'FYTD_Return',
    'FYTD_Excess',
    '12_Return',
    '12_Excess',
    '36_Return',
    '36_Excess',
    '60_Return',
    '60_Excess',
    '84_Return',
    '84_Excess',
    '60_Tracking_Error'
]
df_jpm_table[decimal_to_percentage_list] = (df_jpm_table[decimal_to_percentage_list]*100).round(2)

df_jpm_table = df_jpm_table.rename(columns={
    'LGS Name': 'Strategy',
    'JPM_Market_Value': 'Market Value',
    '60_Tracking_Error': 'Tracking Error'
})

columns_multilevel1 = pd.MultiIndex.from_product([[''], ['Strategy']])
columns_multilevel2 = pd.MultiIndex.from_product([['Market Value'], ['($Mills)']])
columns_multilevel3 = pd.MultiIndex.from_product([['1 Month', '3 Month', 'FYTD', '1 Year', '3 Year', '5 Year', '7 Year'], ['LGS', 'Active']],)
columns_multilevel4 = pd.MultiIndex.from_product([['Tracking'], ['Error']])

# Creates performance tables
df_jpm_table1 = df_jpm_table[['Strategy']]
df_jpm_table2 = df_jpm_table[['Market Value']]
df_jpm_table3 = df_jpm_table[['1_Return',
    '1_Excess',
    '3_Return',
    '3_Excess',
    'FYTD_Return',
    'FYTD_Excess',
    '12_Return',
    '12_Excess',
    '36_Return',
    '36_Excess',
    '60_Return',
    '60_Excess',
    '84_Return',
    '84_Excess']]
df_jpm_table4 = df_jpm_table[['Tracking Error']]
df_jpm_table1.columns = columns_multilevel1
df_jpm_table2.columns = columns_multilevel2
df_jpm_table3.columns = columns_multilevel3
df_jpm_table4.columns = columns_multilevel4

df_jpm_table = pd.concat(
    [
        df_jpm_table1,
        df_jpm_table2,
        df_jpm_table3,
        df_jpm_table4,
        ],
    axis=1
)

with open('U:/CIO/#Data/output/investment/strategy/strategy.tex', 'w') as tf:
    latex_string_temp = df_jpm_table.to_latex(index=False, na_rep='', multicolumn_format='c', column_format='lRRRRRRRRRRRRRRRRRR').replace('-0.00', '0.00')
    tf.write(latex_string_temp)

df_jpm.to_csv('U:/CIO/#Data/output/investment/checker/lgs_strategy.csv', index=False)
