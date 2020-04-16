import os
import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

# START USER INPUT DATA
jpm_main_filepath = 'U:/CIO/#Data/input/jpm/performance/2020/02/Historical Time Series - Monthly - Main Returns and Benchmarks.xlsx'
jpm_alts_filepath = 'U:/CIO/#Data/input/jpm/performance/2020/02/Historical Time Series - Monthly - Alternatives Returns and Benchmarks.xlsx'
jpm_mv_filepath = 'U:/CIO/#Data/input/jpm/performance/2020/02/Historical Time Series - Monthly - Main Market Values.xlsx'
jpm_mv_alts_filepath = 'U:/CIO/#Data/input/jpm/performance/2020/02/Historical Time Series - Monthly - Alternatives Market Values.xlsx'
jpm_strategies_filepath = 'U:/CIO/#Data/input/jpm/performance/2020/02/Historical Time Series - Monthly - Strategy Market Values Returns and Benchmarks_GOF.xlsx'
lgs_dictionary_filepath = 'U:/CIO/#Data/input/lgs/dictionary/2020/02/New Dictionary_v7.xlsx'
lgs_allocations_filepath ='U:/CIO/#Data/input/lgs/allocations/asset_allocations_2020-01-31.csv'

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
df_jpm_main = pd.melt(df_jpm_main, id_vars=['Manager'], value_name='Values')
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
df_jpm_alts = pd.melt(df_jpm_alts, id_vars=['Manager'], value_name='Values')
df_jpm_alts = df_jpm_alts.sort_values(['Manager', 'Date'])
df_jpm_alts = df_jpm_alts.reset_index(drop=True)
df_jpm_alts = df_jpm_alts.replace('-', np.NaN)

df_jpm = pd.concat([df_jpm_alts, df_jpm_main], axis=0).reset_index(drop=True)

df_jpm_returns = df_jpm[~df_jpm.Manager.str.endswith('.1')].reset_index(drop=True)
df_jpm_benchmarks = df_jpm[df_jpm.Manager.str.endswith('.1')].reset_index(drop=True)

df_jpm_returns['Manager'] = [df_jpm_returns['Manager'][i] for i in range(0, len(df_jpm_returns))]
df_jpm_benchmarks['Manager'] = [df_jpm_benchmarks['Manager'][i][:-2] for i in range(0, len(df_jpm_benchmarks))]

df_jpm_returns = df_jpm_returns.rename(columns={'Values': 'JPM Return'})
df_jpm_benchmarks = df_jpm_benchmarks.rename(columns={'Values': 'JPM Benchmark'})

df_jpm = pd.merge(
        left=df_jpm_returns,
        right=df_jpm_benchmarks,
        left_on=['Manager', 'Date'],
        right_on=['Manager', 'Date'],
        how='inner'
)


# Imports the JPM time-series of market values.
df_jpm_main_mv = pd.read_excel(
        pd.ExcelFile(jpm_mv_filepath),
        sheet_name='Sheet1',
        skiprows=use_accountid,
        skipfooter=footnote_rows,
        header=1
        )

# Reshapes the time-series into a panel.
df_jpm_main_mv = df_jpm_main_mv.rename(columns={'Unnamed: 0': 'Date'})
df_jpm_main_mv = df_jpm_main_mv.set_index('Date')
df_jpm_main_mv = df_jpm_main_mv.transpose()
df_jpm_main_mv = df_jpm_main_mv.reset_index(drop=False)
df_jpm_main_mv = df_jpm_main_mv.rename(columns={'index': 'Manager'})
df_jpm_main_mv = pd.melt(df_jpm_main_mv, id_vars=['Manager'], value_name='JPM Market Value')
df_jpm_main_mv = df_jpm_main_mv.sort_values(['Manager', 'Date'])
df_jpm_main_mv = df_jpm_main_mv.reset_index(drop=True)
df_jpm_main_mv = df_jpm_main_mv.replace('-', np.NaN)

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
df_jpm_mv_alts = pd.melt(df_jpm_mv_alts, id_vars=['Manager'], value_name='JPM Market Value')
df_jpm_mv_alts = df_jpm_mv_alts.sort_values(['Manager', 'Date'])
df_jpm_mv_alts = df_jpm_mv_alts.reset_index(drop=True)
df_jpm_mv_alts = df_jpm_mv_alts.replace('-', np.NaN)

df_jpm_mv = pd.concat([df_jpm_main_mv, df_jpm_mv_alts]).reset_index(drop=True)

df_jpm_combined = pd\
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
df_jpm_combined = df_jpm_combined.sort_values(['Manager', 'Date'], ascending=[True, True]).reset_index(drop=True)

df_lgs = pd.read_excel(
        pd.ExcelFile(lgs_dictionary_filepath),
        sheet_name='Sheet1',
        header=0
)
df_lgs = df_lgs.rename(columns={'JPM Account Id': 'Manager'})

df_jpm_combined = pd.merge(
        left=df_jpm_combined,
        right=df_lgs,
        left_on=['Manager'],
        right_on=['Manager'],
        how='inner'
)

# Keep only open accounts
df_jpm_combined = df_jpm_combined[df_jpm_combined['LGS Open'] == 1].reset_index(drop=True)
df_jpm_combined = df_jpm_combined.drop(columns=['LGS Open'], axis=1)

# Keep only reported items
df_jpm_not_reported = df_jpm_combined[df_jpm_combined['JPM ReportName'].isin([np.nan])]
df_jpm_combined = df_jpm_combined[~df_jpm_combined['JPM ReportName'].isin([np.nan])].reset_index(drop=True)

# Sort values VERY IMPORTANT for groupby merging
df_jpm_combined = df_jpm_combined.sort_values(['Manager', 'Date'], ascending=[True, True]).reset_index(drop=True)

# Converts everything into decimals
df_jpm_combined['JPM Return'] = df_jpm_combined['JPM Return'] / 100
df_jpm_combined['JPM Benchmark'] = df_jpm_combined['JPM Benchmark'] / 100

# Selects sectors only
df_jpm_sectors = df_jpm_combined[df_jpm_combined['LGS Sector Aggregate'].isin([1])].reset_index(drop=True)

#df_jpm_sectors_unhedged = df_jpm_combined[df_jpm_combined['LGS Sector Aggregate Unhedged'].isin([1])].reset_index(drop=True)

# Selects managers only
df_jpm_managers = df_jpm_combined[df_jpm_combined['LGS Sector Aggregate'].isin([0])].reset_index(drop=True)

# Imports strategy returns and market values
df_jpm_strategies = pd.read_excel(
        pd.ExcelFile(jpm_strategies_filepath),
        sheet_name='Sheet1',
        skiprows=use_managerid,
        skipfooter=footnote_rows,
        header=1
)

# Reshapes the time-series into a panel.
df_jpm_strategies = df_jpm_strategies.rename(columns={'Unnamed: 0': 'Date'})
df_jpm_strategies = df_jpm_strategies.set_index('Date')
df_jpm_strategies = df_jpm_strategies.transpose()
df_jpm_strategies = df_jpm_strategies.reset_index(drop=False)
df_jpm_strategies = df_jpm_strategies.rename(columns={'index': 'Manager'})
df_jpm_strategies = pd.melt(df_jpm_strategies, id_vars=['Manager'], value_name='Values')
df_jpm_strategies = df_jpm_strategies.sort_values(['Manager', 'Date'])
df_jpm_strategies = df_jpm_strategies.reset_index(drop=True)
df_jpm_strategies = df_jpm_strategies.replace('-', np.NaN)

# Filters nan values
df_jpm_strategies = df_jpm_strategies[~df_jpm_strategies['Values'].isin([np.nan])].reset_index(drop=True)

# Splits df_jpm into a returns and benchmarks
df_jpm_strategies_mv = df_jpm_strategies[~df_jpm_strategies.Manager.str.endswith(('.1', '.2'))].reset_index(drop=True)
df_jpm_strategies_returns = df_jpm_strategies[df_jpm_strategies.Manager.str.endswith('.1')].reset_index(drop=True)
df_jpm_strategies_benchmarks = df_jpm_strategies[df_jpm_strategies.Manager.str.endswith('.2')].reset_index(drop=True)

df_jpm_strategies_returns['Manager'] = [df_jpm_strategies_returns['Manager'][i][:-2] for i in range(0, len(df_jpm_strategies_returns))]
df_jpm_strategies_benchmarks['Manager'] = [df_jpm_strategies_benchmarks['Manager'][i][:-2] for i in range(0, len(df_jpm_strategies_benchmarks))]

df_jpm_strategies_mv = df_jpm_strategies_mv.rename(columns={'Values': 'JPM Strategy Market Value'})
df_jpm_strategies_returns = df_jpm_strategies_returns.rename(columns={'Values': 'JPM Strategy Return'})
df_jpm_strategies_benchmarks = df_jpm_strategies_benchmarks.rename(columns={'Values': 'JPM Strategy Benchmark'})

# Converts the returns to percentage.
df_jpm_strategies_returns['JPM Strategy Return'] = df_jpm_strategies_returns['JPM Strategy Return']/100
df_jpm_strategies_benchmarks['JPM Strategy Benchmark'] = df_jpm_strategies_benchmarks['JPM Strategy Benchmark']/100

# Merges market values, returns and benchmarks
df_jpm_strategies = pd.merge(
        left=df_jpm_strategies_mv,
        right=df_jpm_strategies_returns,
        left_on=['Manager', 'Date'],
        right_on=['Manager', 'Date'],
        how='inner'
)

df_jpm_strategies = pd.merge(
        left=df_jpm_strategies,
        right=df_jpm_strategies_benchmarks,
        left_on=['Manager', 'Date'],
        right_on=['Manager', 'Date'],
        how='inner'
)

df_jpm_strategies = df_jpm_strategies.rename(columns={'Manager': 'Strategy'})

# df_jpm_strategies.drop('JPM Market Value', axis=1)

# Reads in allocation data
df_lgs_allocations = pd.read_csv(
        lgs_allocations_filepath,
        parse_dates=['Date']
        )

df_lgs_allocations['Date'] = [
        df_lgs_allocations['Date'][i] + relativedelta(months=1, day=31)
        for i in range(0, len(df_lgs_allocations))
        ]

df_lgs_allocations['Portfolio Weight'] = df_lgs_allocations['Portfolio Weight'] / 100
df_lgs_allocations['Dynamic Weight'] = df_lgs_allocations['Dynamic Weight'] / 100
df_lgs_allocations['Benchmark Weight'] = df_lgs_allocations['Benchmark Weight'] / 100

df_jpm_combined = pd.merge(
        left=df_lgs_allocations,
        right=df_jpm_combined,
        left_on=['Date', 'Asset Class'],
        right_on=['Date', 'JPM ReportStrategyName']
)

df_jpm_combined = pd.merge(
        left=df_jpm_combined,
        right=df_jpm_strategies,
        left_on=['Date', 'Strategy'],
        right_on=['Date', 'Strategy']
)

df_jpm_combined = df_jpm_combined.sort_values(['Date', 'Strategy', 'LGS Asset Class Order'])



# Performs calculations
df_jpm_combined['TAA'] = (df_jpm_combined['Portfolio Weight'] - df_jpm_combined['Dynamic Weight']) * df_jpm_combined['JPM Benchmark']

df_jpm_combined['DAA'] = (df_jpm_combined['Dynamic Weight'] - df_jpm_combined['Benchmark Weight']) * df_jpm_combined['JPM Benchmark']

df_jpm_combined['SAA'] = df_jpm_combined['TAA'] + df_jpm_combined['DAA']

df_jpm_combined['SS'] = (df_jpm_combined['Benchmark Weight']) * (df_jpm_combined['JPM Return'] - df_jpm_combined['JPM Benchmark'])

df_jpm_combined['Asset Class Total'] = df_jpm_combined['SAA'] + df_jpm_combined['SS']

df_jpm_combined_asset_class_total = df_jpm_combined.groupby(['Date', 'Strategy'])['Asset Class Total'].sum().reset_index(drop=False)

df_jpm_combined_asset_class_total = df_jpm_combined_asset_class_total.rename(columns={'Asset Class Total': 'Strategy Total'})

df_jpm_combined = pd.merge(
        left=df_jpm_combined,
        right=df_jpm_combined_asset_class_total,
        left_on=['Date', 'Strategy'],
        right_on=['Date', 'Strategy'],
        how='inner'
)

"""
# Sort values VERY IMPORTANT for groupby merging
df_jpm_combined = df_jpm_combined.sort_values(['Strategy', 'Asset Class', 'Date'], ascending=[True, True, True]).reset_index(drop=True)

# Sets the dictionary for the holding period returns.
horizon_to_period_dict = {
        '1_': 1,
        '3_': 3,
        'FYTD_': FYTD,
        '12_': 12,
}

# Calculates the holding period returns and annualises for periods greater than 12 months.
for horizon, period in horizon_to_period_dict.items():

    for column in ['JPM Strategy Return', 'JPM Strategy Benchmark', 'TAA', 'DAA', 'SAA', 'SS', 'Asset Class Total', 'Strategy Total']:

        column_name = horizon + column
        return_type = column

        if period <= 12:
            df_jpm_combined[column_name] = (
                df_jpm_combined
                .groupby(['Strategy', 'Asset Class'])[return_type]
                .rolling(period)
                .apply(lambda r: np.prod(1+r)-1, raw=False)
                .reset_index(drop=False)[return_type]
            )

        elif period > 12:
            df_jpm_combined[column_name] = (
                df_jpm_combined
                .groupby(['Strategy', 'Asset Class'])[return_type]
                .rolling(period)
                .apply(lambda r: (np.prod(1+r)**(12/period))-1, raw=False)
                .reset_index(drop=False)[return_type]
            )

    df_jpm_combined_asset_class_total_horizon = df_jpm_combined.groupby(['Date', 'Strategy'])[horizon + 'Asset Class Total'].sum().reset_index(drop=False)

    df_jpm_combined_asset_class_total_horizon = df_jpm_combined_asset_class_total_horizon.rename(columns={horizon + 'Asset Class Total': horizon + 'Strategy Total by Asset Class Sum'})

    df_jpm_combined = pd.merge(
            left=df_jpm_combined,
            right=df_jpm_combined_asset_class_total_horizon,
            left_on=['Date', 'Strategy'],
            right_on=['Date', 'Strategy'],
            how='inner'
    )

    df_jpm_combined[horizon + 'Strategy Excess'] = df_jpm_combined[horizon + 'JPM Strategy Return'] - df_jpm_combined[horizon + 'JPM Strategy Benchmark']

    df_jpm_combined[horizon + 'Residual_SAA'] = df_jpm_combined[horizon + 'SAA'] - (df_jpm_combined[horizon + 'TAA'] + df_jpm_combined[horizon + 'DAA'])

    df_jpm_combined[horizon + 'Residual_Asset_Class_Total'] = df_jpm_combined[horizon + 'Asset Class Total'] - (df_jpm_combined[horizon + 'SAA'] + df_jpm_combined[horizon + 'SS'])

    df_jpm_combined[horizon + 'Residual_Strategy'] = df_jpm_combined[horizon + 'Strategy Total'] - df_jpm_combined[horizon + 'Strategy Total by Asset Class Sum']

    df_jpm_combined[horizon + 'Residual_Actual'] = df_jpm_combined[horizon + 'Strategy Excess'] - df_jpm_combined[horizon + 'Strategy Total']

    df_jpm_combined = df_jpm_combined.sort_values(['Strategy', 'Asset Class', 'Date'], ascending=[True, True, True]).reset_index(drop=True)


df_jpm_combined = df_jpm_combined.sort_values(['Date', 'Strategy', 'LGS Asset Class Order'], ascending=[True, True, True]).reset_index(drop=True)

df_jpm_combined.to_csv('C:/Users/merri/Dropbox/Work/LGS/CIO/#Data/output/prototype_attribution_v2_python.csv', index=False)
"""

# Calculates FX, Pure Active, and Style at the manager level
df_jpm_managers_asset_class_mv_total = df_jpm_managers.groupby(['Date', 'LGS Attribution Asset Class']).sum().reset_index(drop=False)
df_jpm_managers_asset_class_mv_total = df_jpm_managers_asset_class_mv_total[['Date', 'LGS Attribution Asset Class', 'JPM Market Value']]
df_jpm_managers_asset_class_mv_total = df_jpm_managers_asset_class_mv_total.rename(columns={'JPM Market Value': 'LGS Asset Class Market Value'})

df_jpm_managers_combined = pd.merge(
    left=df_jpm_managers,
    right=df_jpm_managers_asset_class_mv_total,
    left_on=['Date', 'LGS Attribution Asset Class'],
    right_on=['Date', 'LGS Attribution Asset Class'],
    how='inner'
)


df_jpm_sectors_returns_benchmarks = df_jpm_sectors[['Date', 'LGS Attribution Asset Class', 'JPM Return', 'JPM Benchmark']]
df_jpm_sectors_returns_benchmarks = df_jpm_sectors_returns_benchmarks.rename(columns={'JPM Return': 'JPM Asset Class Return', 'JPM Benchmark': 'JPM Asset Class Benchmark'})

df_jpm_managers_combined = pd.merge(
    left=df_jpm_managers_combined,
    right=df_jpm_sectors_returns_benchmarks,
    left_on=['Date', 'LGS Attribution Asset Class'],
    right_on=['Date', 'LGS Attribution Asset Class'],
    how='inner'
)

df_jpm_managers_combined = df_jpm_managers_combined.sort_values(['Date', 'LGS Asset Class Order', 'LGS Manager Order']).reset_index(drop=True)
df_jpm_managers_combined['LGS Weight Manager in Asset Class'] = df_jpm_managers_combined['JPM Market Value'] / df_jpm_managers_combined['LGS Asset Class Market Value']

# Calculates Style, Pure Active
df_jpm_managers_combined['Pure Active Excess Return'] = df_jpm_managers_combined['JPM Return'] - df_jpm_managers_combined['JPM Benchmark']
df_jpm_managers_combined['Style Excess Return'] = df_jpm_managers_combined['JPM Benchmark'] - df_jpm_managers_combined['JPM Asset Class Benchmark']

# Calculates weighted style
df_jpm_managers_combined['Weighted Asset Class Return'] = df_jpm_managers_combined['LGS Weight Manager in Asset Class'] * df_jpm_managers_combined['JPM Return']
df_jpm_managers_combined['Weighted Asset Class Style Benchmark'] = df_jpm_managers_combined['LGS Weight Manager in Asset Class'] * df_jpm_managers_combined['JPM Benchmark']
df_jpm_managers_combined['Weighted Asset Class Benchmark'] = df_jpm_managers_combined['LGS Weight Manager in Asset Class'] * df_jpm_managers_combined['JPM Asset Class Benchmark']
df_jpm_managers_combined['Weighted Pure Active Excess Return'] = df_jpm_managers_combined['LGS Weight Manager in Asset Class'] * df_jpm_managers_combined['Pure Active Excess Return']
df_jpm_managers_combined['Weighted Style Excess Return'] = df_jpm_managers_combined['LGS Weight Manager in Asset Class'] * df_jpm_managers_combined['Style Excess Return']
df_jpm_managers_combined['Weighted Asset Class Excess Return'] = df_jpm_managers_combined['LGS Weight Manager in Asset Class'] * (df_jpm_managers_combined['JPM Return'] - df_jpm_managers_combined['JPM Asset Class Benchmark'])

df_jpm_managers_style = df_jpm_managers_combined.groupby(['Date', 'LGS Attribution Asset Class', 'LGS Benchmark']).sum()

df_jpm_managers_style_asset_class = df_jpm_managers_combined.groupby(['Date', 'LGS Attribution Asset Class']).sum().reset_index(drop=False)
df_jpm_managers_style_asset_class = df_jpm_managers_style_asset_class.rename(columns={'JPM Market Value': 'Weighted Asset Class Sum Market Value'})
df_jpm_managers_style_asset_class = df_jpm_managers_style_asset_class[
    [
        'Date',
        'LGS Attribution Asset Class',
        'Weighted Asset Class Sum Market Value',
        'Weighted Asset Class Return',
        'Weighted Asset Class Style Benchmark',
        'Weighted Asset Class Benchmark',
        'Weighted Pure Active Excess Return',
        'Weighted Style Excess Return',
        'Weighted Asset Class Excess Return'
    ]
]

# FX Section
df_jpm_managers_fx = df_jpm_managers[df_jpm_managers['LGS Name'].isin(['FX Overlay'])].reset_index(drop=True)
df_jpm_managers_fx = df_jpm_managers_fx.rename(
    columns={
        'JPM Market Value': 'FX Market Value',
        'JPM Return': 'FX Return',
        'JPM Benchmark': 'FX Benchmark',
    }
)
df_jpm_managers_fx['FX Excess Return'] = df_jpm_managers_fx['FX Return'] - df_jpm_managers_fx['FX Benchmark']
df_jpm_managers_fx['LGS Attribution Asset Class'] = 'IE'
df_jpm_managers_fx = df_jpm_managers_fx[['Date', 'FX Market Value', 'FX Return', 'FX Benchmark', 'FX Excess Return', 'LGS Attribution Asset Class']]

df_jpm_managers_style_asset_class_fx = pd.merge(
    left=df_jpm_managers_style_asset_class,
    right=df_jpm_managers_fx,
    left_on=['Date', 'LGS Attribution Asset Class'],
    right_on=['Date', 'LGS Attribution Asset Class'],
    how='outer'
)

df_jpm_managers_style_asset_class_fx.fillna({'FX Market Value': 0, 'FX Return': 0, 'FX Benchmark': 0, 'FX Excess Return': 0}, inplace=True)

df_jpm_all_combined = pd.merge(
    left=df_jpm_combined,
    right=df_jpm_managers_style_asset_class_fx,
    left_on=['Date', 'LGS Attribution Asset Class'],
    right_on=['Date', 'LGS Attribution Asset Class']
)

df_jpm_all_combined = df_jpm_all_combined.sort_values(['Date', 'Strategy', 'LGS Asset Class Order'], ascending=[True, True, True]).reset_index(drop=True)




# Fix Bonds
# FIX IE Benchmark

# Overwrite PE, OA, DA
overwrite_weighted_asset_class_return = []
overwrite_weighted_asset_class_benchmark = []
for i in range(0, len(df_jpm_all_combined)):
    if df_jpm_all_combined['LGS Attribution Asset Class'][i] in ['PE', 'OA', 'DA']:
        overwrite_weighted_asset_class_return.append(df_jpm_all_combined['JPM Return'][i])
        overwrite_weighted_asset_class_benchmark.append(df_jpm_all_combined['JPM Benchmark'][i])
    else:
        overwrite_weighted_asset_class_return.append(df_jpm_all_combined['Weighted Asset Class Return'][i])
        overwrite_weighted_asset_class_benchmark.append(df_jpm_all_combined['Weighted Asset Class Benchmark'][i])
df_jpm_all_combined['Weighted Asset Class Return'] = overwrite_weighted_asset_class_return
df_jpm_all_combined['Weighted Asset Class Benchmark'] = overwrite_weighted_asset_class_benchmark


# Check sums
df_jpm_all_combined['Check Sum Asset Class Market Value Difference (Millions)'] = (df_jpm_all_combined['JPM Market Value'] - df_jpm_all_combined['Weighted Asset Class Sum Market Value'] - df_jpm_all_combined['FX Market Value'])/1000000
df_jpm_all_combined['Check Sum Asset Class Return Difference'] = df_jpm_all_combined['JPM Return'] - df_jpm_all_combined['Weighted Asset Class Return'] - df_jpm_all_combined['FX Return']
df_jpm_all_combined['Check Sum Asset Class Benchmark Difference'] = df_jpm_all_combined['JPM Benchmark'] - df_jpm_all_combined['Weighted Asset Class Benchmark']

