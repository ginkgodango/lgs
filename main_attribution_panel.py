import os
import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

# START USER INPUT DATA
jpm_main_returns_filepath = 'U:/CIO/#Data/input/jpm/performance/2020/03/Historical Time Series - Monthly - Main Returns_v2.xlsx'
jpm_alts_returns_filepath = 'U:/CIO/#Data/input/jpm/performance/2020/03/Historical Time Series - Monthly - Alternatives Returns_v2.xlsx'

jpm_main_benchmarks_filepath = 'U:/CIO/#Data/input/jpm/performance/2020/03/Historical Time Series - Monthly - Main Benchmarks_v2.xlsx'
jpm_alts_benchmarks_filepath = 'U:/CIO/#Data/input/jpm/performance/2020/03/Historical Time Series - Monthly - Alternatives Benchmarks_v2.xlsx'

jpm_main_mv_filepath = 'U:/CIO/#Data/input/jpm/performance/2020/03/Historical Time Series - Monthly - Main Market Values_v2.xlsx'
jpm_alts_mv_filepath = 'U:/CIO/#Data/input/jpm/performance/2020/03/Historical Time Series - Monthly - Alternatives Market Values_v2.xlsx'

jpm_strategy_returns_benchmarks_mv_filepath = 'U:/CIO/#Data/input/jpm/performance/2020/03/Historical Time Series - Monthly - Strategy Market Values Returns and Benchmarks.xlsx'

lgs_returns_benchmarks_filepath = 'U:/CIO/#Investment_Report/data/input/returns/returns_2020-03-31_attribution.csv'
lgs_dictionary_filepath = 'U:/CIO/#Data/input/lgs/dictionary/2020/03/New Dictionary_v8.xlsx'
lgs_allocations_filepath ='U:/CIO/#Data/input/lgs/allocations/asset_allocations_2020-03-31.csv'

FYTD = 9
report_date = dt.datetime(2020, 3, 31)
# END USER INPUT DATA

use_managerid = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12]
use_accountid = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
footnote_rows = 28

# use_managerid = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 14, 15]
# use_accountid = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15]
# footnote_rows = 28

# Returns
df_jpm_main_returns = pd.read_excel(
        pd.ExcelFile(jpm_main_returns_filepath),
        sheet_name='Sheet1',
        skiprows=use_accountid,
        skipfooter=footnote_rows,
        header=1
        )

# Reshapes the time-series into a panel.
df_jpm_main_returns = df_jpm_main_returns.rename(columns={'Unnamed: 0': 'Date'})
df_jpm_main_returns = df_jpm_main_returns.set_index('Date')
df_jpm_main_returns = df_jpm_main_returns.transpose()
df_jpm_main_returns = df_jpm_main_returns.reset_index(drop=False)
df_jpm_main_returns = df_jpm_main_returns.rename(columns={'index': 'Manager'})
df_jpm_main_returns = pd.melt(df_jpm_main_returns, id_vars=['Manager'], value_name='Values')
df_jpm_main_returns = df_jpm_main_returns.sort_values(['Manager', 'Date'])
df_jpm_main_returns = df_jpm_main_returns.reset_index(drop=True)
df_jpm_main_returns = df_jpm_main_returns.replace('-', np.NaN)

df_jpm_alts_returns = pd.read_excel(
        pd.ExcelFile(jpm_alts_returns_filepath),
        sheet_name='Sheet1',
        skiprows=use_accountid,
        skipfooter=footnote_rows,
        header=1
        )

df_jpm_alts_returns = df_jpm_alts_returns.rename(columns={'Unnamed: 0': 'Date'})
df_jpm_alts_returns = df_jpm_alts_returns.set_index('Date')
df_jpm_alts_returns = df_jpm_alts_returns.transpose()
df_jpm_alts_returns = df_jpm_alts_returns.reset_index(drop=False)
df_jpm_alts_returns = df_jpm_alts_returns.rename(columns={'index': 'Manager'})
df_jpm_alts_returns = pd.melt(df_jpm_alts_returns, id_vars=['Manager'], value_name='Values')
df_jpm_alts_returns = df_jpm_alts_returns.sort_values(['Manager', 'Date'])
df_jpm_alts_returns = df_jpm_alts_returns.reset_index(drop=True)
df_jpm_alts_returns = df_jpm_alts_returns.replace('-', np.NaN)

df_jpm_returns = pd.concat([df_jpm_main_returns, df_jpm_alts_returns], axis=0).reset_index(drop=True)
df_jpm_returns = df_jpm_returns.rename(columns={'Values': 'JPM Return'})

# Benchmarks
df_jpm_main_benchmarks = pd.read_excel(
        pd.ExcelFile(jpm_main_benchmarks_filepath),
        sheet_name='Sheet1',
        skiprows=use_accountid,
        skipfooter=footnote_rows,
        header=1
        )

# Reshapes the time-series into a panel.
df_jpm_main_benchmarks = df_jpm_main_benchmarks.rename(columns={'Unnamed: 0': 'Date'})
df_jpm_main_benchmarks = df_jpm_main_benchmarks.set_index('Date')
df_jpm_main_benchmarks = df_jpm_main_benchmarks.transpose()
df_jpm_main_benchmarks = df_jpm_main_benchmarks.reset_index(drop=False)
df_jpm_main_benchmarks = df_jpm_main_benchmarks.rename(columns={'index': 'Manager'})
df_jpm_main_benchmarks = pd.melt(df_jpm_main_benchmarks, id_vars=['Manager'], value_name='Values')
df_jpm_main_benchmarks = df_jpm_main_benchmarks.sort_values(['Manager', 'Date'])
df_jpm_main_benchmarks = df_jpm_main_benchmarks.reset_index(drop=True)
df_jpm_main_benchmarks = df_jpm_main_benchmarks.replace('-', np.NaN)

df_jpm_alts_benchmarks = pd.read_excel(
        pd.ExcelFile(jpm_alts_benchmarks_filepath),
        sheet_name='Sheet1',
        skiprows=use_accountid,
        skipfooter=footnote_rows,
        header=1
        )

df_jpm_alts_benchmarks = df_jpm_alts_benchmarks.rename(columns={'Unnamed: 0': 'Date'})
df_jpm_alts_benchmarks = df_jpm_alts_benchmarks.set_index('Date')
df_jpm_alts_benchmarks = df_jpm_alts_benchmarks.transpose()
df_jpm_alts_benchmarks = df_jpm_alts_benchmarks.reset_index(drop=False)
df_jpm_alts_benchmarks = df_jpm_alts_benchmarks.rename(columns={'index': 'Manager'})
df_jpm_alts_benchmarks = pd.melt(df_jpm_alts_benchmarks, id_vars=['Manager'], value_name='Values')
df_jpm_alts_benchmarks = df_jpm_alts_benchmarks.sort_values(['Manager', 'Date'])
df_jpm_alts_benchmarks = df_jpm_alts_benchmarks.reset_index(drop=True)
df_jpm_alts_benchmarks = df_jpm_alts_benchmarks.replace('-', np.NaN)

df_jpm_benchmarks = pd.concat([df_jpm_main_benchmarks, df_jpm_alts_returns], axis=0).reset_index(drop=True)
df_jpm_benchmarks = df_jpm_benchmarks.rename(columns={'Values': 'JPM Benchmark'})

df_jpm_returns_benchmarks = pd.merge(
        left=df_jpm_returns,
        right=df_jpm_benchmarks,
        left_on=['Manager', 'Date'],
        right_on=['Manager', 'Date'],
        how='inner'
)


# Imports the JPM time-series of market values.
df_jpm_main_mv = pd.read_excel(
        pd.ExcelFile(jpm_main_mv_filepath),
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
        pd.ExcelFile(jpm_alts_mv_filepath),
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

df_jpm = pd\
    .merge(
        left=df_jpm_mv,
        right=df_jpm_returns_benchmarks,
        left_on=['Manager', 'Date'],
        right_on=['Manager', 'Date'],
        how='right'
    )\
    .sort_values(['Manager', 'Date'])\
    .reset_index(drop=True)

# Sort values VERY IMPORTANT for groupby merging
df_jpm = df_jpm.sort_values(['Manager', 'Date'], ascending=[True, True]).reset_index(drop=True)

df_lgs = pd.read_excel(
        pd.ExcelFile(lgs_dictionary_filepath),
        sheet_name='Sheet1',
        header=0
)
df_lgs = df_lgs.rename(columns={'JPM Account Id': 'Manager'})

df_combined = pd.merge(
        left=df_jpm,
        right=df_lgs,
        left_on=['Manager'],
        right_on=['Manager'],
        how='inner'
).reset_index(drop=True)

# Handles transitions
attunga_move_date = dt.datetime(2020, 3, 31)
lgs_asset_class_level_1_list = []
lgs_asset_class_level_2_list = []
for i in range(0, len(df_combined)):
    if (df_combined['LGS Name'][i] == 'Attunga') and (df_combined['Date'][i] < attunga_move_date):
        lgs_asset_class_level_1_list.append('AR')
        lgs_asset_class_level_2_list.append('AR')
    else:
        lgs_asset_class_level_1_list.append(df_combined['LGS Asset Class Level 1'][i])
        lgs_asset_class_level_2_list.append(df_combined['LGS Asset Class Level 2'][i])
df_combined['LGS Asset Class Level 1'] = lgs_asset_class_level_1_list
df_combined['LGS Asset Class Level 2'] = lgs_asset_class_level_2_list

# Keep only open accounts
# df_combined = df_combined[df_combined['LGS Open'] == 1].reset_index(drop=True)
# df_combined = df_combined.drop(columns=['LGS Open'], axis=1)

# Keep only reported items
df_not_reported = df_combined[df_combined['JPM ReportName'].isin([np.nan])]
df_combined = df_combined[~df_combined['JPM ReportName'].isin([np.nan])].reset_index(drop=True)

# Sort values VERY IMPORTANT for groupby merging
df_combined = df_combined.sort_values(['Manager', 'Date'], ascending=[True, True]).reset_index(drop=True)

# Converts everything into decimals
df_combined['JPM Return'] = df_combined['JPM Return'] / 100
df_combined['JPM Benchmark'] = df_combined['JPM Benchmark'] / 100

# Selects sectors only
df_sectors = df_combined[df_combined['LGS Sector Aggregate'].isin([1])].reset_index(drop=True)

#df_jpm_sectors_unhedged = df_jpm_combined[df_jpm_combined['LGS Sector Aggregate Unhedged'].isin([1])].reset_index(drop=True)

# Selects managers only
df_managers = df_combined[df_combined['LGS Sector Aggregate'].isin([0])].reset_index(drop=True)


# Imports strategy returns and market values
df_jpm_strategies = pd.read_excel(
        pd.ExcelFile(jpm_strategy_returns_benchmarks_mv_filepath),
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

df_combined = pd.merge(
        left=df_lgs_allocations,
        right=df_combined,
        left_on=['Date', 'Asset Class'],
        right_on=['Date', 'JPM ReportStrategyName']
)

df_combined = pd.merge(
        left=df_combined,
        right=df_jpm_strategies,
        left_on=['Date', 'Strategy'],
        right_on=['Date', 'Strategy']
)

df_combined = df_combined.sort_values(['Date', 'Strategy', 'LGS Asset Class Order'])



# Performs calculations
df_combined['TAA'] = (df_combined['Portfolio Weight'] - df_combined['Dynamic Weight']) * df_combined['JPM Benchmark']

df_combined['DAA'] = (df_combined['Dynamic Weight'] - df_combined['Benchmark Weight']) * df_combined['JPM Benchmark']

df_combined['SAA'] = df_combined['TAA'] + df_combined['DAA']

df_combined['SS'] = (df_combined['Benchmark Weight']) * (df_combined['JPM Return'] - df_combined['JPM Benchmark'])

df_combined['In'] = (df_combined['Portfolio Weight'] - df_combined['Benchmark Weight']) * (df_combined['JPM Return'] - df_combined['JPM Benchmark'])

df_combined['Asset Class Total'] = df_combined['SAA'] + df_combined['SS']

df_combined_asset_class_total = df_combined.groupby(['Date', 'Strategy'])['Asset Class Total'].sum().reset_index(drop=False)

df_combined_asset_class_total = df_combined_asset_class_total.rename(columns={'Asset Class Total': 'Strategy Total by Asset Class Sum'})

df_combined = pd.merge(
        left=df_combined,
        right=df_combined_asset_class_total,
        left_on=['Date', 'Strategy'],
        right_on=['Date', 'Strategy'],
        how='inner'
)


# Calculates FX, Pure Active, and Style at the manager level
df_managers_asset_class_mv_total = df_managers.groupby(['Date', 'LGS Asset Class Level 1']).sum().reset_index(drop=False)
df_managers_asset_class_mv_total = df_managers_asset_class_mv_total[['Date', 'LGS Asset Class Level 1', 'JPM Market Value']]
df_managers_asset_class_mv_total = df_managers_asset_class_mv_total.rename(columns={'JPM Market Value': 'LGS Asset Class Market Value'})

df_managers_combined = pd.merge(
    left=df_managers,
    right=df_managers_asset_class_mv_total,
    left_on=['Date', 'LGS Asset Class Level 1'],
    right_on=['Date', 'LGS Asset Class Level 1'],
    how='inner'
)


# Creates the sector returns benchmarks unhedged
df_lgs_returns = pd.read_csv(
    lgs_returns_benchmarks_filepath,
    parse_dates=['Date'],
    )

# Subsets lgs returns
jpm_min_date = df_jpm['Date'].min()
df_lgs_returns = df_lgs_returns[df_lgs_returns['Date'] >= jpm_min_date].reset_index(drop=True)

# Selects the international equity unhedged returns and benchmarks
df_lgs_returns_benchmarks_ieu = df_lgs_returns[['Date', 'IEu', 'MSCI.ACWI.EX.AUS_Index']]
df_lgs_returns_benchmarks_ieu = df_lgs_returns_benchmarks_ieu.rename(columns={'IEu': 'JPM Asset Class Return', 'MSCI.ACWI.EX.AUS_Index': 'JPM Asset Class Benchmark'})
df_lgs_returns_benchmarks_ieu.insert(1, 'LGS Asset Class Level 1', 'IE')
df_lgs_returns_benchmarks_ieu.insert(2, 'LGS Asset Class Level 2', 'IE')

df_sectors_returns_benchmarks = df_sectors[['Date', 'LGS Asset Class Level 1', 'LGS Asset Class Level 2', 'JPM Return', 'JPM Benchmark']]
df_sectors_returns_benchmarks = df_sectors_returns_benchmarks.rename(columns={'JPM Return': 'JPM Asset Class Return', 'JPM Benchmark': 'JPM Asset Class Benchmark'})

# Switches IE returns and benchmarks to unhedged
df_sectors_returns_benchmarks = df_sectors_returns_benchmarks[~df_sectors_returns_benchmarks['LGS Asset Class Level 1'].isin(['IE'])]
df_sectors_returns_benchmarks = pd.concat([df_sectors_returns_benchmarks, df_lgs_returns_benchmarks_ieu]).reset_index(drop=True)

# Filters AFI and IFI benchmarks
df_sectors_returns_benchmarks = df_sectors_returns_benchmarks[~df_sectors_returns_benchmarks['LGS Asset Class Level 2'].isin(['AFI', 'IFI'])].reset_index(drop=True)



df_managers_combined = pd.merge(
    left=df_managers_combined,
    right=df_sectors_returns_benchmarks,
    left_on=['Date', 'LGS Asset Class Level 1'],
    right_on=['Date', 'LGS Asset Class Level 1'],
    how='inner'
)


df_managers_combined = df_managers_combined.sort_values(['Date', 'LGS Asset Class Order', 'LGS Manager Order']).reset_index(drop=True)
df_managers_combined['LGS Weight Manager in Asset Class'] = df_managers_combined['JPM Market Value'] / df_managers_combined['LGS Asset Class Market Value']

# Calculates Style, Pure Active
df_managers_combined['Pure Active Excess Return'] = df_managers_combined['JPM Return'] - df_managers_combined['JPM Benchmark']
df_managers_combined['Style Excess Return'] = df_managers_combined['JPM Benchmark'] - df_managers_combined['JPM Asset Class Benchmark']

# Calculates weighted style
df_managers_combined['Weighted Asset Class Return'] = df_managers_combined['LGS Weight Manager in Asset Class'] * df_managers_combined['JPM Return']
df_managers_combined['Weighted Asset Class Style Benchmark'] = df_managers_combined['LGS Weight Manager in Asset Class'] * df_managers_combined['JPM Benchmark']
df_managers_combined['Weighted Asset Class Benchmark'] = df_managers_combined['LGS Weight Manager in Asset Class'] * df_managers_combined['JPM Asset Class Benchmark']
df_managers_combined['Weighted Pure Active Excess Return'] = df_managers_combined['LGS Weight Manager in Asset Class'] * df_managers_combined['Pure Active Excess Return']
df_managers_combined['Weighted Style Excess Return'] = df_managers_combined['LGS Weight Manager in Asset Class'] * df_managers_combined['Style Excess Return']
df_managers_combined['Weighted Asset Class Excess Return'] = df_managers_combined['LGS Weight Manager in Asset Class'] * (df_managers_combined['JPM Return'] - df_managers_combined['JPM Asset Class Benchmark'])

df_managers_style = df_managers_combined.groupby(['Date', 'LGS Asset Class Level 1', 'LGS Benchmark']).sum()

df_managers_style_asset_class = df_managers_combined.groupby(['Date', 'LGS Asset Class Level 1']).sum().reset_index(drop=False)
df_managers_style_asset_class = df_managers_style_asset_class.rename(columns={'JPM Market Value': 'Weighted Asset Class Sum Market Value'})
df_managers_style_asset_class = df_managers_style_asset_class[
    [
        'Date',
        'LGS Asset Class Level 1',
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
df_managers_fx = df_managers[df_managers['LGS Name'].isin(['FX Overlay'])].reset_index(drop=True)

df_lgs_benchmarks_fx = df_lgs_returns[['Date', 'IECurrencyHedge_Index']]

df_managers_fx = pd.merge(
    left=df_managers_fx,
    right=df_lgs_benchmarks_fx,
    left_on=['Date'],
    right_on=['Date']

)

df_managers_fx = df_managers_fx.drop('JPM Benchmark', axis=1)

df_managers_fx = df_managers_fx.rename(
    columns={
        'JPM Market Value': 'FX Market Value',
        'JPM Return': 'FX Return',
        'JPM Benchmark': 'FX Benchmark',
        'IECurrencyHedge_Index': 'FX Benchmark'
    }
)


df_managers_fx['FX Excess Return'] = df_managers_fx['FX Return'] - df_managers_fx['FX Benchmark']
df_managers_fx['LGS Asset Class Level 1'] = 'IE'
df_managers_fx = df_managers_fx[['Date', 'FX Market Value', 'FX Return', 'FX Benchmark', 'FX Excess Return', 'LGS Asset Class Level 1']]

df_managers_style_asset_class_fx = pd.merge(
    left=df_managers_style_asset_class,
    right=df_managers_fx,
    left_on=['Date', 'LGS Asset Class Level 1'],
    right_on=['Date', 'LGS Asset Class Level 1'],
    how='outer'
)

df_managers_style_asset_class_fx.fillna({'FX Market Value': 0, 'FX Return': 0, 'FX Benchmark': 0, 'FX Excess Return': 0}, inplace=True)

df_combined_all = pd.merge(
    left=df_combined,
    right=df_managers_style_asset_class_fx,
    left_on=['Date', 'LGS Asset Class Level 1'],
    right_on=['Date', 'LGS Asset Class Level 1']
)

df_combined_all = df_combined_all.sort_values(['Date', 'Strategy', 'LGS Asset Class Order'], ascending=[True, True, True]).reset_index(drop=True)




# Fix AR Restructure

# Overwrite PE, OA, DA
overwrite_weighted_asset_class_return = []
overwrite_weighted_asset_class_style_benchmark = []
overwrite_weighted_asset_class_benchmark = []
for i in range(0, len(df_combined_all)):
    if df_combined_all['LGS Asset Class Level 1'][i] in ['PE', 'OA', 'DA']:
        overwrite_weighted_asset_class_return.append(df_combined_all['JPM Return'][i])
        overwrite_weighted_asset_class_style_benchmark.append(df_combined_all['JPM Benchmark'][i])
        overwrite_weighted_asset_class_benchmark.append(df_combined_all['JPM Benchmark'][i])
    else:
        overwrite_weighted_asset_class_return.append(df_combined_all['Weighted Asset Class Return'][i])
        overwrite_weighted_asset_class_style_benchmark.append(df_combined_all['Weighted Asset Class Style Benchmark'][i])
        overwrite_weighted_asset_class_benchmark.append(df_combined_all['Weighted Asset Class Benchmark'][i])
df_combined_all['Weighted Asset Class Return'] = overwrite_weighted_asset_class_return
df_combined_all['Weighted Asset Class Style Benchmark'] = overwrite_weighted_asset_class_style_benchmark
df_combined_all['Weighted Asset Class Benchmark'] = overwrite_weighted_asset_class_benchmark


# Check sums
df_combined_all['Check Sum Asset Class Market Value Difference (Millions)'] = (df_combined_all['JPM Market Value'] - df_combined_all['Weighted Asset Class Sum Market Value'] - df_combined_all['FX Market Value'])/1000000
df_combined_all['Check Sum Asset Class Return Difference'] = df_combined_all['JPM Return'] - df_combined_all['Weighted Asset Class Return'] - df_combined_all['FX Return']
df_combined_all['Check Sum Asset Class Benchmark Difference'] = df_combined_all['JPM Benchmark'] - df_combined_all['Weighted Asset Class Benchmark'] - df_combined_all['FX Benchmark']

# check_ar1 = df_all_combined[df_all_combined['LGS Asset Class Level 1'].isin(['AR'])]
# check_ar2 = df_managers[df_managers['LGS Asset Class Level 1'].isin(['AR'])]


# Sets the dictionary for the holding period returns.
horizon_to_period_dict = {
        '1_': 1,
        '3_': 3
}


# # Direct Geometric Link
# df_multiperiod1 = df_combined_all.sort_values(['Strategy', 'Asset Class', 'Date'], ascending=[True, True, True]).reset_index(drop=True)
#
# for horizon, period in horizon_to_period_dict.items():
#
#     for column in [
#         'JPM Strategy Return',
#         'JPM Strategy Benchmark',
#         'TAA',
#         'DAA',
#         'SAA',
#         'SS',
#         'In',
#         'Weighted Pure Active Excess Return',
#         'Weighted Style Excess Return',
#         'FX Excess Return',
#         'Asset Class Total',
#         'Strategy Total by Asset Class Sum'
#     ]:
#
#         column_name = horizon + column
#         return_type = column
#
#         if period <= 12:
#             df_multiperiod1[column_name] = (
#                 df_multiperiod1
#                 .groupby(['Strategy', 'Asset Class'])[return_type]
#                 .rolling(period)
#                 .apply(lambda r: np.prod(1+r)-1, raw=False)
#                 .reset_index(drop=False)[return_type]
#             )
#
#         elif period > 12:
#             df_multiperiod1[column_name] = (
#                 df_multiperiod1
#                 .groupby(['Strategy', 'Asset Class'])[return_type]
#                 .rolling(period)
#                 .apply(lambda r: (np.prod(1+r)**(12/period))-1, raw=False)
#                 .reset_index(drop=False)[return_type]
#             )
#
#     # df_multiperiod_asset_class_total_horizon = df_multiperiod1.groupby(['Date', 'Strategy'])[horizon + 'Asset Class Total'].sum().reset_index(drop=False)
#     #
#     # df_multiperiod_asset_class_total_horizon = df_multiperiod_asset_class_total_horizon.rename(columns={horizon + 'Asset Class Total': horizon + 'Strategy Total by Asset Class Sum'})
#     #
#     # df_multiperiod1 = pd.merge(
#     #         left=df_multiperiod1,
#     #         right=df_multiperiod_asset_class_total_horizon,
#     #         left_on=['Date', 'Strategy'],
#     #         right_on=['Date', 'Strategy'],
#     #         how='inner'
#     # )
#
#     df_multiperiod1[horizon + 'JPM Strategy Excess'] = df_multiperiod1[horizon + 'JPM Strategy Return'] - df_multiperiod1[horizon + 'JPM Strategy Benchmark']
#
#     df_multiperiod1[horizon + 'Residual_SAA'] = df_multiperiod1[horizon + 'SAA'] - (df_multiperiod1[horizon + 'TAA'] + df_multiperiod1[horizon + 'DAA'])
#
#     df_multiperiod1[horizon + 'Residual_SS'] = df_multiperiod1[horizon + 'SS'] - (df_multiperiod1[horizon + 'FX Excess Return'] + df_multiperiod1[horizon + 'Weighted Pure Active Excess Return'] + df_multiperiod1[horizon + 'Weighted Style Excess Return'])
#
#     df_multiperiod1[horizon + 'Residual_Asset_Class_Total'] = df_multiperiod1[horizon + 'Asset Class Total'] - (df_multiperiod1[horizon + 'SAA'] + df_multiperiod1[horizon + 'SS'] + df_multiperiod1[horizon + 'In'])
#
#     # df_multiperiod1[horizon + 'Residual_Strategy'] = df_multiperiod1[horizon + 'Strategy Total'] - df_multiperiod1[horizon + 'Strategy Total by Asset Class Sum']
#
#     df_multiperiod1[horizon + 'Residual_Actual'] = df_multiperiod1[horizon + 'JPM Strategy Excess'] - df_multiperiod1[horizon + 'Strategy Total by Asset Class Sum']
#
#     df_multiperiod1 = df_multiperiod1.sort_values(['Strategy', 'Asset Class', 'Date'], ascending=[True, True, True]).reset_index(drop=True)
#
#
# df_multiperiod1 = df_multiperiod1.sort_values(['Date', 'Strategy', 'LGS Asset Class Order'], ascending=[True, True, True]).reset_index(drop=True)
#
# df_multiperiod1.to_csv('U:/CIO/#Data/output/prototype_attribution_v3_python_1.csv', index=False)


# Recalculated Geometric Link
df_multiperiod2 = df_combined_all.sort_values(['Strategy', 'Asset Class', 'Date'], ascending=[True, True, True]).reset_index(drop=True)

for horizon, period in horizon_to_period_dict.items():

    for column in [
        'Portfolio Weight',
        'Dynamic Weight',
        'Benchmark Weight',
    ]:

        df_multiperiod2[horizon + column] = df_multiperiod2[column].rolling(period).mean().reset_index(drop=False)[column]

    for column in [
        'JPM Strategy Return',
        'JPM Strategy Benchmark',
        'JPM Return',
        'JPM Benchmark',
        'Weighted Asset Class Return',
        'Weighted Asset Class Style Benchmark',
        'Weighted Asset Class Benchmark',
        'FX Return',
        'FX Benchmark',
        'Asset Class Total'
    ]:

        column_name = horizon + column
        return_type = column

        if period <= 12:
            df_multiperiod2[column_name] = (
                df_multiperiod2
                .groupby(['Strategy', 'Asset Class'])[return_type]
                .rolling(period)
                .apply(lambda r: np.prod(1+r)-1, raw=False)
                .reset_index(drop=False)[return_type]
            )

        elif period > 12:
            df_multiperiod2[column_name] = (
                df_multiperiod2
                .groupby(['Strategy', 'Asset Class'])[return_type]
                .rolling(period)
                .apply(lambda r: (np.prod(1+r)**(12/period))-1, raw=False)
                .reset_index(drop=False)[return_type]
            )

    df_multiperiod2[horizon + 'JPM Excess'] = df_multiperiod2[horizon + 'JPM Return'] - df_multiperiod2[horizon + 'JPM Benchmark']

    df_multiperiod2[horizon + 'TAA'] = (df_multiperiod2[horizon + 'Portfolio Weight'] - df_multiperiod2[horizon + 'Dynamic Weight']) * df_multiperiod2[horizon + 'JPM Benchmark']

    df_multiperiod2[horizon + 'DAA'] = (df_multiperiod2[horizon + 'Dynamic Weight'] - df_multiperiod2[horizon + 'Benchmark Weight']) * df_multiperiod2[horizon + 'JPM Benchmark']

    df_multiperiod2[horizon + 'SAA'] = df_multiperiod2[horizon + 'TAA'] + df_multiperiod2[horizon + 'DAA']

    df_multiperiod2[horizon + 'SS'] = (df_multiperiod2[horizon + 'Benchmark Weight']) * (df_multiperiod2[horizon + 'JPM Return'] - df_multiperiod2[horizon + 'JPM Benchmark'])

    df_multiperiod2[horizon + 'In'] = (df_multiperiod2[horizon + 'Portfolio Weight'] - df_multiperiod2[horizon + 'Benchmark Weight']) * (df_multiperiod2[horizon + 'JPM Return'] - df_multiperiod2[horizon + 'JPM Benchmark'])

    df_multiperiod2[horizon + 'Asset Class Total'] = df_multiperiod2[horizon + 'SAA'] + df_multiperiod2[horizon + 'SS'] + df_multiperiod2[horizon + 'In']

    df_combined_asset_class_total = df_multiperiod2.groupby(['Date', 'Strategy'])[horizon + 'Asset Class Total'].sum().reset_index(drop=False)

    df_combined_asset_class_total = df_combined_asset_class_total.rename(columns={horizon + 'Asset Class Total': horizon + 'Strategy Total'})

    df_multiperiod_asset_class_total_horizon = df_multiperiod2.groupby(['Date', 'Strategy'])[horizon + 'Asset Class Total'].sum().reset_index(drop=False)

    df_multiperiod_asset_class_total_horizon = df_multiperiod_asset_class_total_horizon.rename(columns={horizon + 'Asset Class Total': horizon + 'Strategy Total by Asset Class Sum'})

    df_multiperiod2 = pd.merge(
            left=df_multiperiod2,
            right=df_multiperiod_asset_class_total_horizon,
            left_on=['Date', 'Strategy'],
            right_on=['Date', 'Strategy'],
            how='inner'
    )

    # Calculates Pure Active, Style, FX
    df_multiperiod2[horizon + 'Pure Active'] = df_multiperiod2[horizon + 'Benchmark Weight'] * (df_multiperiod2[horizon + 'Weighted Asset Class Return'] - df_multiperiod2[horizon + 'Weighted Asset Class Style Benchmark'])

    df_multiperiod2[horizon + 'Style'] = df_multiperiod2[horizon + 'Benchmark Weight'] * (df_multiperiod2[horizon + 'Weighted Asset Class Style Benchmark'] - df_multiperiod2[horizon + 'Weighted Asset Class Benchmark'])

    df_multiperiod2[horizon + 'FX'] = df_multiperiod2[horizon + 'Benchmark Weight'] * (df_multiperiod2[horizon + 'FX Return'] - df_multiperiod2[horizon + 'FX Benchmark'])

    df_multiperiod2[horizon + 'JPM Strategy Excess'] = df_multiperiod2[horizon + 'JPM Strategy Return'] - df_multiperiod2[horizon + 'JPM Strategy Benchmark']

    df_multiperiod2[horizon + 'Residual_SAA'] = df_multiperiod2[horizon + 'SAA'] - (df_multiperiod2[horizon + 'TAA'] + df_multiperiod2[horizon + 'DAA'])

    df_multiperiod2[horizon + 'Residual_SS'] = df_multiperiod2[horizon + 'SS'] - (df_multiperiod2[horizon + 'FX'] + df_multiperiod2[horizon + 'Pure Active'] + df_multiperiod2[horizon + 'Style'])

    df_multiperiod2[horizon + 'Residual_Asset_Class_Total'] = df_multiperiod2[horizon + 'Asset Class Total'] - (df_multiperiod2[horizon + 'SAA'] + df_multiperiod2[horizon + 'SS'] + df_multiperiod2[horizon + 'In'])

    # df_multiperiod2[horizon + 'Residual_Strategy'] = df_multiperiod2[horizon + 'Strategy Total'] - df_multiperiod2[horizon + 'Strategy Total by Asset Class Sum']

    df_multiperiod2[horizon + 'Residual_Actual'] = df_multiperiod2[horizon + 'JPM Strategy Excess'] - df_multiperiod2[horizon + 'Strategy Total by Asset Class Sum']

    df_multiperiod2 = df_multiperiod2.sort_values(['Strategy', 'Asset Class', 'Date'], ascending=[True, True, True]).reset_index(drop=True)


df_multiperiod2 = df_multiperiod2.sort_values(['Date', 'Strategy', 'LGS Asset Class Order'], ascending=[True, True, True]).reset_index(drop=True)

df_multiperiod2.to_csv('U:/CIO/#Data/output/prototype_attribution_v3_python_2.csv', index=False)

# Fix Strategy Total


# Creates Tables
strategy_list = ['High Growth', 'Balanced Growth', 'Balanced', 'Conservative', 'Growth', 'Employer Reserve']
df_asset_class_sort = df_lgs[df_lgs['LGS Sector Aggregate'].isin([1])][['LGS Name', 'LGS Asset Class Order']].reset_index(drop=True)
asset_class_sort_dict = {df_asset_class_sort['LGS Name'][i]: df_asset_class_sort['LGS Asset Class Order'][i] for i in range(0, len(df_asset_class_sort))}


# Makes the pivot tables
def pivot_table(df, var):
    columns_list = ['Strategy', 'LGS Name', var]
    df = df[columns_list]
    df = df.pivot(index='LGS Name', columns='Strategy', values=var)
    df = df[strategy_list]
    df = df.reset_index(drop=False)
    df['asset_class_sort'] = df['LGS Name'].map(asset_class_sort_dict)
    df = df.sort_values(['asset_class_sort'])
    df = df.drop('asset_class_sort', axis=1)
    df = df.reset_index(drop=True)
    df = df.rename(columns={'LGS Name': 'Attribution'})
    #df['Manager'] = [modelcode_to_name_dict[df['Manager'][i]] for i in range(0, len(df))]
    #df.columns = latex_column_names
    return df


def sum_pivot_table(df, output_variable_name, drop_column='Attribution'):
    df = df.drop(drop_column, axis=1)
    df = df.sum().reset_index(drop=False).transpose()
    df.columns = df.iloc[0]
    df = df[1:]
    df.insert(0, 'Attribution', output_variable_name)
    return df


# Take current month
df_current2 = df_multiperiod2[df_multiperiod2['Date'].isin([report_date])]


# Table 1
df_current2_table1a = pivot_table(df_current2, '3_JPM Strategy Return')
df_current2_table1b = pivot_table(df_current2, '3_JPM Strategy Benchmark')
df_current2_table1c = pivot_table(df_current2, '3_JPM Strategy Excess')

df_current2_table1a = df_current2_table1a.drop('Attribution', axis=1)
df_current2_table1b = df_current2_table1b.drop('Attribution', axis=1)
df_current2_table1c = df_current2_table1c.drop('Attribution', axis=1)

df_current2_table1a.insert(0, 'Attribution', 'Portfolio')
df_current2_table1b.insert(0, 'Attribution', 'Benchmark')
df_current2_table1c.insert(0, 'Attribution', 'Active')

df_current2_table1a = df_current2_table1a.drop_duplicates(keep='first')
df_current2_table1b = df_current2_table1b.drop_duplicates(keep='first')
df_current2_table1c = df_current2_table1c.drop_duplicates(keep='first')

df_current2_table1 = pd.concat([df_current2_table1a, df_current2_table1b, df_current2_table1c], axis=0).reset_index(drop=True)
df_current2_table1[strategy_list] = (df_current2_table1[strategy_list].astype(float).round(4)*100)


# Table 2
df_current2_table2a_aa = pivot_table(df_current2, '3_SAA')
df_current2_table2a_aa = sum_pivot_table(df_current2_table2a_aa, 'AA')

df_current2_table2b_ss = pivot_table(df_current2, '3_SS')
df_current2_table2b_ss = sum_pivot_table(df_current2_table2b_ss, 'SS')

df_current2_table2c_in = pivot_table(df_current2, '3_In')
df_current2_table2c_in = sum_pivot_table(df_current2_table2c_in, 'In')

df_current2_table2d_residual = (
        df_current2_table1c.loc[0].drop('Attribution') -
        df_current2_table2a_aa.loc[0].drop('Attribution') -
        df_current2_table2b_ss.loc[0].drop('Attribution') -
        df_current2_table2c_in.loc[0].drop('Attribution')
)

df_current2_table2d_residual = df_current2_table2d_residual.reset_index(drop=False).transpose()
df_current2_table2d_residual.columns = df_current2_table2d_residual.iloc[0]
df_current2_table2d_residual = df_current2_table2d_residual[1:]
df_current2_table2d_residual.insert(0, 'Attribution', 'Residual Actual')

df_current2_table2 = pd.concat(
    [
        df_current2_table2a_aa,
        df_current2_table2b_ss,
        df_current2_table2c_in,
        df_current2_table2d_residual,
        df_current2_table1c
    ]
    , axis=0
).reset_index(drop=True)
df_current2_table2[strategy_list] = (df_current2_table2[strategy_list].astype(float).round(4)*100)


# Table 3
df_current2_table3a_taa = pivot_table(df_current2, '3_TAA')
df_current2_table3a_taa = sum_pivot_table(df_current2_table3a_taa, 'TAA')

df_current2_table3b_daa = pivot_table(df_current2, '3_DAA')
df_current2_table3b_daa = sum_pivot_table(df_current2_table3b_daa, 'DAA')

df_current2_table3c_residual_aa = (
    df_current2_table2a_aa.loc[0].drop('Attribution') -
    df_current2_table3a_taa.loc[0].drop('Attribution') -
    df_current2_table3b_daa.loc[0].drop('Attribution')
)

df_current2_table3c_residual_aa = df_current2_table3c_residual_aa.reset_index(drop=False).transpose()
df_current2_table3c_residual_aa.columns = df_current2_table3c_residual_aa.iloc[0]
df_current2_table3c_residual_aa = df_current2_table3c_residual_aa[1:]
df_current2_table3c_residual_aa.insert(0, 'Attribution', 'Residual AA')

df_current2_table3d_fx = pivot_table(df_current2, '3_FX')
df_current2_table3d_fx = sum_pivot_table(df_current2_table3d_fx, 'FX')

df_current2_table3e_pure_active = pivot_table(df_current2, '3_Pure Active')
df_current2_table3e_pure_active = sum_pivot_table(df_current2_table3e_pure_active, 'Pure Active')

df_current2_table3f_style = pivot_table(df_current2, '3_Style')
df_current2_table3f_style = sum_pivot_table(df_current2_table3f_style, 'Style')


df_current2_table3g_residual_ss = (
    df_current2_table2b_ss.loc[0].drop('Attribution') -
    df_current2_table3d_fx.loc[0].drop('Attribution') -
    df_current2_table3e_pure_active.loc[0].drop('Attribution') -
    df_current2_table3f_style.loc[0].drop('Attribution')
)

df_current2_table3g_residual_ss = df_current2_table3g_residual_ss.reset_index(drop=False).transpose()
df_current2_table3g_residual_ss.columns = df_current2_table3g_residual_ss.iloc[0]
df_current2_table3g_residual_ss = df_current2_table3g_residual_ss[1:]
df_current2_table3g_residual_ss.insert(0, 'Attribution', 'Residual SS')

df_current2_table3h_residual_aa_ss = (
        df_current2_table3c_residual_aa.loc[0].drop('Attribution') +
        df_current2_table3g_residual_ss.loc[0].drop('Attribution')
)

df_current2_table3h_residual_aa_ss = df_current2_table3h_residual_aa_ss.reset_index(drop=False).transpose()
df_current2_table3h_residual_aa_ss.columns = df_current2_table3h_residual_aa_ss.iloc[0]
df_current2_table3h_residual_aa_ss = df_current2_table3h_residual_aa_ss[1:]
df_current2_table3h_residual_aa_ss.insert(0, 'Attribution', 'Residual AA + SS')

df_current2_table3 = pd.concat(
    [
        df_current2_table3a_taa,
        df_current2_table3b_daa,
        df_current2_table3c_residual_aa,
        df_current2_table3d_fx,
        df_current2_table3e_pure_active,
        df_current2_table3f_style,
        df_current2_table3g_residual_ss,
        df_current2_table2c_in,
        df_current2_table2d_residual
    ]
)

df_current2_table3i_sum = sum_pivot_table(df_current2_table3, 'Active', drop_column='Attribution')

df_current2_table3 = pd.concat([df_current2_table3, df_current2_table3i_sum], axis=0)

df_current2_table3[strategy_list] = (df_current2_table3[strategy_list].astype(float).round(4)*100)


# Table 4
# Merges Table 2 and Table 3


# Table 5
df_current2_table5a_aa = pivot_table(df_current2, '3_SAA')
df_current2_table5 = pd.concat([df_current2_table5a_aa, df_current2_table2a_aa])
df_current2_table5[strategy_list] = (df_current2_table5[strategy_list].astype(float).round(4)*100)


# Table 6
df_current2_table6a_taa = pivot_table(df_current2, '3_TAA')
df_current2_table6 = pd.concat([df_current2_table6a_taa, df_current2_table3a_taa])
df_current2_table6[strategy_list] = (df_current2_table6[strategy_list].astype(float).round(4)*100)


# Table 7
df_current2_table7a_daa = pivot_table(df_current2, '3_DAA')
df_current2_table7 = pd.concat([df_current2_table7a_daa, df_current2_table3b_daa])
df_current2_table7[strategy_list] = (df_current2_table7[strategy_list].astype(float).round(4)*100)

# Table 8
df_current2_table8a_residual_aa = pivot_table(df_current2, '3_Residual_SAA')
df_current2_table8 = pd.concat([df_current2_table8a_residual_aa, df_current2_table3c_residual_aa])
df_current2_table8[strategy_list] = (df_current2_table8[strategy_list].astype(float).round(4)*100)


# Table 9
df_current2_table9a_ss = pivot_table(df_current2, '3_SS')
df_current2_table9 = pd.concat([df_current2_table9a_ss, df_current2_table2b_ss])
df_current2_table9[strategy_list] = (df_current2_table9[strategy_list].astype(float).round(4)*100)


# Table 10
df_current2_table10a_fx = pivot_table(df_current2, '3_FX')
df_current2_table10 = pd.concat([df_current2_table10a_fx, df_current2_table3d_fx])
df_current2_table10[strategy_list] = (df_current2_table10[strategy_list].astype(float).round(4)*100)


# Table 11
df_current2_table11a_pure_active = pivot_table(df_current2, '3_Pure Active')
df_current2_table11 = pd.concat([df_current2_table11a_pure_active, df_current2_table3e_pure_active])
df_current2_table11[strategy_list] = (df_current2_table11[strategy_list].astype(float).round(4)*100)


# Table 12
df_current2_table12a_style = pivot_table(df_current2, '3_Style')
df_current2_table12 = pd.concat([df_current2_table12a_style, df_current2_table3f_style])
df_current2_table12[strategy_list] = (df_current2_table12[strategy_list].astype(float).round(4)*100)


# Table 13
df_current2_table13a_residual_ss = pivot_table(df_current2, '3_Residual_SS')
df_current2_table13 = pd.concat([df_current2_table13a_residual_ss, df_current2_table3g_residual_ss])
df_current2_table13[strategy_list] = (df_current2_table13[strategy_list].astype(float).round(4)*100)

