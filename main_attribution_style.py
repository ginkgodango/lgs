import datetime
import pandas as pd
import numpy as np
import win32com.client
import matplotlib
import matplotlib.pyplot as plt
import attribution.extraction
from dateutil.relativedelta import relativedelta

start_date = datetime.datetime(2018, 7, 31)
end_date = datetime.datetime(2019, 6, 30)

input_directory = 'U:/CIO/#Investment_Report/Data/input/'
output_directory = 'U:/CIO/#Attribution/tables/base/'

table_filename = 'link_2019-06-30.csv'
# returns_filename = 'returns_2019-06-30.csv'
returns_filename = 'returns_2019-08-31.csv'
# market_values_filename = 'market_values_2019-06-30.csv'
market_values_filename = 'market_values_2019-08-31.csv'
# asset_allocations_filename = 'asset_allocations_2019-06-30.csv'
asset_allocations_filename = 'asset_allocations_2019-08-31.csv'

latex_summary1_column_names = ['Returns', 'High Growth', "Bal' Growth", 'Balanced', 'Conservative', 'Growth', "Emp' Reserve"]
latex_summary2_column_names = ['Attribution', 'High Growth', "Bal' Growth", 'Balanced', 'Conservative', 'Growth', "Emp' Reserve"]
latex_column_names = ['Asset Class', 'High Growth', "Bal' Growth", 'Balanced', 'Conservative', 'Growth', "Emp' Reserve"]

# Creates variable names for linked table
periods = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1
market_value = str(periods) + '_market_value'
r_portfolio = str(periods) + '_r_portfolio'
r_benchmark = str(periods) + '_r_benchmark'
r_excess = str(periods) + '_r_excess'
r_diff = str(periods) + '_r_diff'
r_diff_sq = str(periods) + '_r_diff_sq'
r_active_contribution = str(periods) + '_r_active_contribution'
w_portfolio = str(periods) + '_w_portfolio'
w_benchmark = str(periods) + '_w_benchmark'
AA = str(periods) + '_AA'
SS = str(periods) + '_SS'
interaction = str(periods) + '_interaction'
total_effect = str(periods) + '_total_effect'
residual = str(periods) + '_residual'
total = str(periods) + '_total'
r_pure = str(periods) + '_r_pure'
r_style = str(periods) + '_r_style'
r_passive = str(periods) + '_r_passive'
r_pure_active = str(periods) + '_r_pure_active'
r_style_active = str(periods) + '_r_style_active'
r_effect_active = str(periods) + '_r_effect_active'
r_actual_portfolio = str(periods) + '_r_actual_portfolio'
r_actual_benchmark = str(periods) + '_r_actual_benchmark'
r_residual = str(periods) + '_r_residual'
r_total_active = str(periods) + '_r_total_active'

# Loads table
df_table = attribution.extraction.load_table(input_directory + 'link/' + table_filename)

# Loads returns
df_returns = attribution.extraction.load_returns(input_directory + 'returns/' + returns_filename)

# Reshapes returns dataframe from wide to long
df_returns = df_returns.transpose().reset_index(drop=False).rename(columns={'index': 'Manager'})
df_returns = pd.melt(df_returns, id_vars=['Manager'], value_name='1_r')

# Selects returns for this month or within a date_range
df_returns = df_returns[(df_returns['Date'] >= start_date) & (df_returns['Date'] <= end_date)].reset_index(drop=True)

df_benchmarks = pd.merge(left=df_returns, right=df_table, left_on=['Manager'], right_on=['Associated Benchmark'],
                         how='inner')
df_benchmarks = df_benchmarks[['Date', 'Associated Benchmark', '1_r', 'ModelCode', 'Sector', 'Style']]
df_benchmarks.columns = ['Date', 'Benchmark Name', 'bmk_1_r', 'ModelCode', 'Sector', 'Style']

df_returns_benchmarks = pd.merge(left=df_returns, right=df_benchmarks, left_on=['Date', 'Manager'],
                                 right_on=['Date', 'ModelCode'], how='inner')

# Loads market values
df_market_values = attribution.extraction.load_market_values(input_directory + 'market_values/' + market_values_filename)

# Reshapes market values dataframe from wide to long
df_market_values = df_market_values.transpose().reset_index(drop=False).rename(columns={'index': 'Manager'})
df_market_values = pd.melt(df_market_values, id_vars=['Manager'], value_name='Market Value')

# Forwards the market values by 1 month, which lags it 1 month relative to the returns.
df_market_values['Date'] = df_market_values['Date'] + pd.offsets.MonthEnd(1)

# Selects market values for this month
# df_market_values = df_market_values[df_market_values['Date'] == df_market_values['Date'].max()].reset_index(drop=True)
df_market_values = df_market_values[(df_market_values['Date'] >= start_date) & (df_market_values['Date'] <= end_date)].reset_index(drop=True)

# Joins 1 month returns with market values
df_main = pd.merge(df_returns_benchmarks, df_market_values, how='outer', on=['Manager', 'Date'])

# Loads strategy asset allocations
asset_allocations = ['High Growth', 'Balanced Growth', 'Balanced', 'Conservative', 'Growth', 'Employer Reserve']

df_asset_allocations = pd.read_csv(
    input_directory + 'allocations/' + asset_allocations_filename,
    parse_dates=['Date'],
    infer_datetime_format=True,
    float_precision='round_trip'
)

# Converts to decimals
df_asset_allocations['Portfolio'] = df_asset_allocations['Portfolio']/100
df_asset_allocations['Benchmark'] = df_asset_allocations['Benchmark']/100

# Forwards the asset allocations by 1 month, which lags it 1 month relative to the returns and market values.
df_asset_allocations['Date'] = df_asset_allocations['Date'] + pd.offsets.MonthEnd(1)


# STYLE ATTRIBUTION BELOW HERE
# Finds benchmark returns
df_sector_bmk_r = df_main[df_main['Sector'].isin(['TO'])][['Manager', 'Date', 'bmk_1_r']].reset_index(drop=True)

# Filters IEu and BO from market value
df_main = df_main[~df_main['Manager'].isin(['IEu', 'BO', 'Bonds_BO'])].reset_index(drop=True)
df_main = df_main[~df_main['Market Value'].isin([np.nan])]

# Filters managers with NaN returns
# df_main = df_main[~df_main['1_r'].isin([np.nan])]
df_main['1_r'].fillna(0, inplace=True)

# Imports transition file
df_transition = pd.read_csv(
    'U:/CIO/#Attribution/data/input/transition_201909.csv',
    parse_dates=['Date'],
    infer_datetime_format=True,
    float_precision='round_trip'
)

df_transition_remove = df_transition[['Manager Remove', 'Date']]
df_transition_add = df_transition.drop(['Manager Remove'], axis=1)

# Remove transitioning accounts
df_main = pd.merge(
    left=df_main,
    right=df_transition_remove,
    left_on=['Manager', 'Date'],
    right_on=['Manager Remove', 'Date'],
    how='outer',
    indicator=True
)
df_main = df_main[~df_main['_merge'].isin(['both'])]
df_main = df_main.drop(['Manager Remove', '_merge'], axis=1)

# Adds the transition account to the main dataframe
df_main = pd.concat([df_main, df_transition_add], sort=False)

# Finds sector market values
df_sector_sum_mv = df_main.groupby(['Date', 'Sector']).sum()['Market Value'].reset_index(drop=False)


# START TEST
df_sector_mv = df_market_values[df_market_values['Manager'].isin(['AE', 'IE', 'DP', 'ILP', 'BO', 'AR', 'AC', 'CO', 'SAS'])]
df_test1 = pd.merge(left=df_sector_sum_mv, right=df_sector_mv, left_on=['Sector', 'Date'], right_on=['Manager', 'Date'])
df_test1['deviation'] = (df_test1['Market Value_x'] - df_test1['Market Value_y'])/1000000
# END TEST


# Joins the dataframes together
df_main = pd.merge(left=df_main, right=df_sector_sum_mv, left_on=['Sector', 'Date'], right_on=['Sector', 'Date'])
df_main = pd.merge(left=df_main, right=df_sector_bmk_r, left_on=['Sector', 'Date'], right_on=['Manager', 'Date'])

def style_calculate(data):
    d = dict()
    d['manager_1_r'] = np.dot(data['1_r'], data['Market Value_x']/data['Market Value_y'])
    d['style_1_r'] = np.dot(data['bmk_1_r_x'], data['Market Value_x']/data['Market Value_y'])
    d['passive_1_r'] = np.dot(data['bmk_1_r_y'], data['Market Value_x']/data['Market Value_y'])
    d['Market Value'] = np.sum(data['Market Value_x'])
    return pd.Series(d)


df_style = df_main.groupby(['Date', 'Sector', 'Style']).apply(style_calculate).reset_index(drop=False)
df_style['pure_1_ac'] = df_style['manager_1_r'] - df_style['style_1_r']
df_style['style_1_ac'] = df_style['style_1_r'] - df_style['passive_1_r']
df_style['total_1_ac'] = df_style['manager_1_r'] - df_style['passive_1_r']

df_sector_style = df_style.groupby(['Date', 'Sector']).sum().reset_index(drop=False)

# Removes unnecessary bond sectors
df_sector_style = df_sector_style[~df_sector_style['Sector'].isin(['IFI'])]

# Adds FX Overlay to IE returns
#df_fx = df_returns[df_returns['Manager'].isin(['IECurrencyOverlay_IE'])]

df_fx = df_returns_benchmarks[df_returns_benchmarks['Manager'].isin(['IECurrencyOverlay_IE'])]

df_fx = df_fx.rename(columns={'1_r': '1_r_fx', 'bmk_1_r': 'bmk_1_r_fx'})
df_fx = df_fx.drop(['Manager', 'Benchmark Name', 'ModelCode', 'Sector', 'Style'], axis=1)
df_fx.insert(0, 'Sector', 'IE')

df_sector_style = pd.merge(
    left=df_sector_style,
    right=df_fx,
    left_on=['Sector', 'Date'],
    right_on=['Sector', 'Date'],
    how='outer'
)
df_sector_style['1_r_fx'].fillna(0, inplace=True)
df_sector_style['manager_1_r'] = df_sector_style['manager_1_r'] + df_sector_style['1_r_fx']
df_sector_style['pure_1_ac'] = df_sector_style['pure_1_ac'] + df_sector_style['1_r_fx'] - df_sector_style['bmk_1_r_fx']
df_sector_style['total_1_ac'] = df_sector_style['total_1_ac'] + df_sector_style['1_r_fx'] - df_sector_style['bmk_1_r_fx']

# START TEST df_sector_style
df_sector = df_returns_benchmarks[df_returns_benchmarks['Manager'].isin(['AE', 'IE', 'DP', 'ILP', 'BO', 'AR', 'AC', 'CO', 'SAS'])]
df_sector = df_sector.drop(['ModelCode', 'Sector', 'Style'], axis=1)

df_test2 = pd.merge(
    left=df_sector_style,
    right=df_sector,
    left_on=['Date', 'Sector'],
    right_on=['Date', 'Manager'],
    how='outer'
)

df_test2['Deviation'] = df_test2['manager_1_r'] - df_test2['1_r']

df_test3 = pd.merge(
    left=df_test2,
    right=df_sector_mv,
    left_on=['Sector', 'Date'],
    right_on=['Manager', 'Date'],
    how='outer'
)

df_test3['deviation_mv'] = (df_test3['Market Value_x'] - df_test3['Market Value_y'])/1000000
#END TEST

def link(data):
    d = dict()
    d[r_portfolio] = (np.prod(1 + data['manager_1_r']) - 1)
    d[r_style] = (np.prod(1 + data['style_1_r']) - 1)
    d[r_passive] = (np.prod(1 + data['passive_1_r']) - 1)
    d[r_pure_active] = (np.prod(1 + data['manager_1_r']) - 1) - (np.prod(1 + data['style_1_r']) - 1)
    d[r_style_active] = (np.prod(1 + data['style_1_r']) - 1) - (np.prod(1 + data['passive_1_r']) - 1)
    d[r_effect_active] = (np.prod(1 + data['manager_1_r']) - 1) - (np.prod(1 + data['passive_1_r']) - 1)
    d[r_actual_portfolio] = (np.prod(1 + data['1_r']) - 1)
    d[r_actual_benchmark] = (np.prod(1 + data['bmk_1_r']) - 1)
    d[r_residual] = (
            ((np.prod(1 + data['manager_1_r']) - 1) - (np.prod(1 + data['passive_1_r']) - 1)) -
            ((np.prod(1 + data['1_r']) - 1) - (np.prod(1 + data['bmk_1_r']) - 1))
    )
    return pd.Series(d)


# df_sector_style_linked = df_sector_style.groupby(['Sector']).apply(link).reset_index(drop=False)
df_sector_style_linked = df_test3.groupby(['Sector']).apply(link).reset_index(drop=False)
df_sector_style_linked[r_total_active] = df_sector_style_linked[r_effect_active] - df_sector_style_linked[r_residual]

# Merge test3 with asset allocations
# Multiply returns with asset allocations
assetclass_to_modelcode_dict = {
    'Australian Equity': 'AE',
    'International Equity': 'IE',
    'Australian Listed Property': 'ALP',
    'Property': 'DP',
    'Global Property': 'ILP',
    'Bonds': 'BO',
    'Absolute Return': 'AR',
    'Commodities': 'CO',
    'Cash': 'AC',
    'Legacy Private Equity': 'LPE',
    'Private Equity': 'PE',
    'Opportunistic Alternatives': 'OA',
    'Defensive Alternatives': 'DA',
    'Option Overlay': 'OO',
    'Forwards': 'FW',
    'Total': 'TO'
}

df_asset_allocations['Sector'] = [
    assetclass_to_modelcode_dict[df_asset_allocations['Asset Class'][i]]
    for i in range(0,len(df_asset_allocations))
]
df_asset_allocations = df_asset_allocations.drop(['Market Value', 'Unnamed: 4'], axis=1)

df_test4 = pd.merge(
    left=df_test3,
    right=df_asset_allocations,
    left_on=['Sector', 'Date'],
    right_on=['Sector', 'Date'],
    how='inner'
)


def final(data):
    d = dict()
    d[r_portfolio] = (np.prod(1 + data['manager_1_r']) - 1)
    d[r_style] = (np.prod(1 + data['style_1_r']) - 1)
    d[r_passive] = (np.prod(1 + data['passive_1_r']) - 1)
    d[w_portfolio] = np.average(data['Portfolio'])
    d[w_benchmark] = np.average(data['Benchmark'])
    d[r_actual_portfolio] = (np.prod(1 + data['1_r']) - 1)
    d[r_actual_benchmark] = (np.prod(1 + data['bmk_1_r']) - 1)
    return pd.Series(d)


df_final = df_test4.groupby(['Strategy', 'Sector']).apply(final).reset_index(drop=False)
df_final[r_pure_active] = df_final[w_portfolio] * (df_final[r_portfolio] - df_final[r_style])
df_final[r_style_active] = df_final[w_portfolio] * (df_final[r_style] - df_final[r_passive])
df_final[r_effect_active] = df_final[r_pure_active] + df_final[r_style_active]

df_final[r_active_contribution] = df_final[w_portfolio] * (df_final[r_actual_portfolio] - df_final[r_actual_benchmark])
df_final[r_residual] = df_final[r_active_contribution] - df_final[r_effect_active]
df_final[r_total_active] = df_final[r_residual] + df_final[r_effect_active]
