import datetime
import pandas as pd
import numpy as np
import win32com.client
import matplotlib
import matplotlib.pyplot as plt
import attribution.extraction
from dateutil.relativedelta import relativedelta

start_date = datetime.datetime(2019, 10, 31)
end_date = datetime.datetime(2019, 12, 31)

input_directory = 'U:/CIO/#Investment_Report/Data/input/'
output_directory = 'U:/CIO/#Attribution/tables/style/'

table_filename = 'link_2019-12-31.csv'
returns_filename = 'returns_2019-12-31.csv'
market_values_filename = 'market_values_2019-12-31.csv'
asset_allocations_filename = 'asset_allocations_2019-12-31.csv'

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

df_benchmarks = pd.merge(
    left=df_returns,
    right=df_table,
    left_on=['Manager'],
    right_on=['Associated Benchmark'],
    how='inner'
)
df_benchmarks = df_benchmarks[['Date', 'Associated Benchmark', '1_r', 'ModelCode', 'Sector', 'Style']]
df_benchmarks.columns = ['Date', 'Benchmark Name', 'bmk_1_r', 'ModelCode', 'Sector', 'Style']

df_returns_benchmarks = pd.merge(
    left=df_returns,
    right=df_benchmarks,
    left_on=['Date', 'Manager'],
    right_on=['Date', 'ModelCode'], how='inner'
)

df_benchmarks_ieu = df_returns[df_returns['Manager'].isin(['MSCI.ACWI.EX.AUS_Index'])]
df_benchmarks_ieu = df_benchmarks_ieu.rename(columns={'Manager': 'IEu Name', '1_r': 'IEu 1_Benchmark'})

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
# df_asset_allocations['Date'] = df_asset_allocations['Date'] + pd.offsets.MonthEnd(1)

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

# Replaces the MSCI ACWI ex Aus Net TR 40% Hedged with the MSCI ACWI ex Aus Net TR unhedged to remove FX effects from Style
df_main_ieu = pd.merge(
    left=df_main,
    right=df_benchmarks_ieu,
    left_on=['Date'],
    right_on=['Date'],
    how='inner'
)

# ieu_sector = []
# for i in range(0, len(df_main_ieu)):
#     if df_main_ieu['Sector'][i] == 'IE':
#         ieu_sector.append(df_main_ieu['IEu Name'][i])
#     else:
#         ieu_sector.append(df_main_ieu['Sector'][i])
# df_main_ieu['Sector'] = ieu_sector

ieu_benchmark = []
for i in range(0, len(df_main_ieu)):
    if df_main_ieu['Sector'][i] == 'IE':
        ieu_benchmark.append(df_main_ieu['IEu 1_Benchmark'][i])
    else:
        ieu_benchmark.append(df_main_ieu['bmk_1_r_y'][i])
df_main_ieu['bmk_1_r_y'] = ieu_benchmark

# Calculates Style
# df_style = df_main.groupby(['Date', 'Sector', 'Style']).apply(style_calculate).reset_index(drop=False)
df_style = df_main_ieu.groupby(['Date', 'Sector', 'Style']).apply(style_calculate).reset_index(drop=False)
df_style['pure_1_ac'] = df_style['manager_1_r'] - df_style['style_1_r']
df_style['style_1_ac'] = df_style['style_1_r'] - df_style['passive_1_r']
df_style['total_1_ac'] = df_style['manager_1_r'] - df_style['passive_1_r']

df_sector_style = df_style.groupby(['Date', 'Sector']).sum().reset_index(drop=False)

# Removes unnecessary bond sectors
df_sector_style = df_sector_style[~df_sector_style['Sector'].isin(['IFI'])]

"""
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
"""

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

# Handles alternatives
df_alternative_returns = df_returns_benchmarks[df_returns_benchmarks['Manager'].isin(['LPE', 'PE', 'OA', 'DA', 'OO'])]
df_alternative_returns['manager_1_r'] = df_alternative_returns['1_r']
df_alternative_returns['style_1_r'] = df_alternative_returns['bmk_1_r']
df_alternative_returns['passive_1_r'] = df_alternative_returns['bmk_1_r']
df_alternative_returns['Sector'] = df_alternative_returns['Manager']
df_alternative_returns['Manager_x'] = df_alternative_returns['Manager']
df_alternative_returns['Manager_y'] = df_alternative_returns['Manager']
df_alternative_returns = df_alternative_returns.drop(['ModelCode', 'Style'], axis=1)

df_test3 = pd.concat([df_test3, df_alternative_returns], sort=False)

# Merge test3 with asset allocations
# Multiply returns with asset allocations
assetclass_to_sector_dict = {
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

sector_to_name_dict = {
    'AE': 'Australian Equity',
    'IE': 'International Equity',
    'ALP': 'ALP',
    'DP': 'Australian Property',
    'ILP': 'International Property',
    'BO': 'Bonds',
    'AR': 'Absolute Return',
    'CO': 'Commodities',
    'AC': 'Managed Cash',
    'LPE': 'Legacy PE',
    'PE': 'Private Equity',
    'OA': 'Opportunistic Alts',
    'DA': 'Defensive Alts',
    'OO': 'Options Overlay',
    'FW': 'FW',
    'TO': 'Total'
}

df_asset_allocations['Sector'] = [
    assetclass_to_sector_dict[df_asset_allocations['Asset Class'][i]]
    for i in range(0, len(df_asset_allocations))
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

# Reordering
df_final['Manager'] = [
    sector_to_name_dict[df_final['Sector'][i]]
    for i in range(0, len(df_final))
]

strategy_to_order_dict = {
    'High Growth': 0,
    'Balanced Growth': 1,
    'Balanced': 2,
    'Conservative': 3,
    'Growth': 4,
    'Employer Reserve': 5
}

df_final['strategy_sort'] = df_final.Strategy.map(strategy_to_order_dict)

sector_to_order_dict = {
    'AE': 0,
    'IE': 1,
    'ALP': 2,
    'DP': 3,
    'ILP': 4,
    'BO': 5,
    'AR': 6,
    'CO': 7,
    'AC': 8,
    'PE': 9,
    'OA': 10,
    'DA': 11,
    'LPE': 12,
    'OO': 13
}
df_final['sector_sort'] = df_final.Sector.map(sector_to_order_dict)

df_final = df_final.sort_values(['strategy_sort', 'sector_sort'])

manager_to_order_dict = {
    'Australian Equity': 0,
    'International Equity': 1,
    'Australian Listed Property': 2,
    'Australian Property': 3,
    'International Property': 4,
    'Bonds': 5,
    'Absolute Return': 6,
    'Commodities': 7,
    'Managed Cash': 8,
    'Private Equity': 9,
    'Opportunistic Alts': 10,
    'Defensive Alts': 11,
    'Legacy PE': 12,
    'Options Overlay': 13
}

# Formatting
df_final_format = df_final.copy()
for column in df_final_format.columns:
    try:
        df_final_format[column] / 1
        df_final_format[column] = (df_final_format[column] * 100).round(2)
    except TypeError:
        pass


def pivot_table(df, var):
    columns_list = ['Strategy', 'Manager', var]
    df = df[columns_list]
    df = df.pivot(index='Manager', columns='Strategy', values=var)
    df = df[asset_allocations]
    df.loc['Total'] = df.sum()
    df = df.reset_index(drop=False)
    df['sector_sort'] = df.Manager.map(manager_to_order_dict)
    df = df.sort_values(['sector_sort'])
    df = df.drop('sector_sort', axis=1)
    df = df.reset_index(drop=True)
    df.columns = latex_column_names
    return df


df_pa = pivot_table(df_final_format, r_pure_active)
df_sa = pivot_table(df_final_format, r_style_active)
df_ra = pivot_table(df_final_format, r_residual)
df_ta = pivot_table(df_final_format, r_total_active)

with open(output_directory + 'pa.tex', 'w') as tf:
    tf.write(df_pa.to_latex(index=False).replace('NaN', '').replace('-0.00', '0.00'))

with open(output_directory + 'sa.tex', 'w') as tf:
    tf.write(df_sa.to_latex(index=False).replace('NaN', '').replace('-0.00', '0.00'))

with open(output_directory + 'ra.tex', 'w') as tf:
    tf.write(df_ra.to_latex(index=False).replace('NaN', '').replace('-0.00', '0.00'))

with open(output_directory + 'ta.tex', 'w') as tf:
    tf.write(df_ta.to_latex(index=False).replace('NaN', '').replace('-0.00', '0.00'))


# FX Extension
df_fx1 = df_returns_benchmarks[df_returns_benchmarks['Manager'].isin(['IEu', 'IECurrencyOverlay_IE', 'IE'])]
df_fx1 = df_fx1.drop('Sector', axis=1)
df_fx1['Asset Class'] = 'International Equity'
df_fx1['1_er'] = df_fx1['1_r'] - df_fx1['bmk_1_r']

df_fx2 = pd.merge(
    left=df_fx1,
    right=df_asset_allocations,
    left_on=['Asset Class', 'Date'],
    right_on=['Asset Class', 'Date'],
    how='inner'
)
df_fx2['1_ac'] = df_fx2['1_er'] * df_fx2['Portfolio']


def fx_link(data):
    d = dict()
    d[r_portfolio] = (np.prod(1 + data['1_r']) - 1)
    d[r_benchmark] = (np.prod(1 + data['bmk_1_r']) - 1)
    d[w_portfolio] = np.average(data['Portfolio'])
    return pd.Series(d)


df_fx3 = df_fx2.groupby(['Strategy', 'Manager']).apply(fx_link)
df_fx3['Check'] = df_fx3[r_portfolio] - df_fx3[r_benchmark]
df_fx3[r_active_contribution] = df_fx3[w_portfolio] * (df_fx3[r_portfolio] - df_fx3[r_benchmark])
df_fx3 = df_fx3.reset_index(drop=False)

df_fx_format = df_fx3
df_fx_format[r_active_contribution] = df_fx_format[r_active_contribution]*100
df_fx_format = df_fx_format[~df_fx_format['Manager'].isin(['IECurrencyOverlay_IE'])].reset_index(drop=True)
fx_name_dict = {'IE': 'Intl Equity (Hedged)', 'IEu': 'Intl Equity (Unhedged)', 'IECurrencyOverlay_IE': 'FX Overlay'}
df_fx_format['Manager'] = [fx_name_dict[df_fx_format['Manager'][i]] for i in range(0, len(df_fx_format))]

df_fx_ac = pivot_table(df_fx_format, r_active_contribution)
df_fx_ac = df_fx_ac[~df_fx_ac['Asset Class'].isin(['Total'])]
df_fx_ac = df_fx_ac.set_index('Asset Class')
df_fx_ac.loc['FX Overlay'] = df_fx_ac.loc['Intl Equity (Hedged)'] - df_fx_ac.loc['Intl Equity (Unhedged)']
df_fx_ac = df_fx_ac.reindex(['Intl Equity (Unhedged)', 'FX Overlay', 'Intl Equity (Hedged)'])
df_fx_ac = df_fx_ac.round(2)
df_fx_ac = df_fx_ac.reset_index(drop=False)

with open(output_directory + 'fa.tex', 'w') as tf:
    tf.write(df_fx_ac.to_latex(index=False).replace('NaN', '').replace('-0.00', '0.00'))


# CREATE CHARTS
treegreen = (75/256, 120/256, 56/256)
middlegreen = (141/256, 177/256, 66/256)
lightgreen = (175/256, 215/256, 145/256)
darkred = (256/256, 0, 0)
middlered = (255/256, 102/256, 102/256)
lightred = (255/256, 204/256, 204/256)

green_to_red_dict = {treegreen: darkred, middlegreen: middlered, lightgreen: lightred}

name_tuple_dict = {
    'pa': (df_pa, treegreen, 'Manager Active Contribution (%)', 'pa_chart'),
    'sa': (df_sa, middlegreen, 'Style Active Contribution (%)', 'sa_chart'),
    'fa': (df_fx_ac, treegreen, 'Active Contribution (%)', 'fa_chart'),
    'ra': (df_ra, treegreen, 'Residual Active Contribution (%)', 'ra_chart')
}

for name, tuple in name_tuple_dict.items():
    fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(12.80, 7.20))
    strategy_to_axes_dict = {
        'High Growth': axes[0, 0],
        "Bal' Growth": axes[0, 1],
        'Balanced': axes[0, 2],
        'Conservative': axes[1, 0],
        'Growth': axes[1, 1],
        "Emp' Reserve": axes[1, 2]
    }

    for strategy, axes in strategy_to_axes_dict.items():
        df = tuple[0][['Asset Class', strategy]].reset_index(drop=True)
        df = df[~df['Asset Class'].isin(['Total', 'Intl Equity (Hedged)'])].reset_index(drop=True)
        df = df.set_index('Asset Class')
        df['positive'] = df[strategy] > 0
        if name == 'fa':
            df[strategy].plot.bar(ax=axes, color=df.positive.map({True: tuple[1], False: green_to_red_dict[tuple[1]]}), width=0.3)
        else:
            df[strategy].plot.bar(ax=axes, color=df.positive.map({True: tuple[1], False: green_to_red_dict[tuple[1]]}))
        axes.set_title(strategy)
        axes.set_xlabel('')
        axes.set_ylabel(tuple[2])
        axes.axhline(y=0, linestyle=':', linewidth=1, color='k', )
        fig.tight_layout()
        plt.show()

    fig.savefig('U:/CIO/#Attribution/charts/' + tuple[3] + '.png')

