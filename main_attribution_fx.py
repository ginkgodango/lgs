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
periods = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1

# If hedge ratio changes
hedge_ratio_change_date = datetime.datetime(2018, 10, 4)
hedge_ratio_periods_old = ((hedge_ratio_change_date.year - start_date.year) * 12 + (hedge_ratio_change_date.month - start_date.month))/ periods
hedge_ratio_periods_new = ((end_date.year - hedge_ratio_change_date.year) * 12 + (end_date.month - hedge_ratio_change_date.month) + 1)/ periods
hedge_ratio_old = 0.3
hedge_ratio_new = 0.4
hedge_ratio_adj = hedge_ratio_old * hedge_ratio_periods_old + hedge_ratio_new * hedge_ratio_periods_new

fx_allocation_dict = {'IEu': 1, 'IECurrencyOverlay_IE': hedge_ratio_new}

input_directory = 'U:/CIO/#Investment_Report/Data/input/'
output_directory = 'U:/CIO/#Attribution/tables/fx/'

table_filename = 'link_2019-05-31.csv'
returns_filename = 'returns_2019-06-30.csv'
market_values_filename = 'market_values_2019-06-30.csv'
asset_allocations_filename = 'asset_allocations_2019-06-30.csv'

latex_summary1_column_names = ['Returns', 'High Growth', "Bal' Growth", 'Balanced', 'Conservative', 'Growth', "Emp' Reserve"]
latex_summary2_column_names = ['Attribution', 'High Growth', "Bal' Growth", 'Balanced', 'Conservative', 'Growth', "Emp' Reserve"]
latex_column_names = ['Asset Class', 'High Growth', "Bal' Growth", 'Balanced', 'Conservative', 'Growth', "Emp' Reserve"]

# Creates variable names for linked table
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
df_benchmarks = df_benchmarks[['Date', 'Associated Benchmark', '1_r', 'ModelCode']]
df_benchmarks.columns = ['Date', 'Benchmark Name', 'bmk_1_r', 'ModelCode']

df_returns_benchmarks = pd.merge(left=df_returns, right=df_benchmarks, left_on=['Date', 'Manager'],
                                 right_on=['Date', 'ModelCode'], how='inner')

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

#MAKE THE HIGH GROWTH, BALANCED GROWTH, ... SPLIT
"""
Test for High Growth
April 2019
manager/sector weight * asset/total weight
"""

strategy_to_modelcode_dict = {
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

strategy_to_name_dict = {
    'Australian Equity': 'Australian Equity',
    'International Equity': 'International Equity',
    'Australian Listed Property': 'ALP',
    'Property': 'Australian Property',
    'Global Property': 'International Property',
    'Bonds': 'Bonds',
    'Absolute Return': 'Absolute Return',
    'Commodities': 'Commodities',
    'Cash': 'Managed Cash',
    'Legacy Private Equity': 'LPE',
    'Private Equity': 'Private Equity',
    'Opportunistic Alternatives': 'Opportunistic Alts',
    'Defensive Alternatives': 'Defensive Alts',
    'Option Overlay': 'OO',
    'Forwards': 'FW',
    'Total': 'Total'
}

modelcode_to_name_dict = {
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

df_asset_allocations['ModelCode'] = [
    strategy_to_modelcode_dict[df_asset_allocations['Asset Class'][i]]
    for i in range(0, len(df_asset_allocations))
]

filter_list = ['ALP', 'FW', 'TO']
df_asset_allocations = df_asset_allocations[~df_asset_allocations['ModelCode'].isin(filter_list)].reset_index(drop=True)


# Begin the FX Feature
select_columns = ['Manager', 'Date', 'Benchmark Name', '1_r', 'bmk_1_r']

df_main = df_returns_benchmarks[select_columns]

fx_dict = {'IE': {
    'IEu': {'join_key': 'FX', 'allocation': 1},
    'IECurrencyOverlay_IE': {'join_key': 'FX', 'allocation': hedge_ratio_new}
}}

df_fx = df_main[df_main['Manager'].isin(fx_dict['IE'])].reset_index(drop=True)

# Join the IE allocation to IEu and IECurrencyOverlay_IE
fx_join_keys = []
for i in range(0,len(df_fx)):
    if df_fx['Manager'][i] in fx_dict['IE']:
        fx_join_keys.append('FX')
    else:
        fx_join_keys.append(np.nan)
df_fx['join_key'] = fx_join_keys

asset_allocations_join_keys = []
for i in range(0, len(df_asset_allocations)):
    if df_asset_allocations['ModelCode'][i] == 'IE':
        asset_allocations_join_keys.append('FX')
    else:
        asset_allocations_join_keys.append(np.nan)
df_asset_allocations['join_key'] = asset_allocations_join_keys

select_allocations_columns = ['Date', 'Strategy', 'Asset Class', 'ModelCode', 'Portfolio', 'Benchmark', 'join_key']

df_asset_allocations = df_asset_allocations[select_allocations_columns]

df_fx = pd.merge(left=df_fx, right=df_asset_allocations, left_on=['Date', 'join_key'], right_on=['Date', 'join_key'])

# Converts returns to contributions relative to strategies
df_fx['1_r'] = df_fx['1_r'] * df_fx['Portfolio']
df_fx['bmk_1_r'] = df_fx['bmk_1_r'] * df_fx['Portfolio']

# Converts the FXOverlay and benchmark into 100% hedged returns
fx_returns = []
fx_benchmarks = []
fx_portfolio_sub = []
fx_benchmark_sub = []
for i in range(0,len(df_fx)):
    if df_fx['Manager'][i] == 'IECurrencyOverlay_IE':
        if df_fx['Date'][i] < hedge_ratio_change_date:
            fx_returns.append(df_fx['1_r'][i]/hedge_ratio_old)
            fx_benchmarks.append(df_fx['bmk_1_r'][i]/hedge_ratio_new)
            fx_portfolio_sub.append(hedge_ratio_old)
            fx_benchmark_sub.append(hedge_ratio_new)
        else:
            fx_returns.append(df_fx['1_r'][i] / hedge_ratio_new)
            fx_benchmarks.append(df_fx['bmk_1_r'][i] / hedge_ratio_new)
            fx_portfolio_sub.append(hedge_ratio_new)
            fx_benchmark_sub.append(hedge_ratio_new)
    else:
        fx_returns.append(df_fx['1_r'][i])
        fx_benchmarks.append(df_fx['bmk_1_r'][i])
        fx_portfolio_sub.append(1)
        fx_benchmark_sub.append(1)

df_fx['1_r'] = fx_returns
df_fx['bmk_1_r'] = fx_benchmarks
df_fx['Portfolio_sub'] = fx_portfolio_sub
df_fx['Benchmark_sub'] = fx_benchmark_sub

# Adds portfolio and benchmark weights
# df_fx['Portfolio_sub'] = [fx_dict['IE'][df_fx['Manager'][i]]['allocation'] for i in range(0, len(df_fx))]
# df_fx['Benchmark_sub'] = [fx_dict['IE'][df_fx['Manager'][i]]['allocation'] for i in range(0, len(df_fx))]

# Removes previous level effects
# df_fx['Portfolio_sub'] = df_fx['Portfolio_sub']*df_fx['Benchmark']/df_fx['Portfolio']

df_hedged = df_main[df_main['Manager'].isin(['IE'])]
df_hedged = df_hedged.rename(columns={
    'Manager': 'Asset Class',
    '1_r': 'R',
    'Benchmark Name': 'Asset Class Benchmark',
    'bmk_1_r': 'R_bar'
})

# Start Testing
def build_ie_aggregate(data):
    d = dict()
    d['Asset Class'] = 'IE',
    d['R'] = np.dot(data['1_r'], data['Portfolio_sub']),
    d['Asset Class Benchmark'] = 'IESectorBmk_Index',
    d['R_bar'] = np.dot(data['bmk_1_r'], data['Benchmark_sub'])
    return pd.Series(d)


df_ie = df_fx.groupby(['Strategy', 'Date']).apply(build_ie_aggregate).reset_index(drop=False)
df_ie['R'] = [df_ie['R'][i][0] for i in range(0,len(df_ie))]
df_ie = df_ie[['Strategy', 'Date', 'R', 'R_bar']]

df_fx_hedged = pd.merge(left=df_fx, right=df_ie, left_on=['Strategy', 'Date'], right_on=['Strategy', 'Date'], how='inner')
# End Testing

# df_fx_hedged = pd.merge(left=df_fx, right=df_hedged, left_on='Date', right_on='Date', how='inner')

# df_fx_hedged['R_bar'] = df_fx_hedged['R_bar'] * df_fx_hedged['Portfolio']

# BHB Special Case
df_fx_hedged['R_bar'] = 0
# End BHB Special Case

df_fx_hedged['AA'] = (df_fx_hedged['Portfolio_sub'] - df_fx_hedged['Benchmark_sub']) * (df_fx_hedged['bmk_1_r'] - df_fx_hedged['R_bar'])

df_fx_hedged['SS'] = df_fx_hedged['Benchmark_sub'] * (df_fx_hedged['1_r'] - df_fx_hedged['bmk_1_r'])

df_fx_hedged['In'] = (df_fx_hedged['Portfolio_sub'] - df_fx_hedged['Benchmark_sub']) * (df_fx_hedged['1_r'] - df_fx_hedged['bmk_1_r'])

df_fx_hedged['Total'] = df_fx_hedged['AA'] + df_fx_hedged['SS'] + df_fx_hedged['In']

def weighted_average(data):
    d = dict()
    d['Manager'] = 'IEh'
    d['Portfolio_sub'] = np.sum(data['Portfolio_sub'])
    d['Benchmark_sub'] = np.sum(data['Benchmark_sub'])
    d['1_r'] = np.dot(data['1_r'], data['Portfolio_sub'])
    d['bmk_1_r'] = np.dot(data['bmk_1_r'], data['Benchmark_sub'])
    d['AA'] = np.sum(data['AA'])
    d['SS'] = np.sum(data['SS'])
    d['In'] = np.sum(data['In'])
    d['Total'] = np.sum(data['Total'])
    return pd.Series(d)


df_fx_hedged_total = df_fx_hedged.groupby(['Date', 'Strategy']).apply(weighted_average).reset_index(drop=False)

df_fx_hedged = pd.concat([df_fx_hedged, df_fx_hedged_total], sort=False).reset_index(drop=True)

# Calculates excess return
df_fx_hedged['1_er'] = df_fx_hedged['1_r'] - df_fx_hedged['bmk_1_r']

#Chain Linking Mencheros
def link2(data):
    d = dict()
    # d[market_value] = np.average(data['Market Value'])
    d[r_portfolio] = (np.prod(1 + data['1_r']) - 1)
    d[r_benchmark] = (np.prod(1 + data['bmk_1_r']) - 1)
    d[r_diff] = np.sum(data['1_r'] - data['bmk_1_r'])
    d[r_diff_sq] = np.sum((data['1_r'] - data['bmk_1_r'])**2)
    return pd.Series(d)


df_linked = df_fx_hedged.copy()

df_linked = df_linked.groupby(['Strategy', 'Manager']).apply(link2)

# JPM Version
df_linked[r_excess] = df_linked[r_portfolio] - df_linked[r_benchmark]
df_linked = df_linked.reset_index(drop=False)

df_linked_test = df_linked[df_linked['Manager'].isin(['IEh'])]
# A good
df_linked_test['A'] = (
        ((df_linked_test[r_portfolio] - df_linked_test[r_benchmark])/periods)
        / ((1 + df_linked_test[r_portfolio])**(1/periods) - (1 + df_linked_test[r_benchmark])**(1/periods))
)

df_linked_test['C'] = (df_linked_test[r_portfolio] - df_linked_test[r_benchmark] - df_linked_test['A']*df_linked_test[r_diff]) / df_linked_test[r_diff_sq]
# Needs to fix
df_linked_test['alpha'] = df_linked_test['C']*df_linked_test[r_excess]
df_linked_test['beta'] = df_linked_test['A'] + df_linked_test['alpha']

df_linked_coef = df_linked_test[['Strategy', 'A', 'C']]

df_hedged_scale = pd.merge(
    df_fx_hedged,
    df_linked_coef,
    left_on=['Strategy'],
    right_on=['Strategy'],
    how='inner'
)

# Calculates the scaling factor
df_hedged_scale['beta'] = df_hedged_scale['A'] + df_hedged_scale['C']*(df_hedged_scale['R']-df_hedged_scale['R_bar'])

# Multiplies AA, SS, In, and Total by the scaling factor
df_hedged_scale['AA'] = df_hedged_scale['beta'] * df_hedged_scale['AA']
df_hedged_scale['SS'] = df_hedged_scale['beta'] * df_hedged_scale['SS']
df_hedged_scale['In'] = df_hedged_scale['beta'] * df_hedged_scale['In']
df_hedged_scale['Total'] = df_hedged_scale['beta'] * df_hedged_scale['Total']

# Sum the total rows
df_hedged_scale_1 = df_hedged_scale[~df_hedged_scale['Manager'].isin(['IEh'])]

def total_sum_scale(data):
    d = dict()
    #d['Market Value'] = np.sum(data['Market Value'])
    d['Portfolio_sub'] = np.sum(data['Portfolio_sub'])
    d['Benchmark_sub'] = np.sum(data['Benchmark_sub'])
    d['1_r'] = np.dot(data['1_r'], data['Portfolio_sub'])
    d['bmk_1_r'] = np.dot(data['bmk_1_r'], data['Benchmark_sub'])
    #d['Active Contribution'] = np.sum(data['Active Contribution'])
    d['AA'] = np.sum(data['AA'])
    d['SS'] = np.sum(data['SS'])
    d['In'] = np.sum(data['In'])
    return pd.Series(d)


df_hedged_scale_2 = df_hedged_scale_1.groupby(['Date', 'Strategy']).apply(total_sum_scale).reset_index(drop=False)
df_hedged_scale_2['Manager'] = 'IEh'

df_hedged_scale_2['Total'] = df_hedged_scale_2['AA'] + df_hedged_scale_2['SS'] + df_hedged_scale_2['In']

df_attribution_fx = pd.concat([df_hedged_scale_1, df_hedged_scale_2], sort=True)

# Rearrange the columns
column_order_fx = [
    'Date',
    'Strategy',
    'Manager',
    '1_r',
    'bmk_1_r',
    '1_er',
    'R',
    'R_bar',
    'Portfolio_sub',
    'Benchmark_sub',
    'AA',
    'SS',
    'In',
    'Total'
]

df_attribution_fx = df_attribution_fx[column_order_fx]

def final(data):
    d = dict()
    #d[market_value] = np.average(data['Market Value'])
    d[r_portfolio] = (np.prod(1 + data['1_r']) - 1)
    d[r_benchmark] = (np.prod(1 + data['bmk_1_r']) - 1)
    d[w_portfolio] = np.average(data['Portfolio_sub'])
    d[w_benchmark] = np.average(data['Benchmark_sub'])
    d[AA] = np.sum(data['AA'])
    d[SS] = np.sum(data['SS'])
    d[interaction] = np.sum(data['In'])
    d[total_effect] = np.sum(data['Total'])
    return pd.Series(d)


df_attribution_fx = df_attribution_fx.groupby(['Strategy', 'Manager']).apply(final).reset_index(drop=False)

# Fix active return and and r_active contribution
df_attribution_fx[r_excess] = df_attribution_fx[r_portfolio] - df_attribution_fx[r_benchmark]
df_attribution_fx[r_active_contribution] = df_attribution_fx[w_portfolio] * df_attribution_fx[r_excess]


# Fix IEh active contribution
def final_ieh_ac(data):
    d = dict()
    d[r_active_contribution] = np.dot(data[w_portfolio], data[r_excess])
    return pd.Series(d)

df_attribution_fx_ieh = df_attribution_fx[~df_attribution_fx['Manager'].isin(['IEh'])]
df_attribution_fx_ieh = df_attribution_fx_ieh.groupby(['Strategy']).apply(final_ieh_ac).to_dict()

fx_ac = []
for i in range(0,len(df_attribution_fx)):
    if df_attribution_fx['Manager'][i] == 'IEh':
        fx_ac.append(df_attribution_fx_ieh[r_active_contribution][df_attribution_fx['Strategy'][i]])
    else:
        fx_ac.append(df_attribution_fx[r_active_contribution][i])
df_attribution_fx[r_active_contribution] = fx_ac

# Calculates residual and total
df_attribution_fx[residual] = df_attribution_fx[r_active_contribution] - df_attribution_fx[total_effect]
df_attribution_fx[total] = df_attribution_fx[total_effect] + df_attribution_fx[residual]


#Output Below Here
output_columns = ['Strategy', 'Manager', r_portfolio, r_benchmark, r_active_contribution, AA, SS, interaction, residual, total_effect, total]

df_output = df_attribution_fx[output_columns]

percentage_column = [r_portfolio, r_benchmark, r_active_contribution, AA, SS, interaction, residual, total_effect, total]

df_output[percentage_column] = (df_output[percentage_column]*100).round(2)

# df_output = df_output.reset_index(drop=True)
#
# fx_sort = {
#     'IEu': 0,
#     'IECurrencyOverlay_IE': 1,
#     'IEh': 2
# }
#
# df_output['fx_sort'] = [fx_sort[df_output['Manager'][i]] for i in range(0, len(df_output))]
#
# df_output = df_output.sort_values(['Strategy', 'fx_sort']).reset_index(drop=True)

manager_to_name_dict = {
    'IEu': '1. Int\'l Equity (Unhedged)',
    'IECurrencyOverlay_IE': '2. FX Overlay',
    'IEh': '3. Int\'l Equity (Hedged)'
}

df_output['Manager'] = [manager_to_name_dict[df_output['Manager'][i]] for i in range(0, len(df_output))]


def pivot_table(df, var):
    columns_list = ['Strategy', 'Manager', var]
    df = df[columns_list]
    df = df.pivot(index='Manager', columns='Strategy', values=var)
    df = df[asset_allocations]
    df = df.reset_index(drop=False)
    df = df.reset_index(drop=True)
    df.columns = latex_column_names
    return df


df_ac = pivot_table(df_output, r_active_contribution)
df_aa = pivot_table(df_output, AA)
df_ss = pivot_table(df_output, SS)
df_in = pivot_table(df_output, interaction)
df_re = pivot_table(df_output, residual)
df_te = pivot_table(df_output, total_effect)
df_to = pivot_table(df_output, total)

with open(output_directory + 'fx_ac.tex', 'w') as tf:
    tf.write(df_ac.to_latex(index=False).replace('NaN', ''))

with open(output_directory + 'fx_aa.tex', 'w') as tf:
    tf.write(df_aa.to_latex(index=False).replace('NaN', ''))

with open(output_directory + 'fx_ss.tex', 'w') as tf:
    tf.write(df_ss.to_latex(index=False).replace('NaN', ''))

with open(output_directory + 'fx_in.tex', 'w') as tf:
    tf.write(df_in.to_latex(index=False).replace('NaN', ''))

with open(output_directory + 'fx_re.tex', 'w') as tf:
    tf.write(df_re.to_latex(index=False).replace('NaN', ''))

with open(output_directory + 'fx_to.tex', 'w') as tf:
    tf.write(df_to.to_latex(index=False).replace('NaN', ''))


# #Output Below Here
# output_columns = ['Strategy', 'Manager', 'Date', '1_r', 'bmk_1_r', '1_er', 'AA', 'SS', 'In', 'Total']
#
# df_output = df_fx_hedged[output_columns]
#
# percentage_column = ['1_r', 'bmk_1_r', '1_er', 'AA', 'SS', 'In', 'Total']
#
# df_output[percentage_column] = (df_output[percentage_column]*100).round(2)
#
# manager_to_name_dict = {
#     'IEu': '1. Int\'l Equity (Unhedged)',
#     'IECurrencyOverlay_IE': '2. FX Overlay',
#     'IEh': '3. Int\'l Equity (Hedged)'
# }
#
# df_output['Manager'] = [manager_to_name_dict[df_output['Manager'][i]] for i in range(0, len(df_output))]
#
#
# def pivot_table(df, var):
#     columns_list = ['Strategy', 'Manager', var]
#     df = df[columns_list]
#     df = df.pivot(index='Manager', columns='Strategy', values=var)
#     df = df[asset_allocations]
#     df = df.reset_index(drop=False)
#     #df['asset_class_sort'] = df.Manager.map(asset_class_sort)
#     #df = df.sort_values(['asset_class_sort'])
#     #df = df.drop('asset_class_sort', axis=1)
#     df = df.reset_index(drop=True)
#     #df['Manager'] = [modelcode_to_name_dict[df['Manager'][i]] for i in range(0, len(df))]
#     df.columns = latex_column_names
#     return df
#
#
# df_ac = pivot_table(df_output, '1_er')
# df_aa = pivot_table(df_output, 'AA')
# df_ss = pivot_table(df_output, 'SS')
# df_in = pivot_table(df_output, 'In')
# df_to = pivot_table(df_output, 'Total')
#
# with open(output_directory + 'fx_ac.tex', 'w') as tf:
#     tf.write(df_ac.to_latex(index=False).replace('NaN', ''))
#
# with open(output_directory + 'fx_aa.tex', 'w') as tf:
#     tf.write(df_aa.to_latex(index=False).replace('NaN', ''))
#
# with open(output_directory + 'fx_ss.tex', 'w') as tf:
#     tf.write(df_ss.to_latex(index=False).replace('NaN', ''))
#
# with open(output_directory + 'fx_in.tex', 'w') as tf:
#     tf.write(df_in.to_latex(index=False).replace('NaN', ''))
#
# with open(output_directory + 'fx_to.tex', 'w') as tf:
#     tf.write(df_to.to_latex(index=False).replace('NaN', ''))
