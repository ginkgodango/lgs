"""
Attribution
"""
import datetime
import pandas as pd
import numpy as np
import win32com.client
import matplotlib
import matplotlib.pyplot as plt
import attribution.extraction
from dateutil.relativedelta import relativedelta

start_date = datetime.datetime(2018, 4, 30)
end_date = datetime.datetime(2019, 3, 31)

directory = 'D:/automation/final/attribution/2019/04/'
output_directory = 'D:/automation/final/attribution/tables/'
table_filename = 'table_NOF_201904.csv'
returns_filename = 'returns_NOF_201904v2.csv'
market_values_filename = 'market_values_NOF_201904.csv'
asset_allocations_filename = 'asset_allocations_201904.csv'
performance_report_filepath = 'D:/automation/final/investment/2019/04/LGSS Preliminary Performance April 2019_Addkeys.xlsx'

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

# Loads table
df_table = attribution.extraction.load_table(directory + table_filename)

# Loads returns
df_returns = attribution.extraction.load_returns(directory + returns_filename)

# Reshapes returns dataframe from wide to long
df_returns = df_returns.transpose().reset_index(drop=False).rename(columns={'index': 'Manager'})
df_returns = pd.melt(df_returns, id_vars=['Manager'], value_name='1_r')

# Selects returns for this month or within a date_range
# df_returns = df_returns[df_returns['Date'] == df_returns['Date'].max()].reset_index(drop=True)
df_returns = df_returns[(df_returns['Date'] >= start_date) & (df_returns['Date'] <= end_date)].reset_index(drop=True)

df_benchmarks = pd.merge(left=df_returns, right=df_table, left_on=['Manager'], right_on=['Associated Benchmark'],
                         how='inner')
df_benchmarks = df_benchmarks[['Date', 'Associated Benchmark', '1_r', 'ModelCode']]
df_benchmarks.columns = ['Date', 'Benchmark Name', 'bmk_1_r', 'ModelCode']

df_returns_benchmarks = pd.merge(left=df_returns, right=df_benchmarks, left_on=['Date', 'Manager'],
                                 right_on=['Date', 'ModelCode'], how='inner')

# Loads market values
df_market_values = attribution.extraction.load_market_values(directory + market_values_filename)

# Reshapes market values dataframe from wide to long
df_market_values = df_market_values.transpose().reset_index(drop=False).rename(columns={'index': 'Manager'})
df_market_values = pd.melt(df_market_values, id_vars=['Manager'], value_name='Market Value')

# Selects market values for this month
df_market_values = df_market_values[df_market_values['Date'] == df_market_values['Date'].max()].reset_index(drop=True)

# Joins 1 month returns with market values
df_main = pd.merge(df_returns_benchmarks, df_market_values, how='outer', on=['Manager', 'Date'])

# Loads strategy asset allocations
asset_allocations = ['High Growth', 'Balanced Growth', 'Balanced', 'Conservative', 'Growth', 'Employer Reserve']

df_asset_allocations = pd.read_csv(
    directory + asset_allocations_filename,
    parse_dates=['Date'],
    infer_datetime_format=True,
    float_precision='round_trip'
)

# Converts to decimals
df_asset_allocations['Portfolio'] = df_asset_allocations['Portfolio']/100
df_asset_allocations['Benchmark'] = df_asset_allocations['Benchmark']/100

# Forwards the asset allocations by 1 month, which lags it 1 month relative to the returns and market values.
df_asset_allocations['Date'] = df_asset_allocations['Date'] + pd.offsets.MonthEnd(1)
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
# filter_strategy_list = ['Australian Listed Property', 'Option Overlay', 'Forwards']

df_asset_allocations = df_asset_allocations[~df_asset_allocations['ModelCode'].isin(filter_list)]

df_attribution = pd.merge(
    df_returns_benchmarks,
    df_asset_allocations,
    left_on=['Date', 'ModelCode'],
    right_on=['Date', 'ModelCode'],
    how='inner'
)

# Calculate and expresses AA, SS, and Interaction as decimals
df_attribution['AA'] = (df_attribution['Portfolio'] - df_attribution['Benchmark']) * df_attribution['bmk_1_r']

df_attribution['SS'] = (df_attribution['1_r'] - df_attribution['bmk_1_r']) * df_attribution['Benchmark']

df_attribution['Interaction'] = (df_attribution['1_r'] - df_attribution['bmk_1_r']) * (
            df_attribution['Portfolio'] - df_attribution['Benchmark'])

# Converts Market Value, AA, SS, and Interaction to float
column_float = ['Market Value', 'Portfolio', 'AA', 'SS', 'Interaction']
df_attribution[column_float] = df_attribution[column_float].astype(float)
df_attribution = df_attribution.sort_values(['Date', 'Strategy', 'Manager']).reset_index(drop=True)


def weighted_average(data):
    d = dict()
    d['Manager'] = 'TO'
    d['ModelCode'] = 'TO'
    d['Asset Class'] = 'Total'
    d['Market Value'] = np.sum(data['Market Value'])
    d['Portfolio'] = np.sum(data['Portfolio'])
    # d['Benchmark'] = np.sum(data['Benchmark'])
    d['1_r'] = np.average(data['1_r'], weights=data['Portfolio'])
    d['bmk_1_r'] = np.average(data['bmk_1_r'], weights=data['Benchmark'])
    d['AA'] = np.sum(data['AA'])
    d['SS'] = np.sum(data['SS'])
    d['Interaction'] = np.sum(data['Interaction'])
    return pd.Series(d)


df_attribution_total = df_attribution.groupby(['Date', 'Strategy']).apply(weighted_average).reset_index(drop=False)

df_attribution = pd.concat([df_attribution, df_attribution_total], sort=False)

df_attribution['Total'] = df_attribution['AA'] + df_attribution['SS'] + df_attribution['Interaction']
df_attribution['1_er'] = df_attribution['1_r'] - df_attribution['bmk_1_r']
df_attribution['Active Contribution'] = df_attribution['1_er'] * df_attribution['Portfolio']

# Menchero (2004) definition of AA
df_attribution_total_benchmark = df_attribution_total[['Date', 'Strategy', '1_r', 'bmk_1_r']]
df_attribution_total_benchmark = df_attribution_total_benchmark.rename(columns={'1_r': 'R', 'bmk_1_r': 'R_bar'})
df_attribution = pd.merge(
    df_attribution,
    df_attribution_total_benchmark,
    left_on=['Date', 'Strategy'],
    right_on=['Date', 'Strategy'],
    how='inner'
)
df_attribution['AA'] = (df_attribution['Portfolio'] - df_attribution['Benchmark']) * (df_attribution['bmk_1_r'] - df_attribution['R'])
df_attribution['Total'] = df_attribution['AA'] + df_attribution['SS'] + df_attribution['Interaction']

# Sorting
strategy_sort = {
    'High Growth': 0,
    'Balanced Growth': 1,
    'Balanced': 2,
    'Conservative': 3,
    'Growth': 4,
    'Employer Reserve': 5
}

df_attribution['strategy_sort'] = df_attribution.Strategy.map(strategy_sort)

asset_class_sort = {
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

df_attribution['asset_class_sort'] = df_attribution.Manager.map(asset_class_sort)

df_attribution = df_attribution.sort_values(['Date', 'strategy_sort', 'asset_class_sort'])

column_order = [
    'Date',
    'Strategy',
    'Manager',
    'Asset Class',
    'Market Value',
    '1_r',
    'bmk_1_r',
    '1_er',
    'R',
    'R_bar',
    'Portfolio',
    'Benchmark',
    'Active Contribution',
    'AA',
    'SS',
    'Interaction',
    'Total'
]

df_attribution = df_attribution[column_order]

df_attribution['Market Value'] = (df_attribution['Market Value'] / 1000000).round(2)

df_attribution = df_attribution.reset_index(drop=True)


# Test of chain linking
# Compound returns each period
def link(data):
    d = dict()
    d[market_value] = np.average(data['Market Value'])
    d[r_portfolio] = (np.prod(1 + data['1_r']) - 1)
    d[r_benchmark] = (np.prod(1 + data['bmk_1_r']) - 1)
    d[r_excess] = (np.prod(1 + data['1_er']) - 1)
    d[r_active_contribution] = (np.prod(1 + data['Active Contribution']) - 1)
    d[w_portfolio] = np.average(data['Portfolio'])
    d[w_benchmark] = np.average(data['Benchmark'])
    d[AA] = (np.prod(1 + data['AA']) - 1)
    d[SS] = (np.prod(1 + data['SS']) - 1)
    d[interaction] = (np.prod(1 + data['Interaction']) - 1)
    return pd.Series(d)

# JPM version, calculate 3 month averages than calculate AA, SS, Interaction FIX r_diff and r_diff_sq
def link2(data):
    d = dict()
    # d[market_value] = np.average(data['Market Value'])
    d[r_portfolio] = (np.prod(1 + data['1_r']) - 1)
    d[r_benchmark] = (np.prod(1 + data['bmk_1_r']) - 1)
    d[r_diff] = np.sum(data['1_r'] - data['bmk_1_r'])
    d[r_diff_sq] = np.sum((data['1_r'] - data['bmk_1_r'])**2)
    # d[r_excess] = (np.prod(1 + data['1_er']) - 1)
    # d[r_active_contribution] = (np.prod(1 + data['Active Contribution']) - 1)
    # d[w_portfolio] = np.average(data['Portfolio'])
    # d[w_benchmark] = np.average(data['Benchmark'])
    # d[AA] = (np.prod(1 + data['AA']) - 1)
    # d[SS] = (np.prod(1 + data['SS']) - 1)
    # d[interaction] = (np.prod(1 + data['Interaction']) - 1)
    return pd.Series(d)


df_linked = df_attribution.copy()

df_linked = df_linked.groupby(['Strategy', 'Asset Class', 'Manager']).apply(link2)

# JPM Version
df_linked[r_excess] = df_linked[r_portfolio] - df_linked[r_benchmark]
# df_linked[r_active_contribution] = df_linked[r_excess] * df_linked[w_portfolio]
# df_linked[AA] = (df_linked[w_portfolio] - df_linked[w_benchmark]) * df_linked[r_benchmark]
# df_linked[SS] = (df_linked[r_portfolio] - df_linked[r_benchmark]) * df_linked[w_benchmark]
# df_linked[interaction] = (df_linked[w_portfolio] - df_linked[w_benchmark]) * (df_linked[r_portfolio] - df_linked[r_benchmark])


# Summing effects
# df_linked[total_effect] = df_linked[AA] + df_linked[SS] + df_linked[interaction]
# df_linked[residual] = df_linked[r_active_contribution] - df_linked[total_effect]
# df_linked[total] = df_linked[total_effect] + df_linked[residual]
df_linked = df_linked.reset_index(drop=False)

df_linked_test = df_linked[df_linked['Manager'].isin(['TO'])]
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

df_attribution_scale = pd.merge(
    df_attribution,
    df_linked_coef,
    left_on=['Strategy'],
    right_on=['Strategy'],
    how='inner'
)


df_attribution_scale['beta'] = df_attribution_scale['A'] + df_attribution_scale['C']*(df_attribution_scale['R']-df_attribution_scale['R_bar'])

df_attribution_scale['AA'] = df_attribution_scale['beta'] * df_attribution_scale['AA']
df_attribution_scale['SS'] = df_attribution_scale['beta'] * df_attribution_scale['SS']
df_attribution_scale['Interaction'] = df_attribution_scale['beta'] * df_attribution_scale['Interaction']
df_attribution_scale['Total'] = df_attribution_scale['beta'] * df_attribution_scale['Total']

# test = df_attribution_scale.groupby(['Strategy', 'Manager']).sum()

# Sum the total rows
df_attribution_scale_1 = df_attribution_scale[~df_attribution_scale['Manager'].isin(['TO'])]

def total_sum_scale(data):
    d = dict()
    d['Market Value'] = np.sum(data['Market Value'])
    # d['Manager'] = 'TO',
    # d['Asset Class'] = 'Total',
    d['Portfolio'] = np.sum(data['Portfolio'])
    d['Benchmark'] = np.sum(data['Benchmark'])
    d['1_r'] = np.average(data['1_r'], weights=data['Portfolio'])
    d['bmk_1_r'] = np.average(data['bmk_1_r'], weights=data['Benchmark'])
    d['Active Contribution'] = np.sum(data['Active Contribution'])
    d['AA'] = np.sum(data['AA'])
    d['SS'] = np.sum(data['SS'])
    d['Interaction'] = np.sum(data['Interaction'])
    #d['Total'] = np.sum(data[total])
    return pd.Series(d)


df_attribution_scale_2 = df_attribution_scale_1.groupby(['Date', 'Strategy']).apply(total_sum_scale).reset_index(drop=False)
df_attribution_scale_2['Manager'] = 'TO'
df_attribution_scale_2['Asset Class'] = 'Total'

df_attribution_scale_2['Total'] = df_attribution_scale_2['AA'] + df_attribution_scale_2['SS'] + df_attribution_scale_2['Interaction']

df_attribution_test = pd.concat([df_attribution_scale_1, df_attribution_scale_2], sort=True)

df_attribution_test = df_attribution_test[column_order]


def final(data):
    d = dict()
    d[market_value] = np.average(data['Market Value'])
    d[r_portfolio] = (np.prod(1 + data['1_r']) - 1)
    d[r_benchmark] = (np.prod(1 + data['bmk_1_r']) - 1)
    d[w_portfolio] = np.average(data['Portfolio'])
    d[w_benchmark] = np.average(data['Benchmark'])
    d[AA] = np.sum(data['AA'])
    d[SS] = np.sum(data['SS'])
    d[interaction] = np.sum(data['Interaction'])
    d[total_effect] = np.sum(data['Total'])
    return pd.Series(d)


df_attribution_test = df_attribution_test.groupby(['Strategy', 'Manager', 'Asset Class']).apply(final).reset_index(drop=False)
# df_attribution_test['Manager'] = [df_attribution_test['Manager'][i][0] for i in range(0, len(df_attribution_test))]
# df_attribution_test['Asset Class'] = [df_attribution_test['Asset Class'][i][0] for i in range(0, len(df_attribution_test))]

# Fix active return and and r_active contribution
df_attribution_test[r_excess] = df_attribution_test[r_portfolio] - df_attribution_test[r_benchmark]
df_attribution_test[r_active_contribution] = df_attribution_test[w_portfolio] * df_attribution_test[r_excess]
df_attribution_test[residual] = df_attribution_test[r_active_contribution] - df_attribution_test[total_effect]
df_attribution_test[total] = df_attribution_test[total_effect] + df_attribution_test[residual]

df_linked = df_attribution_test

# Convert numbers to percentage
column_percentage = [
    r_portfolio,
    r_benchmark,
    r_excess,
    r_active_contribution,
    w_portfolio,
    w_benchmark,
    AA,
    SS,
    interaction,
    total_effect,
    residual,
    total
]

df_linked[column_percentage] = df_linked[column_percentage]*100

# ROUNDS NUMBERS
column_round = [
    market_value,
    r_portfolio,
    r_benchmark,
    r_excess,
    r_active_contribution,
    w_portfolio,
    w_benchmark,
    AA,
    SS,
    interaction,
    total_effect,
    residual,
    total
]

df_linked[column_round] = df_linked[column_round].astype(float).round(2)

# Makes summary tables
df_linked_total = df_linked[df_linked['Manager'].isin(['TO'])].reset_index(drop=True)

summary_column_list = [
    'Strategy',
    market_value,
    r_portfolio,
    r_benchmark,
    r_excess,
    AA,
    SS,
    interaction,
    residual,
    total,
]

df_linked_summary = df_linked_total[summary_column_list]

df_linked_summary = df_linked_summary.rename(columns={
    r_portfolio: 'Portfolio',
    r_benchmark: 'Benchmark',
    r_excess: 'Active',
    AA: 'AA',
    SS: 'SS',
    interaction: 'Interaction',
    residual: 'Residual',
    total: 'Total',
    market_value: 'Market Value'
})

summary1_columns_list = [
    'Strategy',
    'Portfolio',
    'Benchmark',
    'Active'
]

summary2_columns_list = [
    'Strategy',
    'AA',
    'SS',
    'Interaction',
    'Residual',
    'Total'
]

df_linked_summary1 = df_linked_summary[summary1_columns_list]

df_linked_summary1 = df_linked_summary1.set_index('Strategy').transpose()

df_linked_summary1 = df_linked_summary1[asset_allocations].reset_index(drop=False)

df_linked_summary1.columns = latex_summary1_column_names

df_linked_summary2 = df_linked_summary[summary2_columns_list]

df_linked_summary2 = df_linked_summary2.set_index('Strategy').transpose()

df_linked_summary2 = df_linked_summary2[asset_allocations].reset_index(drop=False)

df_linked_summary2.columns = latex_summary2_column_names

with open(output_directory + 'summary1.tex', 'w') as tf:
    tf.write(df_linked_summary1.to_latex(index=False))

with open(output_directory + 'summary2.tex', 'w') as tf:
    tf.write(df_linked_summary2.to_latex(index=False))

# Makes the pivot tables
def pivot_table(df, var):
    columns_list = ['Strategy', 'Manager', var]
    df = df[columns_list]
    df = df.pivot(index='Manager', columns='Strategy', values=var)
    df = df[asset_allocations]
    df = df.reset_index(drop=False)
    df['asset_class_sort'] = df.Manager.map(asset_class_sort)
    df = df.sort_values(['asset_class_sort'])
    df = df.drop('asset_class_sort', axis=1)
    df = df.reset_index(drop=True)
    df['Manager'] = [modelcode_to_name_dict[df['Manager'][i]] for i in range(0, len(df))]
    df.columns = latex_column_names
    return df


df_ac = pivot_table(df_linked, r_active_contribution)
df_mv = pivot_table(df_linked, market_value)
df_aa = pivot_table(df_linked, AA)
df_ss = pivot_table(df_linked, SS)
df_in = pivot_table(df_linked, interaction)
df_re = pivot_table(df_linked, residual)
df_to = pivot_table(df_linked, total)

with open(output_directory + 'ac.tex', 'w') as tf:
    tf.write(df_ac.to_latex(index=False).replace('NaN', ''))

with open(output_directory + 'mv.tex', 'w') as tf:
    tf.write(df_mv.to_latex(index=False).replace('NaN', ''))

with open(output_directory + 'aa.tex', 'w') as tf:
    tf.write(df_aa.to_latex(index=False).replace('NaN', ''))

with open(output_directory + 'ss.tex', 'w') as tf:
    tf.write(df_ss.to_latex(index=False).replace('NaN', ''))

with open(output_directory + 'in.tex', 'w') as tf:
    tf.write(df_in.to_latex(index=False).replace('NaN', ''))

with open(output_directory + 're.tex', 'w') as tf:
    tf.write(df_re.to_latex(index=False).replace('NaN', ''))

with open(output_directory + 'to.tex', 'w') as tf:
    tf.write(df_to.to_latex(index=False).replace('NaN', ''))
