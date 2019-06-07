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

start_date = datetime.datetime(2018, 7, 31)
end_date = datetime.datetime(2018, 9, 30)

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

periods = (end_date.year - start_date.year)*12 + (end_date.month - start_date.month) + 1
r_portfolio = str(periods) + '_r_portfolio'
r_benchmark = str(periods) + '_r_benchmark'
r_excess = str(periods) + '_r_excess'
r_active_contribution = str(periods) + '_r_active_contribution'
w_portfolio = str(periods) + '_w_portfolio'
w_benchmark = str(periods) + '_w_benchmark'
AA = str(periods) + '_AA'
SS = str(periods) + '_SS'
Interaction = str(periods) + '_Interaction'

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

df_benchmarks = pd.merge(left=df_returns, right=df_table, left_on=['Manager'], right_on=['Associated Benchmark'], how='inner')
df_benchmarks = df_benchmarks[['Date', 'Associated Benchmark', '1_r', 'ModelCode']]
df_benchmarks.columns = ['Date', 'Benchmark Name', 'bmk_1_r', 'ModelCode']

df_returns_benchmarks = pd.merge(left=df_returns, right=df_benchmarks, left_on=['Date', 'Manager'], right_on=['Date', 'ModelCode'], how='inner')

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


df_asset_allocations['ModelCode'] = [
    strategy_to_modelcode_dict[df_asset_allocations['Asset Class'][i]]
    for i in range(0, len(df_asset_allocations))
]

filter_list = ['ALP', 'OO', 'FW', 'TO']
filter_strategy_list = ['Australian Listed Property', 'Legacy Private Equity', 'Option Overlay', 'Forwards']

df_asset_allocations = df_asset_allocations[~df_asset_allocations['ModelCode'].isin(filter_list)]

df_attribution = pd.merge(df_returns_benchmarks, df_asset_allocations, left_on=['Date', 'ModelCode'], right_on=['Date', 'ModelCode'], how='inner')

# Expresses AA, SS, and Interaction as decimals
df_attribution['AA'] = (df_attribution['Portfolio'] - df_attribution['Benchmark'])*df_attribution['bmk_1_r']/100

df_attribution['SS'] = (df_attribution['1_r'] - df_attribution['bmk_1_r'])*df_attribution['Benchmark']/100

df_attribution['Interaction'] = (df_attribution['1_r'] - df_attribution['bmk_1_r'])*(df_attribution['Portfolio'] - df_attribution['Benchmark'])/100


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
    d['1_r'] = np.average(data['1_r'], weights=data['Portfolio'])
    d['bmk_1_r'] = np.average(data['bmk_1_r'], weights=data['Benchmark'])
    d['AA'] = np.sum(data['AA'])
    d['SS'] = np.sum(data['SS'])
    d['Interaction'] = np.sum(data['Interaction'])
    return pd.Series(d)


df_attribution_total = df_attribution.groupby(['Date', 'Strategy']).apply(weighted_average).reset_index(drop=False)

df_attribution = pd.concat([df_attribution, df_attribution_total], sort=False)

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
    'LPE': 9,
    'PE': 10,
    'OA': 11,
    'DA': 12,
    'OO': 13
}

df_attribution['asset_class_sort'] = df_attribution.Manager.map(asset_class_sort)

df_attribution = df_attribution.sort_values(['Date', 'strategy_sort', 'asset_class_sort'])

df_attribution['Total'] = df_attribution['AA'] + df_attribution['SS'] + df_attribution['Interaction']
df_attribution['1_er'] = df_attribution['1_r'] - df_attribution['bmk_1_r']
df_attribution['Active Contribution'] = df_attribution['1_er'] * df_attribution['Portfolio']/100

# df_attribution[['AA', 'SS', 'Interaction']] = df_attribution[['AA', 'SS', 'Interaction']].astype(float).round(4)*100
column_order = [
    'Date',
    'Strategy',
    'Manager',
    'Asset Class',
    'Market Value',
    '1_r',
    'bmk_1_r',
    '1_er',
    'Portfolio',
    'Benchmark',
    'Active Contribution',
    'AA',
    'SS',
    'Interaction',
    'Total'
]

df_attribution = df_attribution[column_order]

column_percentage = [
    '1_r',
    'bmk_1_r',
    '1_er',
    'Active Contribution',
    'AA',
    'SS',
    'Interaction',
    'Total'
]

df_attribution[column_percentage] = df_attribution[column_percentage].astype(float)*100

column_round = [
    'Market Value',
    '1_r',
    'bmk_1_r',
    '1_er',
    'Portfolio',
    'Benchmark',
    'Active Contribution',
    'AA',
    'SS',
    'Interaction',
    'Total'
]

df_attribution[column_round] = df_attribution[column_round].round(2)

df_attribution['Market Value'] = (df_attribution['Market Value']/1000000).round(2)

df_attribution = df_attribution.reset_index(drop=True)

df_attribution_total = df_attribution[df_attribution['Manager'] == 'TO'].reset_index(drop=True)

# Test of chain linking
def test(data):
    d = dict()
    d[r_portfolio] = np.prod(1 + data['1_r']) - 1
    d[r_benchmark] = np.prod(1 + data['bmk_1_r']) - 1
    d[r_active_contribution] = np.prod(1 + data['Active Contribution']) - 1
    d[w_portfolio] = np.average(data['Portfolio']/100)
    d[w_benchmark] = np.average(data['Benchmark']/100)
    d[AA] = np.prod(1 + data['AA']) - 1
    d[SS] = np.prod(1 + data['SS']) - 1
    d['Interaction'] = np.prod(1 + data['Interaction']) - 1
    return pd.Series(d)


df_test = df_attribution.copy()

df_test_c = df_test.groupby(['Strategy', 'Manager']).apply(test)

df_test_c[r_excess] = df_test_c[r_portfolio] - df_test_c[r_benchmark]

df_test_c['Total'] = df_test_c[AA] + df_test_c[SS] + df_test_c['Interaction']

df_test_c['Residual'] = df_test_c[r_active_contribution] - df_test_c['Total']

"""
summary_column_list = [
    'Strategy',
    '1_r',
    'bmk_1_r',
    '1_er',
    'AA',
    'SS',
    'Interaction',
    'Total',
    'Market Value'
]

df_attribution_summary = df_attribution_total[summary_column_list]

df_attribution_summary = df_attribution_summary.rename(columns={
    '1_r': 'Portfolio',
    'bmk_1_r': 'Benchmark',
    '1_er': 'Active',
    'AA': 'AA',
    'SS': 'SS',
    'Interaction': 'Interaction',
    'Total': 'Total',
    'Market Value': 'Market Value'
})

summary1_columns_list =[
    'Strategy',
    'Portfolio',
    'Benchmark',
    'Active'
]

summary2_columns_list =[
    'Strategy',
    'AA',
    'SS',
    'Interaction',
    'Total'
]

df_attribution_summary1 = df_attribution_summary[summary1_columns_list]

df_attribution_summary1 = df_attribution_summary1.set_index('Strategy').transpose()

df_attribution_summary1 = df_attribution_summary1[asset_allocations].reset_index(drop=False)

df_attribution_summary1.columns = latex_summary1_column_names

df_attribution_summary2 = df_attribution_summary[summary2_columns_list]

df_attribution_summary2 = df_attribution_summary2.set_index('Strategy').transpose()

df_attribution_summary2 = df_attribution_summary2[asset_allocations].reset_index(drop=False)

df_attribution_summary2.columns = latex_summary2_column_names

with open(output_directory + 'summary1.tex', 'w') as tf:
    tf.write(df_attribution_summary1.to_latex(index=False))

with open(output_directory + 'summary2.tex', 'w') as tf:
    tf.write(df_attribution_summary2.to_latex(index=False))


mv_columns_list = ['Strategy', 'Asset Class', 'Market Value']

df_mv = df_attribution[mv_columns_list]

df_mv = df_mv.pivot(index='Asset Class', columns='Strategy', values='Market Value')

df_mv = df_mv[asset_allocations]

df_mv = df_mv.reindex(list(strategy_to_modelcode_dict))

df_mv = df_mv[~df_mv.index.isin(filter_strategy_list)]

df_mv = df_mv.reset_index(drop=False)

df_mv['Asset Class'] = [strategy_to_name_dict[df_mv['Asset Class'][i]] for i in range(0, len(df_mv))]

df_mv.columns = latex_column_names

with open(output_directory + 'mv.tex', 'w') as tf:
    tf.write(df_mv.to_latex(index=False))


aa_columns_list = ['Strategy', 'Asset Class', 'AA']

df_aa = df_attribution[aa_columns_list]

df_aa = df_aa.pivot(index='Asset Class', columns='Strategy', values='AA')

df_aa = df_aa[asset_allocations]

df_aa = df_aa.reindex(list(strategy_to_modelcode_dict))

df_aa = df_aa[~df_aa.index.isin(filter_strategy_list)]

df_aa = df_aa.reset_index(drop=False)

df_aa['Asset Class'] = [strategy_to_name_dict[df_aa['Asset Class'][i]] for i in range(0, len(df_aa))]

df_aa.columns = latex_column_names

with open(output_directory + 'aa.tex', 'w') as tf:
    tf.write(df_aa.to_latex(index=False))


ss_columns_list = ['Strategy', 'Asset Class', 'SS']

df_ss = df_attribution[ss_columns_list]

df_ss = df_ss.pivot(index='Asset Class', columns='Strategy', values='SS')

df_ss = df_ss[asset_allocations]

df_ss = df_ss.reindex(list(strategy_to_modelcode_dict))

df_ss = df_ss[~df_ss.index.isin(filter_strategy_list)]

df_ss = df_ss.reset_index(drop=False)

df_ss['Asset Class'] = [strategy_to_name_dict[df_ss['Asset Class'][i]] for i in range(0, len(df_ss))]

df_ss.columns = latex_column_names

with open(output_directory + 'ss.tex', 'w') as tf:
    tf.write(df_ss.to_latex(index=False))


in_columns_list = ['Strategy', 'Asset Class', 'Interaction']

df_in = df_attribution[in_columns_list]

df_in = df_in.pivot(index='Asset Class', columns='Strategy', values='Interaction')

df_in = df_in[asset_allocations]

df_in = df_in.reindex(list(strategy_to_modelcode_dict))

df_in = df_in[~df_in.index.isin(filter_strategy_list)]

df_in = df_in.reset_index(drop=False)

df_in['Asset Class'] = [strategy_to_name_dict[df_in['Asset Class'][i]] for i in range(0, len(df_in))]

df_in.columns = latex_column_names

with open(output_directory + 'in.tex', 'w') as tf:
    tf.write(df_in.to_latex(index=False))


ac_columns_list = ['Strategy', 'Asset Class', 'Active Contribution']

df_ac = df_attribution[ac_columns_list]

df_ac = df_ac.pivot(index='Asset Class', columns='Strategy', values='Active Contribution')

df_ac = df_ac[asset_allocations]

df_ac = df_ac.reindex(list(strategy_to_modelcode_dict))

df_ac = df_ac[~df_ac.index.isin(filter_strategy_list)]

# df_ac.plot.bar().axhline(linewidth=0.5, color='black')

df_ac = df_ac.reset_index(drop=False)

df_ac['Asset Class'] = [strategy_to_name_dict[df_ac['Asset Class'][i]] for i in range(0, len(df_ac))]

df_ac.columns = latex_column_names

with open(output_directory + 'ac.tex', 'w') as tf:
    tf.write(df_ac.to_latex(index=False))


to_columns_list = ['Strategy', 'Asset Class', 'Total']

df_to = df_attribution[to_columns_list]

df_to = df_to.pivot(index='Asset Class', columns='Strategy', values='Total')

df_to = df_to[asset_allocations]

df_to = df_to.reindex(list(strategy_to_modelcode_dict))

df_to = df_to[~df_to.index.isin(filter_strategy_list)]

df_to = df_to.reset_index(drop=False)

df_to['Asset Class'] = [strategy_to_name_dict[df_to['Asset Class'][i]] for i in range(0, len(df_ac))]

df_to.columns = latex_column_names

with open(output_directory + 'to.tex', 'w') as tf:
    tf.write(df_to.to_latex(index=False))
"""
