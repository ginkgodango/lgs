"""
Attribution
"""
import pandas as pd
import numpy as np
import win32com.client
import attribution.extraction


directory = 'D:/automation/final/attribution/2019/04/'
output_directory = 'D:/automation/final/attribution/tables/'
table_filename = 'table_NOF_201904.csv'
returns_filename = 'returns_NOF_201904v2.csv'
market_values_filename = 'market_values_NOF_201904.csv'
performance_report_filepath = 'D:/automation/final/investment/2019/04/LGSS Preliminary Performance April 2019_Addkeys.xlsx'

# Loads table
df_table = attribution.extraction.load_table(directory + table_filename)

# Loads returns
df_returns = attribution.extraction.load_returns(directory + returns_filename)

# Reshapes returns dataframe from wide to long
df_returns = df_returns.transpose().reset_index(drop=False).rename(columns={'index': 'Manager'})
df_returns = pd.melt(df_returns, id_vars=['Manager'], value_name='1_r')

# Selects returns for this month
df_returns = df_returns[df_returns['Date'] == df_returns['Date'].max()].reset_index(drop=True)

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
sheet_numbers = [17, 17, 18, 18, 19, 19]
start_cells = ['C:8', 'C:35', 'C:8', 'C:35', 'C:8', 'C:35']
end_cells = ['G:22', 'G:49', 'G:22', 'G:49', 'G:22', 'G:50']

excel = win32com.client.Dispatch("Excel.Application")
df_asset_allocations = pd.DataFrame()
for i in range(0, len(asset_allocations)):
    df = attribution.extraction.load_asset_allocation(
        performance_report_filepath,
        sheet_numbers[i],
        start_cells[i],
        end_cells[i],
        excel
    )

    df_asset_allocations = pd.concat([df_asset_allocations, df], sort=True).reset_index(drop=True)
excel.Quit()

"""
Test for High Growth
April 2019
manager/sector weight * asset/total weight
"""
# ae_large = ['BT_AE', 'Ubique_AE', 'DNR_AE', 'Blackrock_AE', 'SSgA_AE', 'DSRI_AE']
# ae_small = ['WSCF_AE', 'ECP_AE']
# ae = ae_large + ae_small
#
# ie_developed = ['LSV_IE', 'MFS_IE', 'Hermes_IE', 'Longview_IE', 'AQR_IE', 'Impax_IE']
# ie_emerging = ['WellingtonEMEquity_IE', 'Delaware_IE']
# ie = ie_developed + ie_emerging
#
# ilp = ['ILP']
#
# dp = ['DP']
#
# ae_market_value = 0
# for manager in ae:
#     for i in range(0, len(df_main)):
#         if df_main['Manager'][i] == manager:
#             ae_market_value += df_main['Market Value'][i]
#
# sector_weight = dict()
# manager_weight = dict()
# for manager in ae:
#     for i in range(0,len(df_main)):
#         if df_main['Manager'][i] == manager:
#             sector_weight[manager] = df_main['Market Value'][i]/ae_market_value
#
#             for i in range(0, len(df_asset_allocations)):
#                 if df_asset_allocations['Asset Class'][i] == 'Australian Equity':
#                     # print(df_asset_allocations['Strategy'][i],manager,round(sector_weight[manager] * df_asset_allocations['Portfolio'][i]/100, 2))
#                     pass
#
# ie_market_value = 0
# for manager in ie:
#     for i in range(0, len(df_main)):
#         if df_main['Manager'][i] == manager:
#             ie_market_value += df_main['Market Value'][i]


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

# df_attribution = pd.merge(df_main, df_asset_allocations, left_on=['Manager'], right_on=['ModelCode'], how='inner')

df_attribution = pd.merge(df_returns_benchmarks, df_asset_allocations, left_on=['ModelCode'], right_on=['ModelCode'], how='inner')


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
df_attribution = df_attribution.sort_values(['Date', 'Strategy', 'Manager']).reset_index(drop=True)

df_attribution['Total'] = df_attribution['AA'] + df_attribution['SS'] + df_attribution['Interaction']
df_attribution['1_er'] = df_attribution['1_r'] - df_attribution['bmk_1_r']

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
    'AA',
    'SS',
    'Interaction',
    'Total'
]

df_attribution[column_round] = df_attribution[column_round].round(2)

df_attribution['Market Value'] = (df_attribution['Market Value']/1000000).round(2)

df_attribution_total = df_attribution[df_attribution['Manager'] == 'TO'].reset_index(drop=True)

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

df_attribution_summary1 = df_attribution_summary1[asset_allocations]

df_attribution_summary2 = df_attribution_summary[summary2_columns_list]

df_attribution_summary2 = df_attribution_summary2.set_index('Strategy').transpose()

df_attribution_summary2 = df_attribution_summary2[asset_allocations]

with open(output_directory + 'summary1.tex', 'w') as tf:
    tf.write(df_attribution_summary1.to_latex(index=True))

with open(output_directory + 'summary2.tex', 'w') as tf:
    tf.write(df_attribution_summary2.to_latex(index=True))


mv_columns_list = ['Strategy', 'Asset Class', 'Market Value']

df_mv = df_attribution[mv_columns_list]

df_mv = df_mv.pivot(index='Asset Class', columns='Strategy', values='Market Value')

df_mv = df_mv[asset_allocations]

df_mv = df_mv.reindex(list(strategy_to_modelcode_dict))

df_mv = df_mv[~df_mv.index.isin(filter_strategy_list)]

df_mv = df_mv.reset_index(drop=False)

df_mv['Asset Class'] = [strategy_to_name_dict[df_mv['Asset Class'][i]] for i in range(0, len(df_mv))]

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

with open(output_directory + 'in.tex', 'w') as tf:
    tf.write(df_in.to_latex(index=False))


er_columns_list = ['Strategy', 'Asset Class', '1_er']

df_er = df_attribution[er_columns_list]

df_er = df_er.pivot(index='Asset Class', columns='Strategy', values='1_er')

df_er = df_er[asset_allocations]

df_er = df_er.reindex(list(strategy_to_modelcode_dict))

df_er = df_er[~df_er.index.isin(filter_strategy_list)]

df_er = df_er.reset_index(drop=False)

df_er['Asset Class'] = [strategy_to_name_dict[df_er['Asset Class'][i]] for i in range(0, len(df_er))]

with open(output_directory + 'er.tex', 'w') as tf:
    tf.write(df_er.to_latex(index=False))
