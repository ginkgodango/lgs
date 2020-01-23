from datetime import datetime
import pandas as pd
import numpy as np
input_directory = 'U:/CIO/#Investment_Report/Data/input/alternatives/'
filename = 'alternatives_2019-12-31.csv'
output_directory = 'U:/CIO/#Investment_Report/Data/output/alternatives/'
ac_filename = 'alts_ac_2019_12_31.csv'
sustainable_filename = 'sustainable_alts_2019_12_31.xlsx'

df_jpm = pd.read_csv(input_directory + filename, parse_dates=['Date'])

# df_jpm['Market Value'] = df_jpm['Market Value']/1000000

round_columns = [
    'Market Value',
    '1 Month',
    '3 Month',
    'FYTD',
    '1 Year',
    '3 Year',
    '5 Year',
    '7 Year'
]

decimals_columns = [
    '1 Month',
    '3 Month',
    'FYTD',
    '1 Year',
    '3 Year',
    '5 Year',
    '7 Year'
]

df_jpm[decimals_columns] = df_jpm[decimals_columns]/100

df_jpm[round_columns] = df_jpm[round_columns].round(4)

pe = [
    'Private Equity Aggregate',
    'Cambridge Associates LLC U.S Private Equity Index**',
    'Excess Return'
]

oa = [
    'Opportunistic Alternatives Aggregate',
    'Merrill Lynch US High Yield, BB - B Rated Bond Index (Hedged) combined',
    'Excess Return'
]

da = [
    'Defensive Alternatives Aggregate',
    'AU 10 Year Govenrment Bond Yield + 4%',
    'Excess Return'
]

df_jpm1 = df_jpm[df_jpm['Date'] == df_jpm['Date'].max()].reset_index(drop=True)

df_jpm1 = df_jpm1.drop('Date', axis=1)

df_jpm1['Market Value'] = df_jpm1['Market Value']/1000000

percentage_columns = [
    '1 Month',
    '3 Month',
    'FYTD',
    '1 Year',
    '3 Year',
    '5 Year',
    '7 Year'
]

df_jpm1[percentage_columns] = df_jpm1[percentage_columns]*100

df_jpm1 = df_jpm1.round(2)

df_pe = df_jpm1[df_jpm1['Manager'].isin(pe)].reset_index(drop=True)[0:3]

df_oa = df_jpm1[df_jpm1['Manager'].isin(oa)].reset_index(drop=True)[1:4]

df_da = df_jpm1[df_jpm1['Manager'].isin(da)].reset_index(drop=True)[2:5]

with open(output_directory + 'PE.tex', 'w') as tf:
    df_pe_string = (
        df_pe
        .to_latex(index=False)
        .replace('NaN', '')
        #.replace('tabular', 'tabularx')
        .replace('llrrrrrrr', 'p{8cm}RRRRRRRR')
        .replace('lrrrrrrrr', 'p{8cm}RRRRRRRR')
        .replace('llrrrrrrl', 'p{8cm}RRRRRRRR')
    )
    tf.write(df_pe_string)

with open(output_directory + 'OA.tex', 'w') as tf:
    df_oa_string = (
        df_oa
        .to_latex(index=False)
        .replace('NaN', '')
        #.replace('tabular', 'tabularx')
        .replace('llrrrrrrr', 'p{8cm}RRRRRRRR')
        .replace('lrrrrrrrr', 'p{8cm}RRRRRRRR')
        .replace('llrrrrrrl', 'p{8cm}RRRRRRRR')
    )
    tf.write(df_oa_string)

with open(output_directory + 'DA.tex', 'w') as tf:
    df_da_string = (
        df_da
        .to_latex(index=False)
        .replace('NaN', '')
        #.replace('tabular', 'tabularx')
        .replace('llrrrrrrr', 'p{8cm}RRRRRRRR')
        .replace('lrrrrrrrr', 'p{8cm}RRRRRRRR')
        .replace('llrrrrrrl', 'p{8cm}RRRRRRRR')
    )
    tf.write(df_da_string)


asset_class = None
asset_class_list = []
for i in range(0, len(df_jpm)):
    if df_jpm['Manager'][i] in ['Private Equity', 'Opportunistic Alternatives', 'Defensive Alternatives']:
        asset_class = df_jpm['Manager'][i]
    asset_class_list.append(asset_class)

df_jpm['Asset Class'] = asset_class_list

benchmark_list = [
    'Cambridge Associates LLC U.S Private Equity Index**',
    'Merrill Lynch US High Yield, BB - B Rated Bond Index (Hedged) combined',
    'AU 10 Year Govenrment Bond Yield + 4%'
]
df_jpm_benchmarks = df_jpm[df_jpm['Manager'].isin(benchmark_list)].reset_index(drop=True)

df_jpm_main = pd.merge(
    left=df_jpm,
    right=df_jpm_benchmarks,
    left_on=['Date', 'Asset Class'],
    right_on=['Date', 'Asset Class'],
    suffixes=['_x', '_y']
)

columns_diff_list = [
    '1 Month',
    '3 Month',
    'FYTD',
    '1 Year',
    '3 Year',
    '5 Year',
    '7 Year'
]

for column in columns_diff_list:
    column_x = column + '_x'
    column_y = column + '_y'
    column_z = column + '_z'
    df_jpm_main[column_z] = df_jpm_main[column_x] - df_jpm_main[column_y]


# Outputs to active contribution table
def active_contribution_link(data):
    d = dict()
    d['12-Month Average'] = np.average(data['Market Value_x'][-12:])
    d['12_er'] = float(data['1 Year_z'][-1:])
    return pd.Series(d)


df_ac = df_jpm_main.groupby(['Manager_x']).apply(active_contribution_link)

df_ac = df_ac.reset_index(drop=False)

df_ac.columns = ['Manager', '12-Month Average', '12_er']

df_ac = df_ac[df_ac['12-Month Average'].notnull()]

df_ac = df_ac[~df_ac['Manager'].isin(pe + oa + da)]

df_ac = df_ac.reset_index(drop=True)

df_ac.to_csv(output_directory + ac_filename, index=False)

# Output to sustainable table
sustainable_columns_list = [
    'Manager_x', 'Market Value_x', '1 Month_x', '3 Month_x',
    'FYTD_x', '1 Year_x', '3 Year_x', '1 Month_z',
    '3 Month_z', 'FYTD_z', '1 Year_z', '3 Year_z'
]

sustainable_managers_to_name_dict = {
    'ACTIS Emerging Markets 3 Fund': 'Actis EM III',
    'Actis Energy 4': 'Actis EM IV',
    'Archer Capital Growth Fund 2B': 'Archer Growth II',
    'Growth Fund III': 'Archer Growth III',
    'Quentin Ayers Cerberus IREP III': 'Cerberus III',
    'Cerberus INS IV LP': 'Cerberus IV',
    'Quentin Ayers EQT Infrastructure Fund': 'EQT Infra I',
    'EQT Infrastructure Fund II': 'EQT Infra II',
    'EQT Infrasturcture III': 'EQT Infra III',
    'EQT VI': 'EQT Infra VI',
    'Stafford Clean Tech Fund': 'Clean Tech I',
    'Stafford Clean Tech Fund II L.P.': 'Clean Tech II',
    'Quadrant Private Equity No 3D': 'Quadrant III',
    'Quadrant PE 4': 'Quadrant IV',
    'Quadrant PE V': 'Quadrant V',
    'Quadrant Private Equity 6 LP': 'Quadrant VI',
    'AMP Capital Community Infrastructure': 'AMP Community Infra',
    'IFM Australian Infrastructure Wholesale Fund': 'IFM Australian Infra',
    'Lighthouse Solar Fund': 'Lighthouse'
}

df_sustainable = df_jpm_main[df_jpm_main['Date'] == df_jpm_main['Date'].max()]

df_sustainable = df_sustainable[sustainable_columns_list]

df_sustainable = df_sustainable[df_sustainable['Manager_x'].isin(sustainable_managers_to_name_dict)].reset_index(drop=True)

df_sustainable['Manager_x'] = [sustainable_managers_to_name_dict[df_sustainable['Manager_x'][i]] for i in range(0, len(df_sustainable))]

df_sustainable = df_sustainable.sort_values(['Manager_x'])

sustainable_reorder_columns = [
    'Manager_x', 'Market Value_x',
    '1 Month_x', '1 Month_z',
    '3 Month_x', '3 Month_z',
    '1 Year_x', '1 Year_z',
    '3 Year_x', '3 Year_z'
]
df_sustainable = df_sustainable[sustainable_reorder_columns]

df_sustainable.columns = [
    'manager', 'market_value',
    '1_r', '1_er',
    '3_r', '3_er',
    '12_r', '12_er',
    '36_r', '36_er'
]
df_sustainable.to_excel(output_directory + sustainable_filename, index=False)

