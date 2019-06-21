from datetime import datetime
import pandas as pd
import numpy as np
input_directory = 'D:/data/LGS/JPM/monthly/'
output_directory = 'D:/output/LGS/alternatives/'
jpm_filename = '201905_LGSS Preliminary Performance May 2019_AddKeys.xlsx'
sustainable_filename = 'alts_sustainable_201905.xlsx'

xlsx = pd.ExcelFile(input_directory + jpm_filename)

df_jpm = pd.read_excel(
    xlsx,
    sheet_name='Page 8',
    usecols='E:O',
    skiprows=[0, 1, 2]
)
df_jpm = df_jpm.rename(
    columns={
        'Unnamed: 4': 'Manager',
        'Market Value': 'Market Value',
        '1 Month': '1 Month',
        '3 Months': '3 Month',
        'FYTD': 'FYTD',
        '1 Year': '1 Year',
        '3 Years': '3 Year',
        '5 Years': '5 Year',
        '7 Years': '7 Year'

    }
)
df_jpm = df_jpm.drop(columns=['Unnamed: 6', '2 Years'], axis=1)

# Removes NaN rows and last 2 rows which are footnotes
df_jpm = df_jpm[df_jpm['Manager'].notnull()][:-2].reset_index(drop=True)

df_jpm = df_jpm.replace('-', np.nan)

df_jpm['Market Value'] = df_jpm['Market Value']/1000000

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

df_pe = df_jpm[df_jpm['Manager'].isin(pe)].reset_index(drop=True)[0:3]

df_oa = df_jpm[df_jpm['Manager'].isin(oa)].reset_index(drop=True)[1:4]

df_da = df_jpm[df_jpm['Manager'].isin(da)].reset_index(drop=True)[2:5]

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
    left_on=['Asset Class'],
    right_on=['Asset Class'],
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

# Output to sustainable table
sustainable_columns_list = [
    'Manager_x', 'Market Value_x', '1 Month_x', '3 Month_x',
    'FYTD_x', '1 Year_x', '3 Year_x', '1 Month_z',
    '3 Month_z', 'FYTD_z', '1 Year_z', '3 Year_z'
]

sustainable_managers_to_name_dict = {
    'IFM Australian Infrastructure Wholesale Fund': 'IFM Australian Infra',
    'AMP Capital Community Infrastructure': 'AMP Community Infra',
    'ACTIS Emerging Markets 3 Fund': 'Actis EM III',
    'Archer Capital Growth Fund 2B': 'Archer Growth II',
    'EQT VI': 'EQT VI',
    'Quentin Ayers EQT Infrastructure Fund': 'EQT Infra I',
    'EQT Infrastructure Fund II': 'EQT Infra II',
    'Stafford Clean Tech Fund': 'Clean Tech I',
    'Stafford Clean Tech Fund II L.P.': 'Clean Tech II'
}

df_sustainable = df_jpm_main[sustainable_columns_list]

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
