import datetime as dt
import pandas as pd
import numpy as np
# jpm_directory = 'U:/CIO/#Holdings/Data/input/holdings/jpm/2019/07/'
# jpm_filename = 'Priced Positions - All.csv'
# jpm_filepath = jpm_directory + jpm_filename
#
# dict_directory = 'U:/CIO/#Holdings/Data/input/dictionary/2019/07/'
# dict_filename = 'jpm_dictionary.csv'
# dict_filepath = dict_directory + dict_filename
#
# df_jpm = pd.read_csv(jpm_filepath, header=3)
#
# df_dict = pd.read_csv(dict_filepath, header=0)
#
# df_jpm = pd.merge(
#     left=df_jpm,
#     right=df_dict,
#     left_on=['Account Number', 'Account Name'],
#     right_on=['Account Number', 'Account Name']
# )
#
# df_bonds = df_jpm[df_jpm['Sector Code'].isin(['AF', 'IF'])].reset_index(drop=True)
# df_bonds = df_bonds.groupby(['Bloomberg Industry Sector']).sum()
# df_bonds = df_bonds[['Total Market Value (Base)']]
# df_bonds.to_csv(jpm_directory + 'bonds.csv', index=True)

report_date = dt.datetime(2019, 9, 30)

wscf_market_value = 180006855.20
aqr_market_value = 215381730.60
delaware_market_value = 171733052.90
wellington_market_value = 177579128.11

jpm_filepath = 'U:/CIO/#Holdings/Data/input/holdings/jpm/2019/09/Priced Positions - All.csv'
wscf_filepath = 'U:/CIO/#Holdings/Data/input/holdings/unitprices/2019/09/wscf_holdings.xls'
aqr_filepath = 'U:/CIO/#Holdings/Data/input/holdings/unitprices/2019/09/aqr_holdings.xls'
delaware_filepath = 'U:/CIO/#Holdings/Data/input/holdings/unitprices/2019/09/delaware_holdings.xlsx'
wellington_filepath = 'U:/CIO/#Holdings/Data/input/holdings/unitprices/2019/09/wellington_holdings.xls'
aeq_filepath = 'U:/CIO/#Holdings/Data/input/exclusions/LGS Exclusions List_December 2018_AEQ_Manager Version.xlsx'
ieq_filepath = 'U:/CIO/#Holdings/Data/input/exclusions/LGS Exclusions List_December 2018_IEQ_Manager Version.xlsx'
aeq_exclusions_filepath = 'U:/CIO/#Holdings/Data/output/exclusions/aeq_exclusions_' + str(report_date.date()) + '.csv'
ieq_exclusions_filepath = 'U:/CIO/#Holdings/Data/output/exclusions/ieq_exclusions_' + str(report_date.date()) + '.csv'

# Imports JPM Mandates holdings data
df_jpm = pd.read_csv(
        jpm_filepath,
        skiprows=[0, 1, 2, 3],
        header=0,
        usecols=[
                'Account Number',
                'Account Name',
                'Security ID',
                'ISIN',
                'Security Name',
                'Asset Type Description',
                'Price Date',
                'Market Price',
                'Total Units',
                'Total Market Value (Base)',
                'Local Currency'
        ],
        parse_dates=['Price Date'],
        infer_datetime_format=True
)

df_jpm = df_jpm.rename(
        columns={
                'Security ID': 'SEDOL',
                'Asset Type Description': 'Asset Type',
                'Price Date': 'Date',
                'Market Price': 'Purchase Price',
                'Total Units': 'Quantity',
                'Total Market Value (Base)': 'Market Value',
                'Local Currency': 'Currency'
        }
)

# Imports WSCF holdings data
df_wscf = pd.read_excel(
        pd.ExcelFile(wscf_filepath),
        sheet_name='Holdings',
        skiprows=[0, 1, 2, 3, 4, 5, 6, 8],
        header=0,
        usecols=[
                'Security SEDOL',
                'Security ISIN',
                'Security Name',
                'Unit Holdings',
                'Market Value (Local Currency)',
                'Security Currency'
        ]
)

df_wscf = df_wscf.rename(
        columns={
                'Security SEDOL': 'SEDOL',
                'Security ISIN': 'ISIN',
                'Unit Holdings': 'Quantity',
                'Market Value (Local Currency)': 'Market Value',
                'Security Currency': 'Currency'
        }
)

wscf_scaling_factor = wscf_market_value/df_wscf['Market Value'].sum()
df_wscf['Market Value'] = wscf_scaling_factor * df_wscf['Market Value']
df_wscf['Quantity'] = wscf_scaling_factor * df_wscf['Market Value']
df_wscf['Purchase Price'] = df_wscf['Market Value'] / df_wscf['Quantity']
df_wscf['Account Number'] = 'WSCF'
df_wscf['Account Name'] = 'LGS AUSTRALIAN EQUITIES - WSCF'
df_wscf['Date'] = report_date
df_wscf['Asset Type'] = np.nan

# Imports AQR holdings data
df_aqr = pd.read_excel(
        pd.ExcelFile(aqr_filepath),
        sheet_name='Holdings',
        skiprows=[0, 1, 2, 3, 4, 5, 6, 7],
        header=0,
        usecols=[
                'Sedol',
                'Isin',
                'Investment Description',
                'Asset Type',
                'Base Price',
                'Quantity',
                'MV Base',
                'Ccy'
        ]
)

df_aqr = df_aqr.rename(
        columns={
                'Sedol': 'SEDOL',
                'Isin': 'ISIN',
                'Investment Description': 'Security Name',
                'Base Price': 'Purchase Price',
                'MV Base': 'Market Value',
                'Ccy': 'Currency'
        })

aqr_scaling_factor = aqr_market_value/df_aqr['Market Value'].sum()
df_aqr['Market Value'] = aqr_scaling_factor * df_aqr['Market Value']
df_aqr['Quantity'] = aqr_scaling_factor * df_aqr['Quantity']
df_aqr['Account Number'] = 'AQR'
df_aqr['Account Name'] = 'LGS INTERNATIONAL EQUITIES - AQR'
df_aqr['Date'] = report_date


# Imports Delaware holdings data
df_delaware = pd.read_excel(
        pd.ExcelFile(delaware_filepath),
        sheet_name='EM UCITS holdings 8-31-19',
        header=0,
        usecols=[
                'Security SEDOL',
                'Security ISIN',
                'Security Description (Short)',
                'Position Date',
                'Shares/Par',
                'Trading Currency',
                'Traded Market Value (AUD)'
        ]
)

df_delaware = df_delaware.rename(
        columns={
                'Security SEDOL': 'SEDOL',
                'Security ISIN': 'ISIN',
                'Security Description (Short)': 'Security Name',
                'Position Date': 'Date',
                'Shares/Par': 'Quantity',
                'Trading Currency': 'Currency',
                'Traded Market Value (AUD)': 'Market Value'
        }
)

delaware_scaling_factor = delaware_market_value/df_delaware['Market Value'].sum()
df_delaware['Market Value'] = delaware_scaling_factor * df_delaware['Market Value']
df_delaware['Quantity'] = delaware_scaling_factor * df_aqr['Quantity']
df_delaware['Purchase Price'] = df_delaware['Market Value'] / df_delaware['Quantity']
df_delaware['Account Number'] = 'DELAWARE'
df_delaware['Account Name'] = 'LGS INTERNATIONAL EQUITIES - DELAWARE'
df_delaware['Date'] = report_date

# Imports Wellington holdings data
df_wellington = pd.read_excel(
        pd.ExcelFile(wellington_filepath),
        sheet_name='wellington_holdings',
        header=0,
        usecols=[
                'SEDOL',
                'ISIN',
                'Security',
                'Shares or Par Value',
                'ISO Code',
                'Market Value (Report Currency)'
        ]
)

df_wellington = df_wellington.rename(
        columns={
                'Security': 'Security Name',
                'Shares or Par Value': 'Quantity',
                'ISO Code': 'Currency',
                'Market Value (Report Currency)': 'Market Value'
        }
)

wellington_scaling_factor = wellington_market_value/df_wellington['Market Value'].sum()
df_wellington['Market Value'] = wellington_scaling_factor * df_wellington['Market Value']
df_wellington['Quantity'] = wellington_scaling_factor * df_wellington['Quantity']
df_wellington['Purchase Price'] = df_wellington['Market Value'] / df_wellington['Quantity']
df_wellington['Account Number'] = 'WELLINGTON'
df_wellington['Account Name'] = 'LGS INTERNATIONAL EQUITIES - WELLINGTON'
df_wellington['Date'] = report_date

# Joins all the dataframes
df_main = pd.concat([df_jpm, df_wscf, df_aqr, df_delaware, df_wellington], axis=0, sort=True).reset_index(drop=True)

# Removes SEDOLS with np.nan value
df_main_nan = df_main[df_main['SEDOL'].isin([np.nan])]

df_main = df_main[~df_main['SEDOL'].isin([np.nan])].reset_index(drop=True)
df_main = df_main[~df_main['ISIN'].isin([np.nan])].reset_index(drop=True)

# Cleans the SEDOL and ISIN strings
df_main['SEDOL'] = [str(df_main['SEDOL'][i]).replace(" ", "").upper() for i in range(0, len(df_main))]
df_main['ISIN'] = [str(df_main['ISIN'][i]).replace(" ", "").upper() for i in range(0, len(df_main))]

df_aeq_exclusions = pd.read_excel(
        pd.ExcelFile(aeq_filepath),
        sheet_name='AEQ',
        skiprows=[0, 1],
        header=0,
        usecols=[
                'ISSUER_ISIN',
                'ISSUER_\nSEDOL',
                'SCREEN'
        ]
)

df_aeq_exclusions = df_aeq_exclusions.rename(columns={'ISSUER_ISIN': 'ISIN', 'ISSUER_\nSEDOL': 'SEDOL'})
df_aeq_exclusions['SEDOL'] = [str(df_aeq_exclusions['SEDOL'][i]).replace(" ", "").upper() for i in range(0, len(df_aeq_exclusions))]
df_aeq_exclusions['ISIN'] = [str(df_aeq_exclusions['ISIN'][i]).replace(" ", "").upper() for i in range(0, len(df_aeq_exclusions))]

df_ieq_exclusions = pd.read_excel(
        pd.ExcelFile(ieq_filepath),
        sheet_name='IEQ',
        skiprows=[0, 1],
        header=0,
        usecols=[
                'ISSUER_ISIN',
                'SEDOL',
                'Screen'
        ]
)

df_ieq_exclusions = df_ieq_exclusions.rename(columns={'ISSUER_ISIN': 'ISIN'})
df_ieq_exclusions['SEDOL'] = [str(df_ieq_exclusions['SEDOL'][i]).replace(" ", "").upper() for i in range(0, len(df_ieq_exclusions))]
df_ieq_exclusions['ISIN'] = [str(df_ieq_exclusions['ISIN'][i]).replace(" ", "").upper() for i in range(0, len(df_ieq_exclusions))]

df_main_aeq_exclusions = pd.merge(
        left=df_main,
        right=df_aeq_exclusions,
        left_on=['SEDOL', 'ISIN'],
        right_on=['SEDOL', 'ISIN'],
        how='inner'
)

df_main_ieq_exclusions = pd.merge(
        left=df_main,
        right=df_ieq_exclusions,
        left_on=['SEDOL', 'ISIN'],
        right_on=['SEDOL', 'ISIN'],
        how='inner'
)

df_main_aeq_exclusions.to_csv(aeq_exclusions_filepath, index=False)
df_main_ieq_exclusions.to_csv(ieq_exclusions_filepath, index=False)

