# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 15:07:53 2019

@author: MerrillN
"""
import pandas as pd
import numpy as np
lgs_filepath = 'U:/CIO/#Investment_Report/Data/input/returns/returns_2019-06-30.csv'
jpm_filepath = 'U:/CIO/#Investment_Report/Data/input/returns/20190726 Historical Time Series.xlsx'
dict_filepath = 'U:/CIO/#Investment_Report/Data/input/link/Data Dictionary.xlsx'
tolerance = 0.05

# Imports the LGS return file
df_lgs = pd.read_csv(
        lgs_filepath,
        index_col='Date',
        parse_dates=['Date'],
        infer_datetime_format=True,
        float_precision='round_trip'
        )

df_lgs = df_lgs.transpose()
df_lgs = df_lgs.reset_index(drop=False)
df_lgs = df_lgs.rename(columns={'index': 'Manager'})
df_lgs = pd.melt(df_lgs, id_vars=['Manager'], value_name='Return_LGS')
df_lgs = df_lgs.sort_values(['Manager', 'Date'])
df_lgs = df_lgs.reset_index(drop=True)

# Imports the JPM return file
jpm_xlsx = pd.ExcelFile(jpm_filepath)
use_managerid = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11]
use_accountid = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14]
footnote_rows = 28

df_jpm = pd.read_excel(
        jpm_xlsx,
        sheet_name='Sheet1',
        skiprows=use_accountid,
        skipfooter=footnote_rows,
        header=1
        )

df_jpm = df_jpm.rename(columns={'Unnamed: 0': 'Date'})
df_jpm = df_jpm.set_index('Date')
df_jpm = df_jpm.transpose()
df_jpm = df_jpm.reset_index(drop=False)
df_jpm = df_jpm.rename(columns={'index': 'Manager'})
df_jpm = pd.melt(df_jpm, id_vars=['Manager'], value_name='Return_JPM')
df_jpm = df_jpm.sort_values(['Manager', 'Date'])
df_jpm = df_jpm.reset_index(drop=True)

df_jpm = df_jpm.replace('-', np.NaN)
df_jpm['Return_JPM'] = df_jpm['Return_JPM']/100

# Imports the data dictionary
dict_xlsx = pd.ExcelFile('U:/CIO/#Investment_Report/Data/input/link/Data Dictionary.xlsx')
df_dict = pd.read_excel(
        dict_xlsx,
        sheet_name='Sheet1'
        )

df_lgs_dict = pd.merge(
        left=df_lgs,
        right=df_dict,
        left_on=['Manager'],
        right_on=['LGS Data'],
        how='inner'
        )

df_jpm_dict = pd.merge(
        left=df_jpm,
        right=df_dict,
        left_on=['Manager'],
        right_on=['JPM Code'],
        how='inner'
        )

# Diagnostics
s1 = set(df_lgs_dict['LGS Data'])
s2 = set(df_jpm_dict['LGS Data'])
print('These managers are in the LGS file but not in the JPM file file:\n', s1 - s2)

df_lgs_jpm = pd.merge(
        left=df_lgs_dict,
        right=df_jpm_dict,
        left_on=['JPM Code', 'Date'],
        right_on=['JPM Code', 'Date'],
        how='inner'
        )

df_lgs_jpm['Deviation'] = df_lgs_jpm['Return_JPM'] - df_lgs_jpm['Return_LGS']
df_lgs_jpm['Deviation_ABS'] = abs(df_lgs_jpm['Return_JPM'] - df_lgs_jpm['Return_LGS'])
df_lgs_jpm['In Tolerance'] = df_lgs_jpm['Deviation_ABS'] <= tolerance

# Selects columns
select_columns = [
        'Manager_x',
        'Manager_y',
        'JPM Data_y',
        'Date',
        'Return_JPM',
        'Return_LGS',
        'Deviation',
        'Deviation_ABS',
        'In Tolerance'
        ]

df_final = df_lgs_jpm[select_columns]

# Ask Merrill before touching below this line..................................
# Outputs to csv file
#df_final.to_csv('U:/CIO/#Investment_Report/Data/output/verification/verification.csv', index=False)

