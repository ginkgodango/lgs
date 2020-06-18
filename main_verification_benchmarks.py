# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 15:07:53 2019

@author: MerrillN
"""
import pandas as pd
import numpy as np
import datetime as dt

report_date = dt.datetime(2020, 4, 30)
lgs_filepath = 'U:/CIO/#Investment_Report/Data/input/returns/returns_2020-04-30.csv'
jpm_returns_filepath = 'U:/CIO/#Data/input/jpm/performance/2020/04/Historical Time Series - Monthly - Main Returns.xlsx'
jpm_benchmarks_filepath = 'U:/CIO/#Data/input/jpm/performance/2020/04/Historical Time Series - Monthly - Main Benchmarks.xlsx'
dict_filepath = 'U:/CIO/#Investment_Report/Data/input/link/Data Dictionary V3.xlsx'
tolerance = 0.01

# Imports LGS dictionary
df_dict = pd.read_excel(
        pd.ExcelFile(dict_filepath),
        sheet_name='Sheet1'
        )

benchmark_list = list(set(df_dict['Associated Benchmark']))

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
df_lgs = pd.melt(df_lgs, id_vars=['Manager'], value_name='LGS Benchmark')
df_lgs = df_lgs.sort_values(['Manager', 'Date'])
df_lgs = df_lgs.reset_index(drop=True)

df_lgs_returns = df_lgs[~df_lgs['Manager'].isin(benchmark_list)]
df_lgs_benchmarks = df_lgs[df_lgs['Manager'].isin(benchmark_list)]



# Imports the JPM return file
use_managerid = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12]
use_accountid = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12]

# use_managerid = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11]
# use_accountid = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14]
footnote_rows = 28

df_jpm_returns = pd.read_excel(
        pd.ExcelFile(jpm_returns_filepath),
        sheet_name='Sheet1',
        skiprows=use_accountid,
        skipfooter=footnote_rows,
        header=1
        )

df_jpm_returns = df_jpm_returns.rename(columns={'Unnamed: 0': 'Date'})
df_jpm_returns = df_jpm_returns.set_index('Date')
df_jpm_returns = df_jpm_returns.transpose()
df_jpm_returns = df_jpm_returns.reset_index(drop=False)
df_jpm_returns = df_jpm_returns.rename(columns={'index': 'Manager'})
df_jpm_returns = pd.melt(df_jpm_returns, id_vars=['Manager'], value_name='JPM Return')
df_jpm_returns = df_jpm_returns.sort_values(['Manager', 'Date'])
df_jpm_returns = df_jpm_returns.reset_index(drop=True)

df_jpm_returns = df_jpm_returns.replace('-', np.NaN)
df_jpm_returns['JPM Return'] = df_jpm_returns['JPM Return']/100

df_jpm_benchmarks = pd.read_excel(
        pd.ExcelFile(jpm_benchmarks_filepath),
        sheet_name='Sheet1',
        skiprows=use_accountid,
        skipfooter=footnote_rows,
        header=1
        )

df_jpm_benchmarks = df_jpm_benchmarks.rename(columns={'Unnamed: 0': 'Date'})
df_jpm_benchmarks = df_jpm_benchmarks.set_index('Date')
df_jpm_benchmarks = df_jpm_benchmarks.transpose()
df_jpm_benchmarks = df_jpm_benchmarks.reset_index(drop=False)
df_jpm_benchmarks = df_jpm_benchmarks.rename(columns={'index': 'Manager'})
df_jpm_benchmarks = pd.melt(df_jpm_benchmarks, id_vars=['Manager'], value_name='JPM Benchmark')
df_jpm_benchmarks = df_jpm_benchmarks.sort_values(['Manager', 'Date'])
df_jpm_benchmarks = df_jpm_benchmarks.reset_index(drop=True)

df_jpm_benchmarks = df_jpm_benchmarks.replace('-', np.NaN)
df_jpm_benchmarks['JPM Benchmark'] = df_jpm_benchmarks['JPM Benchmark']/100

# Merges the LGS time series with the LGS data dictionary
# df_lgs_dict = pd.merge(
#         left=df_lgs,
#         right=df_dict,
#         left_on=['Manager'],
#         right_on=['LGS Data'],
#         how='inner'
#         )

df_jpm = pd.merge(
        left=df_jpm_returns,
        right=df_jpm_benchmarks,
        left_on=['Manager', 'Date'],
        right_on=['Manager', 'Date'],
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
# s1 = set(df_lgs_dict['LGS Data'])
# s2 = set(df_jpm_dict['LGS Data'])
# print('These managers are in the LGS file but not in the JPM file file:\n', s1 - s2)

df_lgs_jpm = pd.merge(
        left=df_lgs_benchmarks,
        right=df_jpm_dict,
        left_on=['Manager', 'Date'],
        right_on=['Associated Benchmark', 'Date'],
        how='inner'
        )

df_lgs_jpm['Deviation'] = df_lgs_jpm['JPM Benchmark'] - df_lgs_jpm['LGS Benchmark']
df_lgs_jpm['Deviation_ABS'] = abs(df_lgs_jpm['JPM Benchmark'] - df_lgs_jpm['LGS Benchmark'])
df_lgs_jpm['In Tolerance'] = df_lgs_jpm['Deviation_ABS'] <= tolerance

# Selects columns
select_columns = [
        'Manager_x',
        'Manager_y',
        'JPM Data',
        'Date',
        'JPM Return',
        'JPM Benchmark',
        'LGS Benchmark',
        'Deviation',
        'Deviation_ABS',
        'In Tolerance'
        ]

df_final = df_lgs_jpm[select_columns]

fix_back_fill_list = []
jpm_back_fill_list = []
lgs_back_fill_list = []
for i in range(0, len(df_final)):
    if pd.isna(df_final['JPM Benchmark'][i]) and pd.isna(df_final['LGS Benchmark'][i]):
        fix_back_fill_list.append(0)
        jpm_back_fill_list.append(0)
        lgs_back_fill_list.append(0)
    elif (pd.isna(df_final['JPM Benchmark'][i]) or df_final['JPM Benchmark'][i] == 0) and not (pd.isna(df_final['LGS Benchmark'][i] or df_final['LGS Benchmark'][i] == 0)):
        fix_back_fill_list.append(1)
        jpm_back_fill_list.append(1)
        lgs_back_fill_list.append(0)
    elif not (pd.isna(df_final['JPM Benchmark'][i]) or df_final['JPM Benchmark'][i] == 0) and (pd.isna(df_final['LGS Benchmark'][i] or df_final['LGS Benchmark'][i] == 0)):
        fix_back_fill_list.append(1)
        jpm_back_fill_list.append(0)
        lgs_back_fill_list.append(1)
    elif (df_final['JPM Benchmark'][i] == 0) or (df_final['LGS Benchmark'][i] == 0):
        fix_back_fill_list.append(1)
        jpm_back_fill_list.append(0)
        lgs_back_fill_list.append(0)
    else:
        fix_back_fill_list.append(0)
        jpm_back_fill_list.append(0)
        lgs_back_fill_list.append(0)

df_final['Fix Needs Backfill'] = fix_back_fill_list
df_final['JPM Needs Backfill'] = jpm_back_fill_list
df_final['LGS Needs Backfill'] = lgs_back_fill_list


df_final = df_final.sort_values(['JPM Data', 'Date'])
df_final = df_final[df_final['Date'] >= dt.datetime(2012, 1, 31)].reset_index(drop=True)
df_final_filtered = df_final[~(df_final['JPM Return'].isin([np.nan]))].reset_index(drop=True)

df_final_today_not_filtered = df_final[df_final['Date'].isin([report_date])].reset_index(drop=True)
df_final_today_not_filtered = df_final_today_not_filtered[df_final_today_not_filtered['In Tolerance'].isin([0])].reset_index(drop=True)

df_final_today_filtered = df_final_filtered[df_final_filtered['Date'].isin([report_date])].reset_index(drop=True)
df_final_today_filtered = df_final_today_filtered[df_final_today_filtered['In Tolerance'].isin([0])].reset_index(drop=True)

df_final_deviants_not_filtered = df_final[df_final['In Tolerance'].isin([0])].reset_index(drop=True)

df_final_deviants_filtered = df_final_filtered[df_final_filtered['In Tolerance'].isin([0])].reset_index(drop=True)

df_final_backfill = df_final[df_final['Fix Needs Backfill'].isin([1])].reset_index(drop=True)

# Ask Merrill before touching below this line..................................
# Outputs to csv file
df_final_deviants_not_filtered.to_csv('U:/CIO/#Investment_Report/Data/output/verification/verification_benchmarks_202004_history_not_filtered.csv', index=False)
df_final_deviants_filtered.to_csv('U:/CIO/#Investment_Report/Data/output/verification/verification_benchmarks_202004_history_filtered.csv', index=False)
df_final_today_not_filtered.to_csv('U:/CIO/#Investment_Report/Data/output/verification/verification_benchmarks_202004_today_not_filtered.csv', index=False)
df_final_today_filtered.to_csv('U:/CIO/#Investment_Report/Data/output/verification/verification_benchmarks_202004_today_filtered.csv', index=False)
