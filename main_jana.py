import pandas as pd
import numpy as np
import datetime

df_link = pd.read_csv('U:/CIO/#Investment_Report/Data/input/link/assetclass_dictionary.csv')
df_link_ie = df_link[df_link['Asset Class'] == 'IE'].reset_index(drop=True)
df_link_ie = df_link_ie[['LGS Code', 'LGS Benchmark']]

df_lgs = pd.read_csv(
        'U:/CIO/#Investment_Report/Data/input/returns/returns_2019-08-31_hybrid.csv',
        index_col='Date',
        parse_dates=['Date'],
        infer_datetime_format=True,
        float_precision='round_trip'
        )
df_lgs = df_lgs.transpose()
df_lgs = df_lgs.reset_index(drop=False)
df_lgs = df_lgs.rename(columns={'index': 'Manager'})
df_lgs = pd.melt(df_lgs, id_vars=['Manager'], value_name='1_month_return')
df_lgs = df_lgs.sort_values(['Manager', 'Date'])
df_lgs = df_lgs.reset_index(drop=True)

ie_manager_list = list(df_link_ie['LGS Code'].drop_duplicates().values.tolist())
df_lgs_ie_managers = df_lgs[df_lgs['Manager'].isin(ie_manager_list)].reset_index(drop=True)

ie_benchmark_list = list(df_link_ie['LGS Benchmark'].drop_duplicates().values.tolist())
df_lgs_ie_benchmarks = df_lgs[df_lgs['Manager'].isin(ie_benchmark_list)].reset_index(drop=True)

df_lgs_ie_managers = pd.merge(
    left=df_lgs_ie_managers,
    right=df_link_ie,
    left_on=['Manager'],
    right_on=['LGS Code'],
    how='inner'
)

df_final = pd.merge(
    left=df_lgs_ie_managers,
    right=df_lgs_ie_benchmarks,
    left_on=['LGS Benchmark', 'Date'],
    right_on=['Manager', 'Date'],
    how='inner'
)

df_final.rename(columns={
    '1_month_return_x': '1_month_return_manager',
    '1_month_return_y': '1_month_return_benchmark'
})

df_final = df_final[['LGS Code', 'Date', '1_month_return_x', 'LGS Benchmark', '1_month_return_y']]

df_final = df_final.sort_values(['LGS Code', 'Date'])

df_final.to_csv('U:/CIO/#Investment_Report/Data/output/jana/ie_panel.csv', index=False)

# Strategy Benchmarks
df_cpi = pd.read_csv(
    'U:/CIO/#Investment_Report/Data/input/product/g1-data_201906.csv',
    header=10,
    usecols=['Series ID', 'GCPIAGQP'],
    parse_dates=['Series ID'],
    index_col=['Series ID']
)

df_cpi = df_cpi.rename(columns={'GCPIAGQP': 'Inflation'})

df_cpi['Inflation'] = df_cpi['Inflation']/100

# df_cpi = df_cpi.reset_index(drop=True)

years = [1, 2, 3, 4, 5, 6, 7]

for year in years:

    column_name = str(year) + '_Year'

    quarters = year*4

    df_cpi[column_name] = df_cpi['Inflation'].rolling(quarters).apply(lambda r: (np.prod(1+r)**(1/year))-1)

df_benchmark = pd.DataFrame()

df_benchmark['High Growth'] = (df_cpi['7_Year'] + 0.035) * (1 - 0.08)

df_benchmark['Growth'] = (df_cpi['5_Year'] + 0.03) * (1 - 0.08)

df_benchmark['Balanced Growth'] = (df_cpi['5_Year'] + 0.03) * (1 - 0.085)

df_benchmark['Balanced'] = (df_cpi['3_Year'] + 0.02) * (1 - 0.1)

df_benchmark['Conservative'] = (df_cpi['2_Year'] + 0.015) * (1 - 0.115)

df_benchmark['Employer Reserve'] = (0.0575) * (1 - 0.08)

df_benchmark = df_benchmark.resample('M').pad()

df_r = pd.read_csv(
        'U:/CIO/#Investment_Report/Data/input/returns/returns_2019-06-30.csv',
        index_col='Date',
        parse_dates=['Date'],
        infer_datetime_format=True,
        float_precision='round_trip',
        usecols=['Date', 'AUBI_Index']
        )

df_r['2_Year'] = df_r['AUBI_Index'].rolling(24).apply(lambda r: (np.prod(1+r)**(1/2))-1)

df_benchmark['Cash'] = (df_r['2_Year'] + 0.0025) * (1 - 0.15)

df_benchmark.to_csv('U:/CIO/#Investment_Report/Data/input/product/strategy_benchmarks_201905.csv')


# ASSET CLASS REVIEW
# # User input data
# jpm_directory = 'D:/data/Managers/JPM/holdings/2019/05/'
# jpm_filename = 'Priced Positions - All.csv'
# jpm_report_date = datetime.datetime(2019, 5, 31)
#
# cfs_directory = 'D:/data/Managers/CFS/holdings/2019/04/'
# cfs_filename = 'WSCF Holdings April 2019.xls'
# cfs_report_date = datetime.datetime(2019, 4, 30)
# cfs_market_value = 170334383
#
# output_directory = 'D:/data/Managers/JANA/'
# output_filename = 'ae_review_201905.xlsx'
#
# # Loads JPM
# df_jpm = pd.read_csv(
#     jpm_directory + jpm_filename,
#     header=3,
#     usecols=[
#         'Account Number',
#         'Account Name',
#         'Security ID',
#         'ISIN',
#         'Security Name',
#         'Asset Type Description',
#         'Price Date',
#         'Market Price',
#         'Total Units',
#         'Total Market Value (Base)',
#         'Local Currency'
#     ],
#     parse_dates=['Price Date'],
#     infer_datetime_format=True
# )
#
# # Renames the column headers
# df_jpm = df_jpm.rename(
#     columns={
#         'Account Number': 'Account Number',
#         'Account Name': 'Account Name',
#         'Security ID': 'SEDOL',
#         'ISIN': 'ISIN',
#         'Security Name': 'Security Name',
#         'Asset Type Description': 'Asset Type',
#         'Price Date': 'Price Date',
#         'Market Price': 'Purchase Price',
#         'Total Units': 'Quantity',
#         'Total Market Value (Base)': 'Market Value',
#         'Local Currency': 'Currency'
#     }
# )
#
# # Creates Account Name to Manager Name dictionary
# jpm_account_name_to_manager_name_dict = {
#     'LGS AUSTRALIAN EQUITIES - PENDAL': 'Pendal',
#     'LGS AUSTRALIAN EQUITIES - UBIQUE': 'Ubique',
#     'LGS AUSTRALIAN EQUITIES - BLACKROCK': 'Blackrock',
#     'LGS AUSTRALIAN EQUITIES DNR CAPITAL': 'DNR',
#     'LGS AUSTRALIAN EQUITIES - SSGA': 'SSGA',
#     'LGS AUSTRALIAN EQUITIES - ECP': 'ECP',
#     'LGS AUSTRALIAN EQUITIES - SRI UBS': 'DSRI'
# }
#
# # Selects Australian Equities
# df_jpm_ae = df_jpm[df_jpm['Account Name'].isin(jpm_account_name_to_manager_name_dict)].reset_index(drop=True)
#
# # Creates Asset Class column
# df_jpm_ae['Asset Class'] = 'Australian Equity'
#
# # Creates Manager Name column
# df_jpm_ae['Manager Name'] = [
#     jpm_account_name_to_manager_name_dict[df_jpm_ae['Account Name'][i]]
#     for i in range(0, len(df_jpm_ae))
# ]
#
# # Creates Report Date column
# df_jpm_ae['Report Date'] = jpm_report_date
#
# # Selects columns
# jpm_column_list = [
#     'Asset Class',
#     'Manager Name',
#     'Report Date',
#     'SEDOL',
#     'ISIN',
#     'Security Name',
#     'Currency',
#     'Quantity',
#     'Purchase Price',
#     'Market Value'
# ]
# df_jpm_ae = df_jpm_ae[jpm_column_list]
#
# # Loads the CFS file
# df_cfs = pd.read_excel(cfs_directory + cfs_filename, header=8)
#
# df_cfs = df_cfs.rename(
#     columns={
#         'Security ISIN': 'ISIN',
#         'Security CUSIP': 'CUSIP',
#         'Security SEDOL': 'SEDOL',
#         'Security Name': 'Security Name',
#         'Sector': 'Sector',
#         'Country': 'Country',
#         'Country ISO Code': 'Country ISO',
#         'Security Currency': 'Currency',
#         'Market Price (Local Currency)': 'Purchase Price',
#         'Stocks in Portfolio': 'Stocks in Portfolio',
#         'Market Value (Base Currency)': 'Market Value',
#         'Market Value (Local Currency)': 'Market Value (Local Currency)',
#         'Maket Value (Calculated)': 'Maket Value (Calculated)',
#         'Market Value (Local Calculated)': 'Market Value (Local Calculated)',
#         'Portfolio Weight': 'Portfolio Weight',
#         'Unit Holdings': 'Quantity',
#         'Coupon': 'Coupon',
#         'Strike Price': 'Strike Price',
#         'Maturity Date': 'Maturity Date',
#         'Position Cost (Base Currency)': 'Position Cost (Base Currency)',
#         'Position Cost (Local Currency)': 'Position Cost (Local Currency)',
#         'Custom Sector Code': 'Custom Sector Code',
#         'Custom Region/Country Code': 'Custom Region/Country Code'
#
#     }
# )
# # Rescales market values and quantity
# cfs_scaling_ratio = cfs_market_value/df_cfs['Market Value'].sum()
# df_cfs['Market Value'] = df_cfs['Market Value'] * cfs_scaling_ratio
# df_cfs['Quantity'] = df_cfs['Quantity'] * cfs_scaling_ratio
#
# df_cfs['Asset Class'] = 'Australian Equity'
# df_cfs['Manager Name'] = 'CFS'
# df_cfs['Report Date'] = cfs_report_date
#
# # Selects CFS columns
# cfs_columns_list = [
#     'Asset Class',
#     'Manager Name',
#     'Report Date',
#     'SEDOL',
#     'ISIN',
#     'Security Name',
#     'Currency',
#     'Quantity',
#     'Purchase Price',
#     'Market Value'
# ]
# df_cfs = df_cfs[cfs_columns_list]
#
# # Totals the market value
# df_ae = pd.concat([df_jpm_ae, df_cfs]).reset_index(drop=True)
#
# ae_market_value = df_ae['Market Value'].sum()
#
#
# def total(data):
#     d = dict()
#     d['Manager Market Value'] = np.sum(data['Market Value'])
#     return pd.Series(d)
#
#
# df_ae_totals = df_ae.groupby(['Asset Class', 'Manager Name', 'Report Date']).apply(total)
#
# df_ae_totals['Asset Class Market Value'] = ae_market_value
#
# df_ae_totals['Proportion of Asset Class'] = df_ae_totals['Manager Market Value']/ae_market_value
#
# df_ae_totals = df_ae_totals.reset_index(drop=False)
#
# manager_name_to_target_allocation_dict = {
#     'Pendal': 0.16,
#     'Ubique': 0.12,
#     'Blackrock': 0.23,
#     'DNR': 0.14,
#     'SSGA': 0.23,
#     'ECP': 0.04,
#     'DSRI': 0.01,
#     'CFS': 0.08
# }
#
# df_ae_totals['Target Allocation'] = [
#     manager_name_to_target_allocation_dict[df_ae_totals['Manager Name'][i]]
#     for i in range(0, len(df_ae_totals))
# ]
#
# writer = pd.ExcelWriter(output_directory + output_filename, engine='xlsxwriter')
#
# df_ae.to_excel(
#     writer,
#     sheet_name='manager_holdings',
#     index=False
# )
#
# df_ae_totals.to_excel(
#     writer,
#     sheet_name='summary_manager_holdings',
#     index=False
# )
#
# writer.save()
