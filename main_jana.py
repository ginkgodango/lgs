import pandas as pd
import numpy as np
import datetime

# User input data
jpm_directory = 'D:/data/Managers/JPM/holdings/2019/05/'
jpm_filename = 'Priced Positions - All.csv'
jpm_report_date = datetime.datetime(2019, 5, 31)

cfs_directory = 'D:/data/Managers/CFS/holdings/2019/04/'
cfs_filename = 'WSCF Holdings April 2019.xls'
cfs_report_date = datetime.datetime(2019, 4, 30)
cfs_market_value = 170334383

output_directory = 'D:/data/Managers/JANA/'
output_filename = 'ae_review_201905.xlsx'

# Loads JPM
df_jpm = pd.read_csv(
    jpm_directory + jpm_filename,
    header=3,
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

# Renames the column headers
df_jpm = df_jpm.rename(
    columns={
        'Account Number': 'Account Number',
        'Account Name': 'Account Name',
        'Security ID': 'SEDOL',
        'ISIN': 'ISIN',
        'Security Name': 'Security Name',
        'Asset Type Description': 'Asset Type',
        'Price Date': 'Price Date',
        'Market Price': 'Purchase Price',
        'Total Units': 'Quantity',
        'Total Market Value (Base)': 'Market Value',
        'Local Currency': 'Currency'
    }
)

# Creates Account Name to Manager Name dictionary
jpm_account_name_to_manager_name_dict = {
    'LGS AUSTRALIAN EQUITIES - PENDAL': 'Pendal',
    'LGS AUSTRALIAN EQUITIES - UBIQUE': 'Ubique',
    'LGS AUSTRALIAN EQUITIES - BLACKROCK': 'Blackrock',
    'LGS AUSTRALIAN EQUITIES DNR CAPITAL': 'DNR',
    'LGS AUSTRALIAN EQUITIES - SSGA': 'SSGA',
    'LGS AUSTRALIAN EQUITIES - ECP': 'ECP',
    'LGS AUSTRALIAN EQUITIES - SRI UBS': 'DSRI'
}

# Selects Australian Equities
df_jpm_ae = df_jpm[df_jpm['Account Name'].isin(jpm_account_name_to_manager_name_dict)].reset_index(drop=True)

# Creates Asset Class column
df_jpm_ae['Asset Class'] = 'Australian Equity'

# Creates Manager Name column
df_jpm_ae['Manager Name'] = [
    jpm_account_name_to_manager_name_dict[df_jpm_ae['Account Name'][i]]
    for i in range(0, len(df_jpm_ae))
]

# Creates Report Date column
df_jpm_ae['Report Date'] = jpm_report_date

# Selects columns
jpm_column_list = [
    'Asset Class',
    'Manager Name',
    'Report Date',
    'SEDOL',
    'ISIN',
    'Security Name',
    'Currency',
    'Quantity',
    'Purchase Price',
    'Market Value'
]
df_jpm_ae = df_jpm_ae[jpm_column_list]

# Loads the CFS file
df_cfs = pd.read_excel(cfs_directory + cfs_filename, header=8)

df_cfs = df_cfs.rename(
    columns={
        'Security ISIN': 'ISIN',
        'Security CUSIP': 'CUSIP',
        'Security SEDOL': 'SEDOL',
        'Security Name': 'Security Name',
        'Sector': 'Sector',
        'Country': 'Country',
        'Country ISO Code': 'Country ISO',
        'Security Currency': 'Currency',
        'Market Price (Local Currency)': 'Purchase Price',
        'Stocks in Portfolio': 'Stocks in Portfolio',
        'Market Value (Base Currency)': 'Market Value',
        'Market Value (Local Currency)': 'Market Value (Local Currency)',
        'Maket Value (Calculated)': 'Maket Value (Calculated)',
        'Market Value (Local Calculated)': 'Market Value (Local Calculated)',
        'Portfolio Weight': 'Portfolio Weight',
        'Unit Holdings': 'Quantity',
        'Coupon': 'Coupon',
        'Strike Price': 'Strike Price',
        'Maturity Date': 'Maturity Date',
        'Position Cost (Base Currency)': 'Position Cost (Base Currency)',
        'Position Cost (Local Currency)': 'Position Cost (Local Currency)',
        'Custom Sector Code': 'Custom Sector Code',
        'Custom Region/Country Code': 'Custom Region/Country Code'

    }
)
# Rescales market values and quantity
cfs_scaling_ratio = cfs_market_value/df_cfs['Market Value'].sum()
df_cfs['Market Value'] = df_cfs['Market Value'] * cfs_scaling_ratio
df_cfs['Quantity'] = df_cfs['Quantity'] * cfs_scaling_ratio

df_cfs['Asset Class'] = 'Australian Equity'
df_cfs['Manager Name'] = 'CFS'
df_cfs['Report Date'] = cfs_report_date

# Selects CFS columns
cfs_columns_list = [
    'Asset Class',
    'Manager Name',
    'Report Date',
    'SEDOL',
    'ISIN',
    'Security Name',
    'Currency',
    'Quantity',
    'Purchase Price',
    'Market Value'
]
df_cfs = df_cfs[cfs_columns_list]

# Totals the market value
df_ae = pd.concat([df_jpm_ae, df_cfs]).reset_index(drop=True)

ae_market_value = df_ae['Market Value'].sum()


def total(data):
    d = dict()
    d['Manager Market Value'] = np.sum(data['Market Value'])
    return pd.Series(d)


df_ae_totals = df_ae.groupby(['Asset Class', 'Manager Name', 'Report Date']).apply(total)

df_ae_totals['Asset Class Market Value'] = ae_market_value

df_ae_totals['Proportion of Asset Class'] = df_ae_totals['Manager Market Value']/ae_market_value

df_ae_totals = df_ae_totals.reset_index(drop=False)

manager_name_to_target_allocation_dict = {
    'Pendal': 0.16,
    'Ubique': 0.12,
    'Blackrock': 0.23,
    'DNR': 0.14,
    'SSGA': 0.23,
    'ECP': 0.04,
    'DSRI': 0.01,
    'CFS': 0.08
}

df_ae_totals['Target Allocation'] = [
    manager_name_to_target_allocation_dict[df_ae_totals['Manager Name'][i]]
    for i in range(0, len(df_ae_totals))
]

writer = pd.ExcelWriter(output_directory + output_filename, engine='xlsxwriter')

df_ae.to_excel(
    writer,
    sheet_name='manager_holdings',
    index=False
)

df_ae_totals.to_excel(
    writer,
    sheet_name='summary_manager_holdings',
    index=False
)

writer.save()
