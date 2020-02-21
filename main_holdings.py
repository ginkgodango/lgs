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

# Begin User Input Data
report_date = dt.datetime(2020, 1, 31)

wscf_market_value = 188197744.70
aqr_market_value = 223033840.40
delaware_market_value = 183241040.60
wellington_market_value = 183357208.68

jpm_filepath = 'U:/CIO/#Holdings/Data/input/holdings/jpm/2020/01/Priced Positions - All.csv'
wscf_filepath = 'U:/CIO/#Holdings/Data/input/holdings/unitprices/2020/01/wscf_holdings.xls'
aqr_filepath = 'U:/CIO/#Holdings/Data/input/holdings/unitprices/2020/01/aqr_holdings.xls'
delaware_filepath = 'U:/CIO/#Holdings/Data/input/holdings/unitprices/2020/01/delaware_holdings.xlsx'
wellington_filepath = 'U:/CIO/#Holdings/Data/input/holdings/unitprices/2020/01/wellington_holdings.xlsx'
tickers_filepath = 'U:/CIO/#Holdings/Data/input/tickers/tickers_201909.xlsx'
asx_filepath = 'D:/data/ASX List/ASX300/20191201-asx300.csv'

aeq_filepath = 'U:/CIO/#Holdings/Data/input/exclusions/LGS Exclusions List_December 2018_AEQ_Manager Version.xlsx'
ieq_filepath = 'U:/CIO/#Holdings/Data/input/exclusions/LGS Exclusions List_December 2018_IEQ_Manager Version.xlsx'
aeq_exclusions_filepath = 'U:/CIO/#Holdings/Data/output/exclusions/aeq_exclusions_' + str(report_date.date()) + '.csv'
ieq_exclusions_filepath = 'U:/CIO/#Holdings/Data/output/exclusions/ieq_exclusions_' + str(report_date.date()) + '.csv'
# End User Input Data

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
                'Total Market Value (Local)',
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
                'Market Price': 'Purchase Price Local',
                'Total Units': 'Quantity',
                'Total Market Value (Local)': 'Market Value Local',
                'Total Market Value (Base)': 'Market Value AUD',
                'Local Currency': 'Currency'
        }
)

df_jpm['Purchase Price AUD'] = df_jpm['Market Value AUD'] / df_jpm['Quantity']

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
                'Market Value (Base Currency)',
                'Security Currency'
        ]
)

df_wscf = df_wscf.rename(
        columns={
                'Security SEDOL': 'SEDOL',
                'Security ISIN': 'ISIN',
                'Unit Holdings': 'Quantity',
                'Market Value (Local Currency)': 'Market Value Local',
                'Market Value (Base Currency)': 'Market Value AUD',
                'Security Currency': 'Currency'
        }
)

wscf_scaling_factor = wscf_market_value/df_wscf['Market Value AUD'].sum()
df_wscf['Market Value Local'] = wscf_scaling_factor * df_wscf['Market Value Local']
df_wscf['Market Value AUD'] = wscf_scaling_factor * df_wscf['Market Value AUD']
df_wscf['Quantity'] = wscf_scaling_factor * df_wscf['Quantity']
df_wscf['Purchase Price Local'] = df_wscf['Market Value Local'] / df_wscf['Quantity']
df_wscf['Purchase Price AUD'] = df_wscf['Market Value AUD'] / df_wscf['Quantity']
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
                'Price Local',
                'Base Price',
                'Quantity',
                'MV Local',
                'MV Base',
                'Ccy'
        ]
)

df_aqr = df_aqr.rename(
        columns={
                'Sedol': 'SEDOL',
                'Isin': 'ISIN',
                'Investment Description': 'Security Name',
                'Price Local': 'Purchase Price Local',
                'Base Price': 'Purchase Price AUD',
                'MV Local': 'Market Value Local',
                'MV Base': 'Market Value AUD',
                'Ccy': 'Currency'
        }
)

aqr_scaling_factor = aqr_market_value/df_aqr['Market Value AUD'].sum()
df_aqr['Market Value Local'] = aqr_scaling_factor * df_aqr['Market Value Local']
df_aqr['Market Value AUD'] = aqr_scaling_factor * df_aqr['Market Value AUD']
df_aqr['Quantity'] = aqr_scaling_factor * df_aqr['Quantity']
df_aqr['Account Number'] = 'AQR'
df_aqr['Account Name'] = 'LGS INTERNATIONAL EQUITIES - AQR'
df_aqr['Date'] = report_date


# Imports Delaware holdings data
df_delaware = pd.read_excel(
        pd.ExcelFile(delaware_filepath),
        sheet_name='EM UCITS holdings 11-30-19',
        header=0,
        usecols=[
                'Security SEDOL',
                'Security ISIN',
                'Security Description (Short)',
                'Position Date',
                'Shares/Par',
                'Trading Currency',
                'Traded Market Value (Local)',
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
                'Traded Market Value (Local)': 'Market Value Local',
                'Traded Market Value (AUD)': 'Market Value AUD'
        }
)

delaware_scaling_factor = delaware_market_value/df_delaware['Market Value AUD'].sum()
df_delaware['Market Value Local'] = delaware_scaling_factor * df_delaware['Market Value Local']
df_delaware['Market Value AUD'] = delaware_scaling_factor * df_delaware['Market Value AUD']
df_delaware['Quantity'] = delaware_scaling_factor * df_aqr['Quantity']
df_delaware['Purchase Price Local'] = df_delaware['Market Value Local'] / df_delaware['Quantity']
df_delaware['Purchase Price AUD'] = df_delaware['Market Value AUD'] / df_delaware['Quantity']
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
                'Market Value (Local)',
                'Market Value (Report Currency)'
        ]
)

df_wellington = df_wellington.rename(
        columns={
                'Security': 'Security Name',
                'Shares or Par Value': 'Quantity',
                'ISO Code': 'Currency',
                'Market Value (Local)': 'Market Value Local',
                'Market Value (Report Currency)': 'Market Value AUD'
        }
)

wellington_scaling_factor = wellington_market_value/df_wellington['Market Value AUD'].sum()
df_wellington['Market Value Local'] = wellington_scaling_factor * df_wellington['Market Value Local']
df_wellington['Market Value AUD'] = wellington_scaling_factor * df_wellington['Market Value AUD']
df_wellington['Quantity'] = wellington_scaling_factor * df_wellington['Quantity']
df_wellington['Purchase Price Local'] = df_wellington['Market Value Local'] / df_wellington['Quantity']
df_wellington['Purchase Price AUD'] = df_wellington['Market Value AUD'] / df_wellington['Quantity']
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


# TOP HOLDINGS SECTION
australian_equity_managers_list = [
        'LGS AUSTRALIAN EQUITIES - BLACKROCK',
        'LGS AUSTRALIAN EQUITIES - ECP',
        'LGS AUSTRALIAN EQUITIES DNR CAPITAL',
        'LGS AUSTRALIAN EQUITIES - PENDAL',
        'LGS AUSTRALIAN EQUITIES - SSGA',
        'LGS AUSTRALIAN EQUITIES - UBIQUE',
        'LGS AUSTRALIAN EQUITIES - WSCF',
        'LGS AUSTRALIAN EQUITIES REBALANCE',
]
international_equity_managers_list = [
        'LGS INTERNATIONAL EQUITIES - WCM',
        'LGS INTERNATIONAL EQUITIES - AQR',
        'LGS INTERNATIONAL EQUITIES - HERMES',
        'LGS INTERNATIONAL EQUITIES - IMPAX',
        'LGS INTERNATIONAL EQUITIES - LONGVI EW',
        'LGS INTERNATIONAL EQUITIES - LSV',
        'LGS INTERNATIONAL EQUITIES - MFS',
        'LGS INTERNATIONAL EQUITIES - DELAWARE',
        'LGS INTERNATIONAL EQUITIES - WELLINGTON',
        'LGS GLOBAL LISTED PROPERTY - RESOLU TION',
]

df_main_aeq = df_main[df_main['Account Name'].isin(australian_equity_managers_list)].reset_index(drop=True)
df_main_ieq = df_main[df_main['Account Name'].isin(international_equity_managers_list)].reset_index(drop=True)

df_main_aeq['(%) of Portfolio'] = (df_main_aeq['Market Value AUD'] / df_main_aeq['Market Value AUD'].sum()) * 100
df_main_ieq['(%) of Portfolio'] = (df_main_ieq['Market Value AUD'] / df_main_ieq['Market Value AUD'].sum()) * 100

a = df_main_aeq['Market Value AUD'].sum()

df_main_aeq = df_main_aeq.groupby(['SEDOL']).sum().sort_values(['Market Value AUD'], ascending=[False])[['Market Value AUD', '(%) of Portfolio']]
df_main_ieq = df_main_ieq.groupby(['SEDOL']).sum().sort_values(['Market Value AUD'], ascending=[False])[['Market Value AUD', '(%) of Portfolio']]

df_security_names = df_main[['SEDOL', 'Security Name']].drop_duplicates(subset=['SEDOL'], keep='first').reset_index(drop=True)

df_main_aeq = pd.merge(
        left=df_main_aeq,
        right=df_security_names,
        left_on=['SEDOL'],
        right_on=['SEDOL'],
        how='outer',
        indicator=True
)
df_main_aeq = df_main_aeq[df_main_aeq['_merge'].isin(['left_only', 'both'])].drop(columns=['_merge'], axis=1)

df_main_ieq = pd.merge(
        left=df_main_ieq,
        right=df_security_names,
        left_on=['SEDOL'],
        right_on=['SEDOL'],
        how='outer',
        indicator=True
)
df_main_ieq = df_main_ieq[df_main_ieq['_merge'].isin(['left_only', 'both'])].drop(columns=['_merge'], axis=1)

sedol_to_common_name_dict = {
        '6215035': 'CBA',
        '6144690': 'BHP',
        '6185495': 'CSL',
        '6624608': 'NAB',
        '6076146': 'Westpac',
        'B28YTC2': 'Macquarie',
        '6065586': 'ANZ',
        '6087289': 'Telstra',
        '6948836': 'Westfarmers',
        '6220103': 'Rio Tinto',
        '6981239': 'Woolworths',
        'BTN1Y11': 'Medtronic',
        'B2PZN04': 'Visa',
        '2661568': 'Oracle',
        '2886907': 'Thermo Fisher',
        '2842040': 'State Street',
        'B4BNMY3': 'Accenture',
        '2044545': 'Comcast',
        '2270726': 'Walt Disney',
        'BD6K457': 'Compass',
        '2210959': 'Canadian Rail',
        '7123870': 'Nestle',
        '2588173': 'Microsoft',
        'B4MGBG6': 'HCA'
}

df_main_aeq_top10 = df_main_aeq.head(10)[['SEDOL', 'Market Value AUD', '(%) of Portfolio']]
df_main_ieq_top10 = df_main_ieq.head(10)[['SEDOL', 'Market Value AUD', '(%) of Portfolio']]

df_main_aeq_top10['Company'] = [sedol_to_common_name_dict[df_main_aeq_top10['SEDOL'][i]] for i in range(0, len(df_main_aeq_top10))]
df_main_ieq_top10['Company'] = [sedol_to_common_name_dict[df_main_ieq_top10['SEDOL'][i]] for i in range(0, len(df_main_ieq_top10))]

df_main_aeq_top10['Market Value'] = df_main_aeq_top10['Market Value AUD'] / 1000000
df_main_ieq_top10['Market Value'] = df_main_ieq_top10['Market Value AUD'] / 1000000

df_main_aeq_top10 = df_main_aeq_top10[['Company', 'Market Value', '(%) of Portfolio']].round(2)
df_main_ieq_top10 = df_main_ieq_top10[['Company', 'Market Value', '(%) of Portfolio']].round(2)

with open('U:/CIO/#Investment_Report/Data/output/holdings/top10_local.tex', 'w') as tf:
    tf.write(df_main_aeq_top10.to_latex(index=False))

with open('U:/CIO/#Investment_Report/Data/output/holdings/top10_foreign.tex', 'w') as tf:
    tf.write(df_main_ieq_top10.to_latex(index=False))

# Writes to excel
writer = pd.ExcelWriter('U:/CIO/#Holdings/Data/output/holdings/top_holdings.xlsx', engine='xlsxwriter')
df_main_aeq.to_excel(writer, sheet_name='local', index=False)
df_main_ieq.to_excel(writer, sheet_name='foreign', index=False)
writer.save()


# EXCLUSIONS SECTION
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


# YAHOO SECTION
# Imports
df_tickers = pd.read_excel(
        pd.ExcelFile(tickers_filepath),
        sheet_name='Sheet1',
        header=0,
        usecols=[
                'SEDOL',
                'ISIN',
                'Security Name',
                'Currency',
                'PRIMARY_EXCHANGE_NAME',
                'TICKER',
        ]
)
df_tickers['SEDOL'] = [str(df_tickers['SEDOL'][i]).replace(" ", "").upper() for i in range(0, len(df_tickers))]
df_tickers['ISIN'] = [str(df_tickers['ISIN'][i]).replace(" ", "").upper() for i in range(0, len(df_tickers))]
df_tickers['TICKER'] = [str(df_tickers['TICKER'][i]).replace(" ", "").upper() for i in range(0, len(df_tickers))]

new_ticker_codes = []
df_tickers['TICKER'] = df_tickers['TICKER'].astype(str)
for i in range(0, len(df_tickers)):
    current_ticker = df_tickers['TICKER'][i]
    if df_tickers['Currency'][i] == 'HKD':
        add_zeros = '0' * (4 - len(current_ticker))
        new_ticker_code = add_zeros + current_ticker
        new_ticker_codes.append(new_ticker_code.replace('/', ''))

    else:
        new_ticker_codes.append(current_ticker.replace('/', ''))
df_tickers['TICKER'] = new_ticker_codes

# Selects equity managers only
equity_managers_list = [
        'LGS AUSTRALIAN EQUITIES - BLACKROCK',
        'LGS AUSTRALIAN EQUITIES - ECP',
        'LGS AUSTRALIAN EQUITIES DNR CAPITAL',
        'LGS AUSTRALIAN EQUITIES - PENDAL',
        'LGS AUSTRALIAN EQUITIES - SSGA',
        'LGS AUSTRALIAN EQUITIES - UBIQUE',
        'LGS INTERNATIONAL EQUITIES - WCM',
        'LGS AUSTRALIAN EQUITIES - WSCF',
        'LGS AUSTRALIAN EQUITIES REBALANCE',
        'LGS INTERNATIONAL EQUITIES - AQR',
        'LGS INTERNATIONAL EQUITIES - HERMES',
        'LGS INTERNATIONAL EQUITIES - IMPAX',
        'LGS INTERNATIONAL EQUITIES - LONGVI EW',
        'LGS INTERNATIONAL EQUITIES - LSV',
        'LGS INTERNATIONAL EQUITIES - MFS',
        'LGS INTERNATIONAL EQUITIES - DELAWARE',
        'LGS INTERNATIONAL EQUITIES - WELLINGTON',
        'LGS GLOBAL LISTED PROPERTY - RESOLU TION',
]
df_main_eq = df_main[df_main['Account Name'].isin(equity_managers_list)].reset_index(drop=True)

# Merges the equity and ticker dataframe
df_main_eq_ticker = pd.merge(
        left=df_main_eq,
        right=df_tickers,
        left_on=['SEDOL', 'ISIN'],
        right_on=['SEDOL', 'ISIN'],
        how='outer',
        indicator=True
)

# Finds missing SEDOL in the TICKER file
df_main_eq_ticker_left = (
        df_main_eq_ticker[df_main_eq_ticker['_merge'].isin(['left_only'])]
        [['Account Name', 'Security Name_x', 'Market Value AUD', 'ISIN', 'SEDOL', 'TICKER', 'Currency_x', 'PRIMARY_EXCHANGE_NAME']]
        .rename(columns={'Security Name_x': 'Security Name', 'Currency_x': 'Currency'})
        .sort_values(['Account Name', 'ISIN'])
        .reset_index(drop=True)
)

df_main_eq_ticker_left.to_csv('U:/CIO/#Holdings/Data/output/missing/missing.csv', index=False)

# Creates Yahoo Upload File
df_main_eq_ticker_both = df_main_eq_ticker[df_main_eq_ticker['_merge'].isin(['both'])].sort_values(['Account Name']).reset_index(drop=True)

exchange_yahoo_suffix_dict = {
        '#N/A Invalid Security': '',
        '#N/A Field Not Applicable': '',
        'ASE': '.AX',
        'Athens': '.AT',
        'B3 Day': '.SA',
        'Bangkok': '.BK',
        'BMV Mexico': '.MX',
        'BrsaItaliana': '.IM',
        'Budapest': '.BD',
        'Bursa Malays': '.KL',
        'Copenhagen': '.CO',
        'Dublin': '.IR',
        'EN Amsterdam': '.AS',
        'EN Brussels': '.BR',
        'EN Lisbon': '.LS',
        'EN Paris': '.PA',
        'Helsinki': '.HE',
        'Hong Kong': '.HK',
        'Indonesia': '.JK',
        'Istanbul': '.IS',
        'Johannesburg': '.JO',
        'Korea SE': '.KS',
        'London': '.L',
        'London Intl': '.IL',
        'MICEX Main': '.ME',
        'MOEX': '.ME',
        'NASDAQ GM': '',
        'NASDAQ GS': '',
        'NZX': '.NZ',
        'Natl India': '.NS',
        'New York': '',
        'Oslo': '.OL',
        'Philippines': '',
        'SIX Swiss Ex': '.SW',
        'Sant. Comerc': '.SN',
        'Singapore': '.SI',
        'Shanghai': '.SS',
        'Shenzhen': '.SZ',
        'Soc.Bol SIBE': '.MC',
        'Stockholm': '.ST',
        'Taiwan': '.TW',
        'Tel Aviv': '.TA',
        'Tokyo': '.T',
        'Toronto': '.TO',
        'Venture': '.V',
        'Vienna': '.VI',
        'Warsaw': '',
        'Xetra': '.DE',
        'nan': '',
        np.nan: ''
        }

df_main_eq_ticker_both['Suffix'] = [
        exchange_yahoo_suffix_dict[df_main_eq_ticker_both['PRIMARY_EXCHANGE_NAME'][i]]
        for i in range(0, len(df_main_eq_ticker_both))
        ]

df_main_eq_ticker_both['Symbol'] = df_main_eq_ticker_both['TICKER'] + df_main_eq_ticker_both['Suffix']

# Adjusts purchase price for Yahoo format
purchase_prices = []
for i in range(0, len(df_main_eq_ticker_both)):
    purchase_price = df_main_eq_ticker_both['Purchase Price Local'][i]
    if df_main_eq_ticker_both['PRIMARY_EXCHANGE_NAME'][i] == 'London':
        purchase_price = df_main_eq_ticker_both['Purchase Price Local'][i] * 100
    purchase_prices.append(purchase_price)
df_main_eq_ticker_both['Purchase Price Local'] = purchase_prices

df_yahoo = (
        df_main_eq_ticker_both[['Account Name', 'Symbol', 'Purchase Price Local', 'Quantity']]
        .rename(columns={'Purchase Price Local': 'Purchase Price'})
        .sort_values(['Account Name', 'Symbol'], ascending=[True, False])
        .reset_index(drop=True)
)

columns_yahoo = [
        'Symbol',
        'Current Price',
        'Date',
        'Time',
        'Change',
        'Open',
        'High',
        'Low',
        'Volume',
        'Trade Date',
        'Purchase Price',
        'Quantity',
        'Commission',
        'High Limit',
        'Low Limit',
        'Comment'
        ]

for column in columns_yahoo:
    if column not in df_yahoo.columns:
        df_yahoo[column] = np.nan

df_yahoo = df_yahoo[['Account Name'] + columns_yahoo]

account_to_df_dict = dict(list(df_yahoo.groupby(['Account Name'])))

for account, df in account_to_df_dict.items():
    df = df[columns_yahoo]
    df.to_csv('U:/CIO/#Holdings/Data/output/yahoo/' + account + '.csv', index=False)


# Renames columns
df_main_eq_ticker = df_main_eq_ticker.rename(columns={'Security Name_x': 'Security Name'})

# Calculates Relative
df_main_relative = df_main_eq_ticker[['Account Name', 'TICKER', 'Security Name', 'Market Value AUD']]
df_main_relative_aeq = df_main_relative[df_main_relative['Account Name'].isin(australian_equity_managers_list)]
df_main_relative_aeq.drop(columns=['Account Name'], axis=1)
df_main_relative_aeq = df_main_relative_aeq.groupby(['TICKER']).sum().reset_index(drop=False)
df_main_relative_aeq['Portfolio Weight'] = ((df_main_relative_aeq['Market Value AUD'] / a)*100).round(2)
df_main_relative_aeq = df_main_relative_aeq.sort_values(['Portfolio Weight'], ascending=[False])

df_asx = pd.read_csv(
        asx_filepath,
        skiprows=[0],
        header=0,
        usecols=[
                'Code',
                'Company',
                'Weight(%)'
        ],
)

df_asx = df_asx.rename(columns={'Weight(%)': 'Benchmark Weight'})

df_main_relative_aeq = pd.merge(
        left=df_main_relative_aeq,
        right=df_asx,
        left_on=['TICKER'],
        right_on=['Code'],
        how='outer'
)

df_main_relative_aeq['Relative Weight'] = df_main_relative_aeq['Portfolio Weight'] - df_main_relative_aeq['Benchmark Weight']
df_main_relative_aeq.to_csv('U:/CIO/#Holdings/Data/output/relative_holdings_aeq.csv', index=False)
