import datetime as dt
import numpy as np
import pandas as pd
import os


def read_jpm_dvr(s):

    return pd.read_csv(s, skiprows=[0, 1, 2, 3])


def process_jpm_dvr(df):

    return df


def read_jpm_pp(s):

    return pd.read_csv(s, skiprows=[0, 1, 2, 3])


def process_jpm_pp(df):

    return df[['Security ID', 'ISIN', 'Market Price']].drop_duplicates().reset_index(drop=True)


def read_fsi(s):

    return pd.read_excel(
        pd.ExcelFile(s),
        sheet_name='Holdings',
        skiprows=[0, 1, 2, 3, 4, 5, 6, 8],
        header=0,
        usecols=[
            'Security SEDOL',
            'Security ISIN',
            'Security Name',
            'Market Price (Local Currency)',
            'Unit Holdings',
            'Market Value (Base Currency)',
            'Security Currency'
        ]
    )


def process_fsi(df, mv, columns):

    df = df.rename(
            columns={
                'Security SEDOL': 'Security ID',
                'Security ISIN': 'ISIN',
                'Security Name': 'Security Name',
                'Market Price (Local Currency)': 'Market Price',
                'Unit Holdings': 'Unit Holding',
                'Market Value (Base Currency)': 'Realizable Value Base',
                'Security Currency': 'Local Currency'
            }
        )

    rescale_ratio = mv / df['Realizable Value Base'].sum()
    df['Realizable Value Base'] *= rescale_ratio
    df['Unit Holding'] *= rescale_ratio
    df['Portfolio Name'] = 'LGS AE FSI'
    df['Category Description'] = np.nan

    return df[columns]


def read_aqr(s):

    return pd.read_excel(
        pd.ExcelFile(s),
        sheet_name='Holdings',
        skiprows=[0, 1, 2, 3, 4, 5, 6, 7],
        header=0,
        usecols=[
            'Sedol',
            'Isin',
            'Investment Description',
            'Price Local',
            'Quantity',
            'MV Base',
            'Ccy'
        ]
    )


def process_aqr(df, mv, columns):

    df = df.rename(columns={
        'Asset Type': 'Category Description',
        'Sedol': 'Security ID',
        'Isin': 'ISIN',
        'Investment Description': 'Security Name',
        'Price Local': 'Market Price',
        'Quantity': 'Unit Holding',
        'MV Base': 'Realizable Value Base',
        'Ccy': 'Local Currency'
    })

    rescale_ratio = mv / df['Realizable Value Base'].sum()
    df['Unit Holding'] *= rescale_ratio
    df['Realizable Value Base'] *= rescale_ratio
    df['Portfolio Name'] = 'LGS IE AQR'
    df['Category Description'] = np.nan

    return df[columns]


def read_mac(s):

    return pd.read_excel(
        pd.ExcelFile(s),
        sheet_name='EM SICAV holdings',
        header=0,
        usecols=[
                'Security SEDOL',
                'Security ISIN',
                'Security Description (Short)',
                'Price (Local)',
                'Shares/Par',
                'Trading Currency',
                'Traded Market Value (AUD)'
        ]
)


def process_mac(df, mv, columns):

    df = df.rename(columns={
        'Security SEDOL': 'Security ID',
        'Security ISIN': 'ISIN',
        'Security Description (Short)': 'Security Name',
        'Price (Local)': 'Market Price',
        'Shares/Par': 'Unit Holding',
        'Trading Currency': 'Local Currency',
        'Traded Market Value (AUD)': 'Realizable Value Base'
    })

    rescale_ratio = mv / df['Realizable Value Base'].sum()
    df['Realizable Value Base'] *= rescale_ratio
    df['Unit Holding'] *= rescale_ratio
    df['Portfolio Name'] = 'LGS IE MAC'
    df['Category Description'] = np.nan

    return df[columns]


def read_wel(s):

    return pd.read_excel(
        pd.ExcelFile(s),
        sheet_name='wel_holdings',
        header=0,
        usecols=[
                'SEDOL',
                'ISIN',
                'Security',
                'Unit Price (Local)',
                'Shares or Par Value',
                'Report Currency',
                'Market Value (Report Currency)',
        ]
)


def process_wel(df, mv, columns):

    df = df.rename(
        columns={
            'SEDOL': 'Security ID',
            'ISIN': 'ISIN',
            'Security': 'Security Name',
            'Unit Price (Local)': 'Market Price',
            'Shares or Par Value': 'Unit Holding',
            'Report Currency': 'Local Currency',
            'Market Value (Report Currency)': 'Realizable Value Base'
        }
    )

    rescale_ratio = mv / df['Realizable Value Base'].sum()
    df['Realizable Value Base'] *= rescale_ratio
    df['Unit Holding'] *= rescale_ratio
    df['Portfolio Name'] = 'LGS IE WEL'
    df['Category Description'] = np.nan

    return df[columns]


def read_tic(s):
    return pd.read_excel(
        pd.ExcelFile(s)
    )


def swap(a, b, condition):

    return a if condition else b


def ric_to_symbol(s):

    return (
        s[:-2] if s.endswith('.N') else
        s[:-3] if s.endswith('.OQ') else
        s[:-2] + '.SW' if s.endswith('.S') else
        s[:-4] + '-A.ST' if s.endswith('a.ST') else
        s[:-4] + '-B.ST' if s.endswith('b.ST') else
        s
    )


columns_list = [
    'Portfolio Name',
    'Category Description',
    'Local Currency',
    'Security ID',
    'ISIN',
    'Security Name',
    'Market Price',
    'Unit Holding',
    'Realizable Value Base'
]

swap_list = [
    'Future',
    'Liquidity',
    'Forward Foreign Exchange'
]

ae_list = [
    'LGS AE ALPH',
    'LGS AE AUS SRI',
    'LGS AE BLCKROCK',
    'LGS AE DNR',
    'LGS AE FSI',
    'LGS AE RE BT',
    'LGS AE RE ECP',
    'LGS AE UBIQUE'
]

ie_list = [
    'LGS IE AQR',
    'LGS IE HERMES',
    'LGS IE IMPAX',
    'LGS IE LONGVIEW',
    'LGS IE LSV',
    'LGS IE MAC',
    'LGS IE MFS',
    'LGS IE TM MAC19',
    'LGS IE UBS',
    'LGS IE WCM',
    'LGS IE WEL'
]

yahoo_columns = [
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

if __name__ == '__main__':

    # Set file directories.
    jpm_dvr_path = 'C:/Users/Mnguyen/LGSS/Investments Team - SandPits - SandPits/data/input/vendors/jpm/markets/investment_accounting/2021/02/Detailed Valuation Report - Equities.csv'
    jpm_pp_path = 'C:/Users/Mnguyen/LGSS/Investments Team - SandPits - SandPits/data/input/vendors/jpm/markets/custody/2021/02/Priced Positions - All.csv'
    fsi_path = 'C:/Users/Mnguyen/LGSS/Investments Team - SandPits - SandPits/data/input/managers/2021/02/fsi_holdings.xlsx'
    aqr_path = 'C:/Users/Mnguyen/LGSS/Investments Team - SandPits - SandPits/data/input/managers/2021/01/aqr_holdings.xlsx'
    mac_path = 'C:/Users/Mnguyen/LGSS/Investments Team - SandPits - SandPits/data/input/managers/2021/02/mac_holdings.xlsx'
    wel_path = 'C:/Users/Mnguyen/LGSS/Investments Team - SandPits - SandPits/data/input/managers/2021/02/wel_holdings.xlsx'
    ric_path = 'C:/Users/Mnguyen/LGSS/Investments Team - SandPits - SandPits/data/input/lgs/isin_ric.csv'

    # Get market value from JPM Investment Accounting System.
    date = dt.datetime(2021, 2, 28)
    fsi_mv = 228147267.6
    # aqr_mv = 178698409.64
    mac_mv = 176133018.68
    wel_mv = 181523880.42

    # Reads in each file as a dataframe and cleans it.
    df_jpm_dvr = process_jpm_dvr(df=read_jpm_dvr(jpm_dvr_path))
    df_jpm_pp = process_jpm_pp(df=read_jpm_pp(jpm_pp_path))
    df_fsi = process_fsi(df=read_fsi(fsi_path), mv=fsi_mv, columns=columns_list)
    # df_aqr = process_aqr(df=read_aqr(aqr_path), mv=aqr_mv, columns=columns_list)
    df_mac = process_mac(df=read_mac(mac_path), mv=mac_mv, columns=columns_list)
    df_wel = process_wel(df=read_wel(wel_path), mv=wel_mv, columns=columns_list)
    df_ric = pd.read_csv(ric_path)

    # Merges ISINs onto SEDOLs
    df_jpm_merge = pd.merge(left=df_jpm_dvr, right=df_jpm_pp, on=['Security ID'], how='outer', indicator=True)
    df_jpm_final = df_jpm_merge[~df_jpm_merge['_merge'].isin(['right_only'])][columns_list]

    # Joins all files into one dataframe.
    df_all = pd.concat([df_jpm_final, df_fsi, df_mac, df_wel], axis=0).sort_values('Portfolio Name').reset_index(drop=True)

    # Swaps the Security IDs and ISINs of the Liquidity accounts with their Security Names. This solves the uniqueness problem.
    df_all['Security ID'] = [str(swap(df_all['Security Name'][i], df_all['Security ID'][i], df_all['Category Description'][i] in ['Liquidity'])) for i in range(len(df_all))]
    # df_all['ISIN'] = [str(swap(df_all['Security Name'][i], df_all['ISIN'][i], df_all['Category Description'][i] in swap_list))for i in range(len(df_all))]

    # TOP HOLDINGS AUSTRALIAN EQUITY AND INTERNATIONAL EQUITY
    # Creates df_info to merge back the Security Names after aggregating on Security ID.
    df_info = df_all[['Security ID', 'Security Name']].drop_duplicates(subset='Security ID', keep="first")

    # Sums Realizable Value Base by Security ID
    df_ae = df_all[df_all['Portfolio Name'].isin(ae_list)].groupby('Security ID').sum()[['Realizable Value Base']].reset_index(drop=False).sort_values('Realizable Value Base', ascending=False)
    df_ie = df_all[df_all['Portfolio Name'].isin(ie_list)].groupby('Security ID').sum()[['Realizable Value Base']].reset_index(drop=False).sort_values('Realizable Value Base', ascending=False)

    # Calculates the percentage of portfolio for each Security ID
    df_ae['% of Portfolio'] = (df_ae['Realizable Value Base'] / df_ae['Realizable Value Base'].sum()) * 100
    df_ie['% of Portfolio'] = (df_ie['Realizable Value Base'] / df_ie['Realizable Value Base'].sum()) * 100

    # Merges the Security Name back onto the Security ID
    df_ae_info = pd.merge(left=df_info, right=df_ae, on=['Security ID'], how='inner').sort_values('Realizable Value Base', ascending=False).reset_index(drop=True)
    df_ie_info = pd.merge(left=df_info, right=df_ie, on=['Security ID'], how='inner').sort_values('Realizable Value Base', ascending=False).reset_index(drop=True)

    # Writes to Excel.
    writer = pd.ExcelWriter('C:/Users/Mnguyen/LGSS/Investments Team - SandPits - SandPits/data/output/lgs/holdings/portfolio_holdings.xlsx', engine='xlsxwriter')
    df_ae_info.to_excel(writer, sheet_name='AE', index=False)
    df_ie_info.to_excel(writer, sheet_name='IE', index=False)
    df_all.to_excel(writer, sheet_name='All', index=False)
    writer.save()

    # YAHOO PROCESSING
    # Converts columns to string
    df_ric['ISIN'] = [str(df_ric['ISIN'][i]) for i in range(len(df_ric))]
    df_ric['Symbol'] = [str(ric_to_symbol(df_ric['RIC'][i])) for i in range(len(df_ric))]

    # Merges the ISIN with RIC for Yahoo
    df_all_isin = df_all[~df_all['ISIN'].isin([np.nan])].reset_index(drop=True)
    df_yahoo_merge = pd.merge(left=df_all_isin, right=df_ric, on=['ISIN'], how='outer', indicator=True).sort_values(['Portfolio Name', 'RIC'])
    df_yahoo_final1 = df_yahoo_merge[~df_yahoo_merge['_merge'].isin(['right_only'])].reset_index(drop=True)
    df_yahoo_missing = df_yahoo_merge[df_yahoo_merge['_merge'].isin(['left_only'])].reset_index(drop=True)

    # Formats for Yahoo Schema
    df_yahoo_final2 = df_yahoo_final1.rename(columns={
        'Market Price': 'Purchase Price',
        'Unit Holding': 'Quantity'
    })

    yahoo_price = []
    for i in range(len(df_yahoo_final2)):
        if df_yahoo_final2['Symbol'][i] is not np.nan and isinstance(df_yahoo_final2['Symbol'][i], str):
            if df_yahoo_final2['Symbol'][i][-2:] == '.L':
                print(df_yahoo_final2['Symbol'][i])
                yahoo_price.append(df_yahoo_final2['Purchase Price'][i]*100)
            else:
                yahoo_price.append(df_yahoo_final2['Purchase Price'][i])
        else:
            yahoo_price.append(df_yahoo_final2['Purchase Price'][i])
    df_yahoo_final2['Purchase Price'] = yahoo_price

    for column in yahoo_columns:
        if column not in df_yahoo_final2.columns:
            df_yahoo_final2[column] = np.nan

    df_yahoo_final3 = df_yahoo_final2[['Portfolio Name'] + yahoo_columns]
    portfolio_to_df_dict = dict(list(df_yahoo_final3.groupby(['Portfolio Name'])))
    for portfolio, df in portfolio_to_df_dict.items():
        df = df.sort_values('Symbol', ascending=False)
        df.to_csv('C:/Users/Mnguyen/LGSS/Investments Team - SandPits - SandPits/data/output/lgs/holdings/yahoo/' + portfolio + '.csv', index=False)
