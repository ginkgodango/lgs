import datetime as dt
import numpy as np
import pandas as pd


def read_jpm(s):

    return pd.read_csv(s, skiprows=[0, 1, 2, 3])


def process_jpm(df, columns):

    return df[columns]


def read_fsi(s):

    return pd.read_excel(
        pd.ExcelFile(s),
        sheet_name='Holdings',
        skiprows=[0, 1, 2, 3, 4, 5, 6, 8],
        header=0,
        usecols=[
            'Security SEDOL',
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
                'Security Name': 'Security Name',
                'Market Price (Local Currency)': 'Local Price',
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
        'Investment Description': 'Security Name',
        'Price Local': 'Local Price',
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
        sheet_name='EM SICAV holdings 8-31-2020',
        header=0,
        usecols=[
                'Security SEDOL',
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
        'Security Description (Short)': 'Security Name',
        'Price (Local)': 'Local Price',
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
        sheet_name='wellington_holdings',
        header=0,
        usecols=[
                'SEDOL',
                'Security',
                'Unit Price',
                'Shares or Par Value',
                'Report Currency',
                'Market Value',
        ]
)


def process_wel(df, mv, columns):

    df = df.rename(
        columns={
            'SEDOL': 'Security ID',
            'Security': 'Security Name',
            'Unit Price': 'Local Price',
            'Shares or Par Value': 'Unit Holding',
            'Report Currency': 'Local Currency',
            'Market Value': 'Realizable Value Base'
        }
    )

    rescale_ratio = mv / df['Realizable Value Base'].sum()
    df['Realizable Value Base'] *= rescale_ratio
    df['Unit Holding'] *= rescale_ratio
    df['Portfolio Name'] = 'LGS IE WEL'
    df['Category Description'] = np.nan

    return df[columns]


def swap(a, b, condition):

    return a if condition else b


columns_list = [
    'Portfolio Name',
    'Category Description',
    'Local Currency',
    'Security ID',
    'Security Name',
    'Local Price',
    'Unit Holding',
    'Realizable Value Base'
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

if __name__ == '__main__':

    # Set file directories.
    jpm_path = 'U:/CIO/#Data/input/jpm/holdings/2020/09/Detailed Valuation Report - Equities.csv'
    fsi_path = 'U:/CIO/#Data/input/lgs/holdings/unitprices/2020/09/wscf_holdings.xlsx'
    aqr_path = 'U:/CIO/#Data/input/lgs/holdings/unitprices/2020/09/aqr_holdings.xlsx'
    mac_path = 'U:/CIO/#Data/input/lgs/holdings/unitprices/2020/09/macquarie_holdings.xlsx'
    wel_path = 'U:/CIO/#Data/input/lgs/holdings/unitprices/2020/09/wellington_holdings.xlsx'

    # Get market value from JPM Investment Accounting System.
    date = dt.datetime(2020, 8, 31)
    fsi_mv = 194719540.46
    aqr_mv = 182239774.63
    mac_mv = 151551731.17
    wel_mv = 149215529.22

    # Reads in each file as a dataframe and cleans it.
    df_jpm = process_jpm(df=read_jpm(jpm_path), columns=columns_list)
    df_fsi = process_fsi(df=read_fsi(fsi_path), mv=fsi_mv, columns=columns_list)
    df_aqr = process_aqr(df=read_aqr(aqr_path), mv=aqr_mv, columns=columns_list)
    df_mac = process_mac(df=read_mac(mac_path), mv=mac_mv, columns=columns_list)
    df_wel = process_wel(df=read_wel(wel_path), mv=wel_mv, columns=columns_list)

    # Joins all files into one dataframe.
    df_all = pd.concat([df_jpm, df_fsi, df_aqr, df_mac, df_wel], axis=0).sort_values('Portfolio Name').reset_index(drop=True)

    # Swaps the Security IDs of the Liquidity accounts with their Security Names. This solves the uniqueness problem.
    df_all['Security ID'] = [str(swap(df_all['Security Name'][i], df_all['Security ID'][i], df_all['Category Description'][i] == 'Liquidity')) for i in range(len(df_all))]

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
    writer = pd.ExcelWriter('U:/CIO/#Data/output/holdings/portfolio_holdings.xlsx', engine='xlsxwriter')
    df_ae_info.to_excel(writer, sheet_name='AE', index=False)
    df_ie_info.to_excel(writer, sheet_name='IE', index=False)
    df_all.to_excel(writer, sheet_name='All', index=False)
    writer.save()
