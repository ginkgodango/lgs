import pandas as pd
import datetime as dt
import numpy as np


def read_aa(path, sheet):

    return pd.read_excel(pd.ExcelFile(path), sheet_name=sheet)


def process_aa(df):

    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m', errors='coerce')

    df['Date'] = [last_day_of_month(df['Date'][i]) for i in range(len(df))]

    df = df.set_index('Date')

    df = df.resample('M').pad()

    df = df.drop('IE', axis=1)

    return df.reset_index(drop=False)


def read_bn(path, sheet):

    return pd.read_excel(pd.ExcelFile(path), sheet_name=sheet)


def process_bn(df):

    df['Date'] = [last_day_of_month(df['Date'][i]) for i in range(len(df))]

    df = df.set_index('Date')

    df = df.resample('M').pad()

    return df.reset_index(drop=False)


def last_day_of_month(any_day):
    next_month = any_day.replace(day=28) + dt.timedelta(days=4)
    return next_month - dt.timedelta(days=next_month.day)


asset_categories = [
        'IE_H',
        'IE_UH',
        'AE',
        'AP',
        'IP',
        'IN',
        'AF',
        'IF',
        'AC',
        'OTHER',
]

horizon_to_period_dict = {
        '1': 1,
        '3': 3,
        '12': 12,
        '36': 36,
        '60': 60,
        '84': 84
}

if __name__ == "__main__":

    report_date = dt.datetime(2020, 9, 30)

    data_path = 'C:/Users/Mnguyen/LGSS/Investments Team - SandPits - SandPits/Email Messages/Allocations Benchmarks 2020 Q3.xlsx'

    aa_sheets = ['HG1', 'HG2', 'BG1', 'BG2', 'BA1', 'BA2', 'CO1', 'CO2']

    bn_sheets = ['BN1', 'BN2']

    df_allocations = pd.concat([process_aa(read_aa(data_path, sheet)) for sheet in aa_sheets]).reset_index(drop=True)

    df_allocations = pd.melt(df_allocations, id_vars=['Date', 'Strategy', 'AA Version'], value_vars=asset_categories).sort_values(['Strategy', 'Date', 'variable']).reset_index(drop=True).rename(columns={'value': 'Allocation'})

    df_benchmarks = pd.concat([process_bn(read_bn(data_path, sheet)) for sheet in bn_sheets]).reset_index(drop=True)

    df_benchmarks = pd.melt(df_benchmarks, id_vars=['Date', 'BN Version'], value_vars=asset_categories).sort_values(['Date', 'variable']).reset_index(drop=True).rename(columns={'value': 'Benchmark'})

    df_taxfees = pd.read_excel(pd.ExcelFile(data_path), sheet_name='TaxFees')

    df_combined = pd.merge(left=df_allocations, right=df_benchmarks, on=['Date', 'variable']).sort_values(['Date', 'Strategy', 'variable', 'AA Version', 'BN Version'])

    df_combined['Contribution'] = df_combined['Allocation'] * df_combined['Benchmark']

    df_apra = (
            df_combined[['Date', 'Strategy', 'AA Version', 'BN Version', 'Contribution']]
            .groupby(['Date', 'Strategy', 'AA Version', 'BN Version'])
            .sum()
            .reset_index(drop=False)
            .rename(columns={'Contribution': 'APRA'})
            .sort_values(['AA Version', 'BN Version', 'Strategy', 'Date'])
            .reset_index(drop=True)
    )

    for horizon, period in horizon_to_period_dict.items():

        column_name = 'APRA_' + horizon

        if period <= 12:
            df_apra[column_name] = (
                df_apra
                .groupby(['AA Version', 'BN Version', 'Strategy'])['APRA']
                .rolling(period)
                .apply(lambda r: np.prod(1 + r) - 1, raw=False)
                .reset_index(drop=False)['APRA']
            )

        elif period > 12:
            df_apra[column_name] = (
                df_apra
                .groupby(['AA Version', 'BN Version', 'Strategy'])['APRA']
                .rolling(period)
                .apply(lambda r: (np.prod(1 + r) ** (12 / period)) - 1, raw=False)
                .reset_index(drop=False)['APRA']
            )

    df_apra_report_date = df_apra[df_apra['Date'].isin([report_date])]

    # Writes to Excel.
    writer = pd.ExcelWriter('C:/Users/Mnguyen/LGSS/Investments Team - SandPits - SandPits/Email Messages/APRA_benchmark.xlsx', engine='xlsxwriter')
    df_apra_report_date.to_excel(writer, sheet_name='Latest', index=False)
    df_apra.to_excel(writer, sheet_name='History', index=False)
    df_combined.to_excel(writer, sheet_name='Data', index=False)
    writer.save()
