import pandas as pd
import numpy as np
import datetime as dt


def read_option_historic_nav(path):

    return (
        pd
        .read_csv(path, skiprows=[0, 1, 2, 3], parse_dates=['Valuation Date'])
        .set_index('Valuation Date')
        .groupby(['Portfolio Code'])
        .resample('M')
        .pad()
        .drop(columns='Portfolio Code', axis=1)
        .reset_index(drop=False)
    )


def read_option_historic_ts(path, skip_rows):

    return pd.read_excel(pd.ExcelFile(path), skiprows=skip_rows)


def jpm_wide_to_long(df, set_date_name, set_index_name, set_values_name):
    """

    :param df:
    :param set_date_name:
    :param set_index_name:
    :param set_values_name:
    :return:
    """
    return (
        pd.melt(
            (df
             .replace('-', np.NaN)
             .rename(columns={'Unnamed: 0': set_date_name})
             .set_index('Date')
             .transpose()
             .reset_index(drop=False)
             .rename(columns={'index': set_index_name})
             ),
            id_vars=[set_index_name],
            value_name=set_values_name)
        .sort_values([set_index_name, set_date_name])
        .reset_index(drop=True)
    )


def columns_suffix(df, account_name):

    return


def suffix_to_metric(s):

    return (
        'Cash Flow' if s.endswith('.1') else
        'Return' if s.endswith('.2') else
        'Benchmark Return' if s.endswith('.3') else
        'Average Balance' if s.endswith('.4') else
        'Market Value'
    )


def remove_suffix(s):

    return (
        s[:-2] if s.endswith(('.1', '.2', '.3', '.4')) else
        s
    )


use_account_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]
footnote_rows = 28

if __name__ == '__main__':

    option_historic_nav_path = 'C:/Users/Mnguyen/LGSS/Investments Team - SandPits - SandPits/data/input/vendors/jpm/markets/investment_accounting/2020/Options Test/Historic NAV Report - Options.csv'

    option_historic_ts_path = 'C:/Users/Mnguyen/LGSS/Investments Team - SandPits - SandPits/data/input/vendors/jpm/markets/performance/2020/09/Historical Time Series - Monthly - Option (1).xlsx'

    df1 = pd.read_excel(pd.ExcelFile(option_historic_ts_path), skiprows=use_account_id, skipfooter=footnote_rows)

    excess_returns_columns = [x for x in df1.columns if x.startswith('--')]

    df2 = df1.drop(columns=excess_returns_columns, axis=1)

    df3 = jpm_wide_to_long(df2, 'Date', 'JPM Account Id', 'Values').reset_index(drop=True)

    df3['Metric'] = [suffix_to_metric(df3['JPM Account Id'][i]) for i in range(len(df3))]

    df3['JPM Account Id'] = [remove_suffix(df3['JPM Account Id'][i]) for i in range(len(df3))]

    df4 = df3[df3['JPM Account Id'].isin(['LFAEUP-ER', 'LFCKUP-ER'])].reset_index(drop=True)

    df5 = pd.pivot_table(df4, index=['JPM Account Id', 'Date'], columns='Metric', values='Values').reset_index(drop=False)

    df5 = df5[['JPM Account Id', 'Date', 'Market Value', 'Cash Flow', 'Return', 'Benchmark Return']]

    df5['Market Value Lag 1'] = df5.groupby(['JPM Account Id'])['Market Value'].shift(1)

    df5['Profit and Loss'] = df5['Market Value'] - df5['Cash Flow'] - df5['Market Value Lag 1']

    df5['LGS Return'] = df5['Profit and Loss'] / df5['Market Value Lag 1']

    df5.to_csv('C:/Users/Mnguyen/LGSS/Investments Team - SandPits - SandPits/documents/lgs/reports/options/option_reconstruction.csv')
