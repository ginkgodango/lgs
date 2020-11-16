import pandas as pd
import numpy as np


def read_jpm_ts(path, skip_rows, footer_rows):

    return pd.read_excel(pd.ExcelFile(path), skiprows=skip_rows, skipfooter=footer_rows)


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


def suffix_to_metric(s):

    return (
        'Return' if s.endswith('.1') else
        'Benchmark Return' if s.endswith('.2') else
        'Market Value'
    )


def remove_suffix(s):

    return (
        s[:-2] if s.endswith(('.1', '.2', '.3', '.4')) else
        s
    )


use_manager_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12]
use_account_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]
footnote_rows = 28

if __name__ == '__main__':

    jpm_path = 'C:/Users/Mnguyen/LGSS/Investments Team - SandPits - SandPits/data/input/vendors/jpm/markets/performance/2020/09/Historical Time Series - Monthly - Strategy Market Values Returns and Benchmarks.xlsx'

    df = jpm_wide_to_long(read_jpm_ts(jpm_path, use_manager_id, footnote_rows), 'Date', 'JPM Account Name', 'Values')

    df['Metric'] = [suffix_to_metric(df['JPM Account Name'][i]) for i in range(len(df))]

    df['JPM Account Name'] = [remove_suffix(df['JPM Account Name'][i]) for i in range(len(df))]

    df = (
        pd
        .pivot_table(df, index=['JPM Account Name', 'Date'], columns='Metric', values='Values')
        .reset_index(drop=False)[['JPM Account Name', 'Date', 'Market Value', 'Return', 'Benchmark Return']]
        .sort_values(['JPM Account Name', 'Date'])
        .reset_index(drop=True)
    )

    df['Return'] /= 100

    df['Benchmark Return'] /= 100

    df['Return Vol 5 Years (Annualised)'] = (
        df
        .groupby(['JPM Account Name'])[['Return']]
        .rolling(60)
        .apply(lambda r: np.std(r, ddof=1)*np.sqrt(12), raw=False)
        .reset_index(drop=False)['Return']
    )

    df['Benchmark Return Vol 5 Years (Annualised)'] = (
        df
        .groupby(['JPM Account Name'])[['Benchmark Return']]
        .rolling(60).apply(lambda r: np.std(r, ddof=1)*np.sqrt(12), raw=False)
        .reset_index(drop=False)['Benchmark Return']
    )

    df.to_csv('C:/Users/Mnguyen/LGSS/Investments Team - SandPits - SandPits/data/output/lgs/volatilities/strategy_volatilities_20200930.csv', index=False)
