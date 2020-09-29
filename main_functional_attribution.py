import pandas as pd
import numpy as np
from functools import reduce


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


if __name__ == "__main__":
    lgs_dictionary_path = 'U:/CIO/#Data/input/lgs/dictionary/2020/09/New Dictionary_v12.xlsx'
    lgs_allocations_path = 'U:/CIO/#Data/input/lgs/allocations/asset_allocations_2020-08-31.csv'
    jpm_main_mv_path = 'U:/CIO/#Data/input/jpm/performance/2020/08/Historical Time Series - Monthly - Main Market Values.xlsx'
    jpm_alts_mv_path = 'U:/CIO/#Data/input/jpm/performance/2020/08/Historical Time Series - Monthly - Alts Market Values.xlsx'
    jpm_main_returns_path = 'U:/CIO/#Data/input/jpm/performance/2020/08/Historical Time Series - Monthly - Main Returns.xlsx'
    jpm_alts_returns_path = 'U:/CIO/#Data/input/jpm/performance/2020/08/Historical Time Series - Monthly - Alts Returns.xlsx'
    jpm_main_benchmarks_path = 'U:/CIO/#Data/input/jpm/performance/2020/08/Historical Time Series - Monthly - Main Benchmarks.xlsx'
    jpm_alts_benchmarks_path = 'U:/CIO/#Data/input/jpm/performance/2020/08/Historical Time Series - Monthly - Alts Benchmarks.xlsx'
    jpm_strategy_returns_benchmarks_mv_path = 'U:/CIO/#Data/input/jpm/performance/2020/08/Historical Time Series - Monthly - Strategy Market Values Returns and Benchmarks.xlsx'

    use_manager_id = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12]
    use_account_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
    footnote_rows = 28

    df_lgs = pd.read_excel(pd.ExcelFile(lgs_dictionary_path), sheet_name='Sheet1', header=0)

    df_jpm_main_mv = jpm_wide_to_long(
        df=pd.read_excel(
            pd.ExcelFile(jpm_main_mv_path),
            sheet_name='Sheet1',
            skiprows=use_account_id,
            skipfooter=footnote_rows,
            header=1
        ),
        set_date_name='Date',
        set_index_name='JPM Account Id',
        set_values_name='JPM Market Value'
    )

    df_jpm_alts_mv = jpm_wide_to_long(
        df=pd.read_excel(
            pd.ExcelFile(jpm_alts_mv_path),
            sheet_name='Sheet1',
            skiprows=use_account_id,
            skipfooter=footnote_rows,
            header=1
        ),
        set_date_name='Date',
        set_index_name='JPM Account Id',
        set_values_name='JPM Market Value'
    )

    df_jpm_main_returns = jpm_wide_to_long(
        df=pd.read_excel(
            pd.ExcelFile(jpm_main_returns_path),
            sheet_name='Sheet1',
            skiprows=use_account_id,
            skipfooter=footnote_rows,
            header=1
        ),
        set_date_name='Date',
        set_index_name='JPM Account Id',
        set_values_name='JPM Return'
    )

    df_jpm_alts_returns = jpm_wide_to_long(
        df=pd.read_excel(
            pd.ExcelFile(jpm_alts_returns_path),
            sheet_name='Sheet1',
            skiprows=use_account_id,
            skipfooter=footnote_rows,
            header=1
        ),
        set_date_name='Date',
        set_index_name='JPM Account Id',
        set_values_name='JPM Return'
    )

    df_jpm_main_benchmarks = jpm_wide_to_long(
        df=pd.read_excel(
            pd.ExcelFile(jpm_main_benchmarks_path),
            sheet_name='Sheet1',
            skiprows=use_account_id,
            skipfooter=footnote_rows,
            header=1
        ),
        set_date_name='Date',
        set_index_name='JPM Account Id',
        set_values_name='JPM Benchmark'
    )

    df_jpm_alts_benchmarks = jpm_wide_to_long(
        df=pd.read_excel(
            pd.ExcelFile(jpm_alts_benchmarks_path),
            sheet_name='Sheet1',
            skiprows=use_account_id,
            skipfooter=footnote_rows,
            header=1
        ),
        set_date_name='Date',
        set_index_name='JPM Account Id',
        set_values_name='JPM Benchmark'
    )

    df_jpms = [
        pd.concat([df_jpm_main_mv, df_jpm_alts_mv]),
        pd.concat([df_jpm_main_returns, df_jpm_alts_returns]),
        pd.concat([df_jpm_main_benchmarks, df_jpm_alts_benchmarks])
        ]

    df_jpm = reduce(lambda x, y: pd.merge(left=x, right=y, on=['JPM Account Id', 'Date']), df_jpms)
