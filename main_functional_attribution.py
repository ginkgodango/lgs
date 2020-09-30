import pandas as pd
import numpy as np
from functools import reduce
from dateutil.relativedelta import relativedelta


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


def rolling_geometric_link(df, column, period, grouping):

    return (
        df.groupby(grouping)[column].rolling(period).apply(lambda x: np.prod(1 + x) - 1).reset_index(drop=False) if abs(period) <= 12 else
        df.groupby(grouping)[column].rolling(period).apply(lambda x: (np.prod(1 + x) ** (period / 12)) - 1).reset_index(drop=False)
    )


def rolling_mean(df, column, period, grouping):

    return df.groupby(grouping)[column].rolling(period).mean().reset_index(drop=False)


def percentage_to_decimal(df, columns):

    df[columns] = df[columns].apply(lambda x: x/100)


def forward_date(df, date_name):

    df[date_name] = list(map(lambda x: x + relativedelta(months=1, day=31), df_lgs_allocations[date_name]))


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

    df_lgs_dictionary = pd.read_excel(pd.ExcelFile(lgs_dictionary_path), sheet_name='Sheet1', header=0)

    df_lgs_allocations = pd.read_csv(lgs_allocations_path, parse_dates=['Date'])

    forward_date(df=df_lgs_allocations, date_name='Date')

    percentage_to_decimal(df=df_lgs_allocations, columns=['Portfolio Weight', 'Dynamic Weight', 'Benchmark Weight'])

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

    df_jpm = reduce(lambda x, y: pd.merge(left=x, right=y, on=['JPM Account Id', 'Date'], how='inner'), df_jpms)

    percentage_to_decimal(df=df_jpm, columns=['JPM Return', 'JPM Benchmark'])

    df_combined1 = pd.merge(left=df_lgs_dictionary, right=df_jpm, on=['JPM Account Id'], how='inner').sort_values(['LGS Asset Class Level 1', 'JPM Account Id', 'Date'])

    df_managers_sum_mv = (
        df_combined1[df_combined1['LGS Sector Aggregate'].isin([0])]
        .groupby(['LGS Asset Class Level 1', 'Date'])['JPM Market Value']
        .sum()
        .reset_index(drop=False)
        .rename(columns={'JPM Market Value': 'Asset Class Sum Manager MV'})
    )

    df_combined2 = pd.merge(
        left=df_combined1,
        right=df_managers_sum_mv,
        left_on=['LGS Asset Class Level 1', 'Date'],
        right_on=['LGS Asset Class Level 1', 'Date']
    ).sort_values(['JPM Account Id', 'Date'])

    df_combined2['W_a_m'] = df_combined2['JPM Market Value'] / df_combined2['Asset Class Sum Manager MV']

    df_sector1 = (
        df_combined1[df_combined1['LGS Sector Aggregate'].isin([1])]
        [['LGS Asset Class Level 1', 'Date', 'JPM Market Value', 'JPM Return', 'JPM Benchmark']]
        .rename(columns={'JPM Market Value': 'Asset Class MV', 'JPM Return': 'R_a_p', 'JPM Benchmark': 'R_a_b'})
    )

    df_combined3 = pd.merge(
        left=df_combined2,
        right=df_sector1,
        left_on=['LGS Asset Class Level 1', 'Date'],
        right_on=['LGS Asset Class Level 1', 'Date']
    ).sort_values(['JPM Account Id', 'Date'])

    df_attribution = (
        df_combined3[[
            'LGS Name',
            'LGS Benchmark',
            'LGS Asset Class Level 1',
            'LGS Sector Aggregate',
            'LGS Asset Class Order',
            'LGS Manager Order',
            'Date',
            'JPM Market Value',
            'Asset Class Sum Manager MV',
            'Asset Class MV',
            'JPM Return',
            'JPM Benchmark',
            'W_a_m',
            'R_a_p',
            'R_a_b'
        ]]
        .rename(
            columns={
                'JPM Return': 'R_a_m_p',
                'JPM Benchmark': 'R_a_m_s'
            }
        )
    )

    

