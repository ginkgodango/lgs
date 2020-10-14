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


def df_rolling_geometric(df, column, period, grouping):

    return pd.concat([rolling_geometric_link(df, column, period, grouping), pd.DataFrame(period, index=(range(0, len(df))), columns=['Period'])], axis=1)


def df_rolling_average(df, column, period, grouping):

    return pd.concat([rolling_average_link(df, column, period, grouping), pd.DataFrame(period, index=(range(0, len(df))), columns=['Period'])], axis=1)


def rolling_geometric_link(df, column, period, grouping):

    return (
        df.groupby(grouping)[column].rolling(period).apply(lambda x: np.prod(1 + x) - 1).reset_index(drop=False) if abs(period) <= 12 else
        df.groupby(grouping)[column].rolling(period).apply(lambda x: (np.prod(1 + x) ** (period / 12)) - 1).reset_index(drop=False)
    )


def rolling_average_link(df, column, period, grouping):

    return df.groupby(grouping)[column].rolling(period).mean().reset_index(drop=False)


def percentage_to_decimal(df, columns):

    df[columns] = df[columns].apply(lambda x: x/100)


def forward_date(df, date_name):

    df[date_name] = list(map(lambda x: x + relativedelta(months=1, day=31), df[date_name]))


def suffix_to_column(s):

    return (
        'R_s_p' if s[-2:] == '.1' else
        'R_s_b' if s[-2:] == '.2' else
        'MV_s'
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

    df_lgs_dictionary = pd.read_excel(pd.ExcelFile(lgs_dictionary_path), sheet_name='Sheet1', header=0)
    df_lgs_dictionary = df_lgs_dictionary[~df_lgs_dictionary['LGS Name'].isin(['Australian Fixed Interest', 'International Fixed Interest', 'Inflation Linked Bonds', 'Liquid Alternatives', 'Short Term Fixed Interest'])].reset_index(drop=True)

    df_lgs_allocations1 = (pd.read_csv(lgs_allocations_path, parse_dates=['Date']).rename(columns={'Market Value': 'MV_s_a', 'Portfolio Weight': 'W_s_a_p', 'Dynamic Weight': 'W_s_a_d', 'Benchmark Weight': 'W_s_a_b'}))
    forward_date(df=df_lgs_allocations1, date_name='Date')
    percentage_to_decimal(df=df_lgs_allocations1, columns=['W_s_a_p', 'W_s_a_d', 'W_s_a_b'])
    df_lgs_allocations2 = pd.merge(left=df_lgs_dictionary, right=df_lgs_allocations1, left_on=['JPM ReportStrategyName'], right_on=['Asset Class'])
    df_lgs_allocations3 = df_lgs_allocations2[
        [
            'Date',
            'Strategy',
            'LGS Asset Class Level 1',
            'MV_s_a',
            'W_s_a_p',
            'W_s_a_d',
            'W_s_a_b'
        ]
    ]

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
        set_values_name='MV_a_m'
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
        set_values_name='MV_a_m'
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
        set_values_name='R_a_m_p'
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
        set_values_name='R_a_m_p'
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
        set_values_name='R_a_m_b'
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
        set_values_name='R_a_m_b'
    )

    df_jpms = [
        pd.concat([df_jpm_main_mv, df_jpm_alts_mv]),
        pd.concat([df_jpm_main_returns, df_jpm_alts_returns]),
        pd.concat([df_jpm_main_benchmarks, df_jpm_alts_benchmarks])
        ]

    df_jpm = reduce(lambda x, y: pd.merge(left=x, right=y, on=['JPM Account Id', 'Date'], how='inner'), df_jpms)

    percentage_to_decimal(df=df_jpm, columns=['R_a_m_p', 'R_a_m_b'])

    df_combined1 = pd.merge(left=df_lgs_dictionary, right=df_jpm, on=['JPM Account Id'], how='inner').sort_values(['LGS Asset Class Level 1', 'JPM Account Id', 'Date'])

    df_managers_sum_mv = (
        df_combined1[df_combined1['LGS Sector Aggregate'].isin([0])]
        .groupby(['LGS Asset Class Level 1', 'Date'])['MV_a_m']
        .sum()
        .reset_index(drop=False)
        .rename(columns={'MV_a_m': 'MV_a_sum_m'})
    )

    df_combined2 = pd.merge(
        left=df_combined1,
        right=df_managers_sum_mv,
        left_on=['LGS Asset Class Level 1', 'Date'],
        right_on=['LGS Asset Class Level 1', 'Date']
    ).sort_values(['JPM Account Id', 'Date']).reset_index(drop=True)

    df_combined2['W_a_m'] = df_combined2['MV_a_m'] / df_combined2['MV_a_sum_m']

    df_sector1 = (
        df_combined1[df_combined1['LGS Sector Aggregate'].isin([1])]
        [['LGS Asset Class Level 1', 'Date', 'MV_a_m', 'R_a_m_p', 'R_a_m_b']]
        .rename(columns={'MV_a_m': 'MV_a', 'R_a_m_p': 'R_a_p', 'R_a_m_b': 'R_a_b'})
    )

    df_combined3 = pd.merge(
        left=df_combined2,
        right=df_sector1,
        left_on=['LGS Asset Class Level 1', 'Date'],
        right_on=['LGS Asset Class Level 1', 'Date']
    ).sort_values(['JPM Account Id', 'Date']).reset_index(drop=True)

    df_combined3['MV_a_diff'] = df_combined3['MV_a_sum_m'] - df_combined3['MV_a']

    df_combined3['W_a_m_R_a_p'] = df_combined3['W_a_m'] * df_combined3['R_a_p']

    df_combined3 = df_combined3[df_combined3['LGS Sector Aggregate'].isin([0])]

    # a = df_combined3.groupby(['LGS Asset Class Level 1', 'Date']).sum().rename(columns={'W_a_m_R_a_p': 'R_a_sum_m_p'})

    df_combined4 = (
        df_combined3[
            [
                'JPM Account Id',
                'LGS Name',
                'LGS Benchmark',
                'LGS Asset Class Level 1',
                'LGS Sector Aggregate',
                'LGS Asset Class Order',
                'LGS Manager Order',
                'Date',
                'MV_a_m',
                'MV_a_sum_m',
                'MV_a',
                'MV_a_diff',
                'W_a_m',
                'R_a_m_p',
                'R_a_m_b',
                'R_a_p',
                'R_a_b'
            ]
        ]
    ).reset_index(drop=True)

    df_fund_ids = df_combined4[
            [
                'JPM Account Id',
                'LGS Name',
                'Date',
                'LGS Benchmark',
                'LGS Asset Class Level 1',
                'LGS Sector Aggregate',
                'LGS Asset Class Order',
                'LGS Manager Order'
            ]
        ].reset_index(drop=False).rename(columns={"index": 'level_1'})

    average = ['W_a_m']

    geometric = ['R_a_m_p', 'R_a_m_b', 'R_a_p', 'R_a_b']

    horizons = [1, 3]

    multiperiod1 = list(map(lambda x: pd.concat(list(map(lambda y: df_rolling_average(df_combined4, x, y, ['JPM Account Id']), horizons)), axis=0), average))

    multiperiod2 = list(map(lambda x: pd.concat(list(map(lambda y: df_rolling_geometric(df_combined4, x, y, ['JPM Account Id']), horizons)), axis=0), geometric))

    multiperiod3 = multiperiod1 + multiperiod2

    df_multiperiod = reduce(lambda x, y: pd.merge(left=x, right=y, on=['JPM Account Id', 'Period', 'level_1'], how='inner'), multiperiod3)

    df_combined5 = pd.merge(
        left=df_fund_ids,
        right=df_multiperiod,
        left_on=['JPM Account Id', 'level_1'],
        right_on=['JPM Account Id', 'level_1'],
        how='inner'
    )

    df_jpm_strategy = jpm_wide_to_long(
        df=pd.read_excel(
            jpm_strategy_returns_benchmarks_mv_path,
            sheet_name='Sheet1',
            skiprows=use_account_id,
            skipfooter=footnote_rows,
            header=1
        ),
        set_date_name='Date',
        set_index_name='JPM Account Id',
        set_values_name='values'
    )

    df_jpm_strategy['column_name'] = [suffix_to_column(df_jpm_strategy['JPM Account Id'][i]) for i in range(0, len(df_jpm_strategy))]

    #df_jpm_strategy1 = df_jpm_strategy.set_index(['JPM Account Id', 'Date'])

    df_jpm_strategy.pivot(index=['JPM Account Id', 'Date'], columns='column_name', values='values')

