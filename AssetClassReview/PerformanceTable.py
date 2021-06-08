import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from functools import reduce


def calculate_returns(df, columnName, time, groupByList):

    if time <= 12:

        return df.groupby(groupByList)[columnName].rolling(time).apply(lambda r: np.prod(1 + r) - 1, raw=False).reset_index(drop=False)[columnName]

    elif time > 12:

        return df.groupby(groupByList)[columnName].rolling(time).apply(lambda r: (np.prod(1 + r) ** (12 / time)) - 1, raw=False).reset_index(drop=False)[columnName]


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
            (
                df
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


def asset_class_heatmap(df, period):

    df_period = df[-period:]

    mask = np.triu(df_period.corr())

    plt.figure(figsize=(12.8, 12.8))

    return sns.heatmap(
        df_period.corr(),
        annot=True,
        mask=mask,
        cmap='coolwarm',
        square=True,
        linewidths=3,
        cbar_kws={"shrink": .5}
    )


if __name__ == "__main__":
    folder_path = "C:/Users/mnguyen/LGSS/Investments Team - SandPits - SandPits/data/input/vendors/jpm/markets/performance/2021/04/"
    lgs_dictionary_path = 'C:/Users/mnguyen/LGSS/Investments Team - SandPits - SandPits/data/input/lgs/dictionary/2021/03/New Dictionary_v21.xlsx'
    jpm_main_mv_path = folder_path + 'Historical Time Series - Monthly - Main Market Values.xlsx'
    jpm_alts_mv_path = folder_path + 'Historical Time Series - Monthly - Alts Market Values.xlsx'
    jpm_main_returns_path = folder_path + 'Historical Time Series - Monthly - Main Returns.xlsx'
    jpm_alts_returns_path = folder_path + 'Historical Time Series - Monthly - Alts Returns.xlsx'
    jpm_main_benchmarks_path = folder_path + 'Historical Time Series - Monthly - Main Benchmarks.xlsx'
    jpm_alts_benchmarks_path = folder_path + 'Historical Time Series - Monthly - Alts Benchmarks.xlsx'

    use_manager_id = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12]
    use_account_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
    footnote_rows = 28

    df_lgs_dictionary = pd.read_excel(pd.ExcelFile(lgs_dictionary_path), sheet_name='Sheet1', header=0)
    df_lgs_dictionary = df_lgs_dictionary[~df_lgs_dictionary['LGS Name'].isin(['Australian Fixed Interest', 'International Fixed Interest', 'Inflation Linked Bonds', 'Liquid Alternatives', 'Short Term Fixed Interest'])].reset_index(drop=True)

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

    
