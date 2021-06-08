import datetime as dt
import numpy as np
import pandas as pd
import seaborn as sns
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


if __name__ == "__main__":
    folder_path = "C:/Users/mnguyen/LGSS/Investments Team - SandPits - SandPits/data/input/vendors/jpm/markets/performance/2021/04/"
    lgs_dictionary_path = 'U:/CIO/#Data/input/lgs/dictionary/2020/12/New Dictionary_v17.xlsx'
    lgs_allocations_path = 'U:/CIO/#Data/input/lgs/allocations/asset_allocations_2021-02-28.csv'
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

    df_combined_1 = pd.merge(left=df_lgs_dictionary, right=df_jpm, on=['JPM Account Id'], how='inner').sort_values(['LGS Asset Class Level 1', 'JPM Account Id', 'Date'])

    df_AR = df_combined_1[df_combined_1['LGS Asset Class Level 1'].isin(['AR'])]

    df_AR_corr = df_AR[['LGS Name', 'Date', 'R_a_m_p']].pivot_table(index='Date', columns='LGS Name', values='R_a_m_p')

    corr_matrix_36_month = df_AR_corr[-36:].corr()

    sns.heatmap(corr_matrix_36_month)
