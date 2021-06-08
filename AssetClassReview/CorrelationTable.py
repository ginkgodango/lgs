import datetime as dt
import matplotlib.pyplot as plt
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

    df_combined_1 = pd.merge(left=df_lgs_dictionary, right=df_jpm, on=['JPM Account Id'], how='inner').sort_values(['LGS Asset Class Level 1', 'JPM Account Id', 'Date'])

    df_combined_2 = df_combined_1[df_combined_1['LGS Open'].isin([1]) & df_combined_1['LGS Liquidity'].isin([0])]

    groupby_dict = dict(list(df_combined_2.groupby(['LGS Asset Class Level 1'])))

    for asset_class, df_temp in groupby_dict.items():

        df_temp = df_temp[['LGS Name', 'Date', 'R_a_m_p']].pivot_table(index='Date', columns='LGS Name', values='R_a_m_p')

        plot = asset_class_heatmap(df_temp, 36)

        plot.figure.savefig("C:/Users/mnguyen/LGSS/Investments Team - SandPits - SandPits/data/output/lgs/correlations/" + str(asset_class) + '.png', dpi=300)

