from datetime import datetime
import pandas as pd
import numpy as np


def load_update(filepath):
    xlsx = pd.ExcelFile(filepath)

    df_main = pd.DataFrame()

    tabs_to_columns_dict = {
        'Page 3 NOF': 'A:N',
        'Page 5 NOF': 'B:O',
        'Page 6 NOF': 'B:O',
        'Page 7 NOF': 'B:O'
    }

    for tab, columns in tabs_to_columns_dict.items():
        print('Accessing:', tab)
        df_tab = pd.read_excel(
            xlsx,
            sheet_name=tab,
            usecols=columns,
            skiprows=[0, 1, 2]
        )

        df_tab = df_tab.rename(
            columns={
                'Unnamed: 0': 'ModelCode',
                'Unnamed: 1': 'JPM_Name'
            }
        )

        df_main = pd.concat([df_main, df_tab], sort=False)

    df_main = df_main.reset_index(drop=True)

    return df_main


def add_report_date(df, report_date):
    df.insert(
        loc=1,
        column='Date',
        value=report_date
        )

    return df


def clean(df):
    df = df[df['ModelCode'].notnull()]
    df = df.set_index('ModelCode')
    df = df[~df.index.duplicated(keep='first')]

    return df


def load_returns(filepath):
    df = pd.read_csv(
        filepath,
        index_col='Date',
        parse_dates=['Date'],
        infer_datetime_format=True,
        float_precision='round_trip'
    )
    return df


dead_funds = [
        'Intrinsic_AE',
        'ActiveIndex_IE',
        'Vanguard_IE',
        'FOAM_IE',
        'AMP_ILP',
        'SSgA_AFI',
        'SSgA_ILB',
        'Pimco_IFI',
        'PMESGGW_AU_IFI',
        'Omega_IFI',
        'WellingtonEMDebt_IFI',
        'IFILiquidity_IFI',
        'Incapture_AR',
        'GMO_SGM_B_AR',
        'Bonds',
        'BO',
        'IFM_DI',
        '<= Managers&Sectors Data Set',
        'Benchmark Data Set=>',
        'AUBI.Plus1_Index',
        'AUBI.Plus2_Index',
        'ASA6PROP_Index',
        'SSgACUSTOM_Index',
        'EOAS_Index',
        'MXEF_Index',
        'UBSAFI0.5yr_Index',
        'UBSG0.YR_Index',
        'M1WOQU_Index',
        'AUGovtBondPlus4_Index',
        'AESectorBmk_Index',
        'PESectorUSD_Index',
        'PESectorAUDUSD_Index'
    ]


def update_check_missing_returns(df_returns, df_update, dead_funds_list = dead_funds):
    update_missing_list = []
    update_matched_list = []

    for column_name in df_returns.columns:
        if column_name in df_update.index:
            update_matched_list.append(column_name)
        else:
            update_missing_list.append(column_name)

    update_missing_list = [item for item in update_missing_list if item not in dead_funds_list]

    print('Managers are missing: \n', update_missing_list)

    return update_missing_list


def update_check_new_returns(df_returns, df_update):
    update_new_list = []
    update_existing_list = []
    for manager in df_update.index:
        if manager in df_returns.columns:
            update_existing_list.append(manager)
        else:
            update_new_list.append(manager)

    print('Managers are new: \n', update_new_list)

    return update_new_list


def create_update_dict(df, days_in_month, SSgACUSTOM_Index, EOAS_Index, MXEF_Index):
    update_dict = dict()
    update_dict['AESectorBmk_Index'] = df['1 Month']['ASA52_Index'] / 100
    update_dict['SSgACUSTOM_Index'] = SSgACUSTOM_Index / 100
    update_dict['EOAS_Index'] = EOAS_Index / 100
    update_dict['AUBI.Plus1_Index'] = df['1 Month']['AUBI_Index'] / 100 + 1 * (days_in_month / 365) / 100
    update_dict['AUBI.Plus2_Index'] = df['1 Month']['AUBI_Index'] / 100 + 2 * (days_in_month / 365) / 100
    update_dict['MXEF_Index'] = MXEF_Index / 100
    update_dict['ASX200+ASX100_Index'] = df['1 Month']['ASA25_Index'] / 100

    for index in df.index:
        update_dict[index] = df['1 Month'][index] / 100

    return update_dict


def update_dict_to_df(update_dict, report_date):
    df = pd.DataFrame(
        [update_dict],
        columns=update_dict.keys(),
        index=[report_date]
    )

    return df


def apply_update_to_df_returns(df_returns, df_update_ready):
    df_returns = pd.concat([df_returns, df_update_ready], sort=False)

    return df_returns


def load_market_values(filepath):
    df = pd.read_csv(
        filepath,
        index_col='Date',
        parse_dates=['Date'],
        infer_datetime_format=True,
        float_precision='round_trip'
    )

    return df


def update_check_missing_market_values(df_market_values, df_update, dead_funds_list = dead_funds):
    update_missing_mv = []
    update_matched_mv = []
    for column_name in df_market_values.columns:
        if column_name in df_update.index:
            update_matched_mv.append(column_name)
        else:
            update_missing_mv.append(column_name)

    update_missing_mv = [item for item in update_missing_mv if item not in dead_funds_list]

    print('Market values are missing: \n', update_missing_mv)

    return update_missing_mv


def update_check_new_market_values(df_market_values, df_update):
    update_mv_new = []
    update_mv_existing = []
    for item in df_update.index:
        if item in df_market_values.columns:
            update_mv_existing.append(item)
        else:
            if not pd.isnull(df_update['Market Value'][item]):
                update_mv_new.append(item)

    print('Market values are new: \n', update_mv_new)

    return update_mv_new


def create_update_market_value_dict(df_update):
    update_dict = dict()
    for index in df_update.index:
        update_dict[index] = df_update['Market Value'][index]

    return update_dict


def update_market_values_dict_to_df(update_mv_dict, report_date):
    df_update_mv = pd.DataFrame(
        [update_mv_dict],
        columns=update_mv_dict.keys(),
        index=[report_date]
    )
    df_update_mv = df_update_mv.dropna(axis=1, how='any')

    return df_update_mv


def apply_update_to_df_market_values(df_market_values, df_update):
    df = pd.concat([df_market_values, df_update], sort=False)

    return df

