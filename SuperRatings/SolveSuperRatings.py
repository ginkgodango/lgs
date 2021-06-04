import datetime as dt
import math as math
import numpy as np
import os as os
import pandas as pd


def string_to_float(x):

    return 0 if x == '-' else float(x.replace(',', ''))


if __name__ == '__main__':

    folderPath = 'C:/Users/mnguyen/LGSS/Investments Team - SandPits - SandPits/data/input/vendors/superratings/All/'

    df_0 = pd.read_csv(folderPath + 'SuperRatings-FCRS-202104.csv', parse_dates=['Date'])

    df_1 = pd.read_csv(folderPath + 'SuperRatings-FCRS-202104.csv', parse_dates=['Date'])[['Option Name', 'Product Type', 'Date', 'SR Index', 'Rolling 3 Year %']]

    df_2 = pd.read_csv(folderPath + 'SuperRatings-FCRS-201904.csv', parse_dates=['Date'])[['Option Name', 'Product Type', 'Date', 'SR Index', 'Rolling 1 Year %']]

    df_1['R_0_3'] = [string_to_float(str(x)) / 100 for x in df_1['Rolling 3 Year %']]

    df_2['R_2_3'] = [string_to_float(str(x)) / 100 for x in df_2['Rolling 1 Year %']]

    remove_sr_indices_list = list(set(df_1['SR Index']).union(df_2['SR Index']))

    remove_aggregates_list = ['Top Quartile', 'Median', 'Bottom Quartile', 'Not for Profit Fund Median', 'Master Trust Median', 'Number of Investment Options Ranked']

    remove_list = remove_sr_indices_list + remove_aggregates_list

    df_1 = df_1[~df_1['Option Name'].isin(remove_list)].reset_index(drop=True)

    df_2 = df_2[~df_2['Option Name'].isin(remove_list)].reset_index(drop=True)

    df_1_count = df_1.groupby(['SR Index'])['Rolling 3 Year %'].count()

    df_2_count = df_1.groupby(['SR Index'])['Rolling 3 Year %'].count()

    df_3 = pd.merge(
        left=df_1,
        right=df_2,
        left_on=['Option Name', 'Product Type', 'SR Index'],
        right_on=['Option Name', 'Product Type', 'SR Index'],
        how='inner'
    )

    df_3['(1 + R_0_3)^3'] = [(1 + x)**3 for x in df_3['R_0_3']]

    df_3['(1 + R_2_3)^1'] = [(1 + x) ** 1 for x in df_3['R_2_3']]

    df_3['(1 + R_0_2)^2'] = df_3['(1 + R_0_3)^3'] / df_3['(1 + R_2_3)^1']

    df_3['R_0_2'] = [math.sqrt(x) - 1 for x in df_3['(1 + R_0_2)^2']]

    df_3['2 Year Rank'] = df_3.groupby(['SR Index'])['R_0_2'].rank(ascending=False)

    df_3_count = df_3.groupby(['SR Index'])['R_0_2'].count()

    df_3.to_csv('C:/Users/mnguyen/LGSS/Investments Team - SandPits - SandPits/data/output/vendors/superratings/SolveSuperRatingsTest.csv', index=False)

    df_4 = df_3[['Option Name', 'Product Type', 'SR Index', 'R_0_2', '2 Year Rank']].reset_index(drop=True)

    df_4['Rolling 2 Year %'] = [str(round(x * 100, 2)) for x in df_4['R_0_2']]

    df_4['Rolling 2 Year Rank'] = [str(round(x, 0)) for x in df_4['2 Year Rank']]

    df_5 = pd.merge(left=df_0, right=df_4, left_on=['Option Name', 'Product Type', 'SR Index'], right_on=['Option Name', 'Product Type', 'SR Index'], how='inner')

    df_6 = df_5[
        [
            'Option Name',
            'Product Type',
            'Date',
            'SR Index',
            'Rolling 1 Year %',
            'Rolling 2 Year %',
            'Rolling 3 Year %',
            'Rolling 5 Year %',
            'Rolling 7 Year %',
            'Rolling 1 Year Rank',
            'Rolling 2 Year Rank',
            'Rolling 3 Year Rank',
            'Rolling 5 Year Rank',
            'Rolling 7 Year Rank'
        ]
    ]

    df_6_HG = df_6[df_6['Option Name'].isin(['Active Super - High Growth'])][
        ['Option Name', 'Rolling 7 Year %', 'Rolling 7 Year Rank']].rename(
        columns={'Rolling 7 Year %': 'Return', 'Rolling 7 Year Rank': 'Rank'})

    df_6_BG = df_6[df_6['Option Name'].isin(['Active Super - Balanced Growth'])][
        ['Option Name', 'Rolling 5 Year %', 'Rolling 5 Year Rank']].rename(
        columns={'Rolling 5 Year %': 'Return', 'Rolling 5 Year Rank': 'Rank'})

    df_6_BA = df_6[df_6['Option Name'].isin(['Active Super - Balanced'])][
        ['Option Name', 'Rolling 3 Year %', 'Rolling 3 Year Rank']].rename(
        columns={'Rolling 3 Year %': 'Return', 'Rolling 3 Year Rank': 'Rank'})

    df_6_CO = df_6[df_6['Option Name'].isin(['Active Super - Conservative'])][
        ['Option Name', 'Rolling 2 Year %', 'Rolling 2 Year Rank']].rename(
        columns={'Rolling 2 Year %': 'Return', 'Rolling 2 Year Rank': 'Rank'})

    df_6_MC = df_6[df_6['Option Name'].isin(['Active Super - Managed Cash'])][
        ['Option Name', 'Rolling 2 Year %', 'Rolling 2 Year Rank']].rename(
        columns={'Rolling 2 Year %': 'Return', 'Rolling 2 Year Rank': 'Rank'})

    df_7 = pd.concat([df_6_HG, df_6_BG, df_6_BA, df_6_CO, df_6_MC]).reset_index(drop=True)
