import datetime as dt
import numpy as np
import os as os
import pandas as pd


def read_folder(folderPath):

    return pd.concat([pd.read_csv(folderPath + fileName, parse_dates=['Date']) for fileName in os.listdir(folderPath)])


def calculate_returns(df, columnName, time, groupByList):

    if time <= 12:

        return df.groupby(groupByList)[columnName].rolling(time).apply(lambda r: np.prod(1 + r) - 1, raw=False).reset_index(drop=False)[columnName]

    elif time > 12:

        return df.groupby(groupByList)[columnName].rolling(time).apply(lambda r: (np.prod(1 + r) ** (12 / time)) - 1, raw=False).reset_index(drop=False)[columnName]


def string_to_float(x):

    return 0 if x == '-' else float(x.replace(',', ''))


if __name__ == '__main__':

    folderPath = 'C:/Users/mnguyen/LGSS/Investments Team - SandPits - SandPits/data/input/vendors/superratings/All/'

    df_0_a = read_folder(folderPath)

    remove_list = ['Top Quartile', 'Median', 'Bottom Quartile', 'Not for Profit Fund Median', 'Master Trust Median'] + list(set(df_0_a['SR Index']))

    df_0_b = df_0_a[~df_0_a['Option Name'].isin(remove_list)].reset_index(drop=True)

    # df_1 = df_0.dropna(subset=['SR Index']).reset_index(drop=True)[['Option Name', 'Date', 'SR Index', 'Size $Mill', 'Monthly Return %']]

    df_1_a = df_0_b.dropna(subset=['Size $Mill']).reset_index(drop=True)[['Option Name', 'Date', 'SR Index', 'Monthly Return %', 'Rolling 1 Year %']]

    df_1_b = df_0_b.dropna(subset=['Size $Mill']).reset_index(drop=True)[['Option Name', 'Date', 'SR Index', 'Monthly Return %']]

    df_2 = df_1_b.copy().sort_values(['Option Name', 'SR Index', 'Date']).reset_index(drop=True)

    # df_2['Size $Mill'] = [string_to_float(x) for x in df_2['Size $Mill']]

    df_2['1 Month Return'] = [string_to_float(x)/100 for x in df_2['Monthly Return %']]

    df_2['1 Year Return'] = calculate_returns(df_2, '1 Month Return', 12, ['Option Name', 'SR Index'])

    df_2['2 Year Return'] = calculate_returns(df_2, '1 Month Return', 24, ['Option Name', 'SR Index'])

    df_3 = df_2.copy()

    df_3['2 Year Rank'] = df_3.groupby(['SR Index', 'Date'])['2 Year Return'].rank(ascending=False)

    df_3 = df_3[df_3['Date'].isin([dt.datetime(2021, 4, 30)])].sort_values(['Option Name'], ascending=[True])

    # df_4 = pd.merge(left=df_1_a, right=df_3, left_on=['Option Name', 'SR Index', 'Date'], right_on=['Option Name', 'SR Index', 'Date'], how='inner')

    # df_5 = df_4[df_4['Date'].isin([dt.datetime(2021, 4, 30)])].sort_values(['Option Name', '1 Year Return'])

    df_6 = df_3[['Option Name', 'Date', 'SR Index', '2 Year Return', '2 Year Rank']].rename(columns={'2 Year Return': 'Rolling 2 Year %', '2 Year Rank': 'Rolling 2 Year Rank'})

    df_6.to_csv('C:/Users/mnguyen/LGSS/Investments Team - SandPits - SandPits/data/input/vendors/superratings/superratings_2years.csv', index=False)
