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


darkgreen = (75/256, 120/256, 56/256)
middlegreen = (141/256, 177/256, 66/256)
lightgreen = (175/256, 215/256, 145/256)


if __name__ == '__main__':

    folderPath = 'C:/Users/mnguyen/LGSS/Investments Team - SandPits - SandPits/data/input/vendors/superratings/All/'

    df_0_a = read_folder(folderPath)

    select_list = [
        'Active Super - Balanced',
        'Active Super - Balanced Growth',
        'Active Super - Conservative',
        'Active Super - High Growth',
        'Active Super - Managed Cash',
        'Active Super MySuper - Balanced Growth'
    ]

    remove_list = ['Top Quartile', 'Median', 'Bottom Quartile', 'Not for Profit Fund Median', 'Master Trust Median', 'Number of Investment Options Ranked'] + list(set(df_0_a['SR Index']))

    df_0_b = df_0_a[~df_0_a['Option Name'].isin(remove_list)].reset_index(drop=True)

    # df_1 = df_0.dropna(subset=['SR Index']).reset_index(drop=True)[['Option Name', 'Date', 'SR Index', 'Size $Mill', 'Monthly Return %']]

    df_1_a = df_0_b.dropna(subset=['Size $Mill']).reset_index(drop=True)[['Option Name', 'Product Type', 'Date', 'SR Index', 'Monthly Return %', 'Rolling 1 Year %', 'Rolling 3 Year %', 'Rolling 1 Year Rank', 'Rolling 3 Year Rank']]

    df_1_b = df_0_b.dropna(subset=['Size $Mill']).reset_index(drop=True)[['Option Name', 'Product Type', 'Date', 'SR Index', 'Monthly Return %']]

    df_2 = df_1_b.copy().sort_values(['Option Name', 'Product Type', 'SR Index', 'Date']).reset_index(drop=True)

    # df_2['Size $Mill'] = [string_to_float(x) for x in df_2['Size $Mill']]

    df_2['1 Month Return'] = [string_to_float(x)/100 for x in df_2['Monthly Return %']]

    df_2['1 Year Return'] = calculate_returns(df_2, '1 Month Return', 12, ['Option Name', 'Product Type', 'SR Index'])

    df_2['2 Year Return'] = calculate_returns(df_2, '1 Month Return', 24, ['Option Name', 'Product Type', 'SR Index'])

    df_2['3 Year Return'] = calculate_returns(df_2, '1 Month Return', 36, ['Option Name', 'Product Type', 'SR Index'])

    # df_2['1 Month Return'].fillna(100, inplace=True)
    # df_2['1 Year Return'].fillna(100, inplace=True)
    # df_2['2 Year Return'].fillna(100, inplace=True)
    # df_2['3 Year Return'].fillna(100, inplace=True)

    df_3 = df_2.copy()

    df_3['1 Year Rank'] = df_3.groupby(['SR Index', 'Date'])['1 Year Return'].rank(ascending=False)

    df_3['2 Year Rank'] = df_3.groupby(['SR Index', 'Date'])['2 Year Return'].rank(ascending=False)

    df_3['3 Year Rank'] = df_3.groupby(['SR Index', 'Date'])['3 Year Return'].rank(ascending=False)

    df_3.to_csv('C:/Users/mnguyen/LGSS/Investments Team - SandPits - SandPits/data/input/vendors/superratings/superratings_2years_data.csv', index=False)

    df_3 = df_3[df_3['Date'].isin([dt.datetime(2021, 4, 30)])].sort_values(['Option Name', 'Product Type'], ascending=[True, True])

    df_4 = pd.merge(left=df_1_a, right=df_3, left_on=['Option Name', 'Product Type', 'SR Index', 'Date'], right_on=['Option Name', 'Product Type', 'SR Index', 'Date'], how='inner').reset_index(drop=True)

    df_4_test = df_4.copy()
    df_4_test['1 Year Return'] = [round(x * 100, 2) for x in df_4_test['1 Year Return']]
    df_4_test['2 Year Return'] = [round(x * 100, 2) for x in df_4_test['2 Year Return']]
    df_4_test['3 Year Return'] = [round(x * 100, 2) for x in df_4_test['3 Year Return']]
    df_4_test['Rolling 1 Year %'] = [string_to_float(x) for x in df_4_test['Rolling 1 Year %']]
    df_4_test['Rolling 3 Year %'] = [string_to_float(x) for x in df_4_test['Rolling 3 Year %']]
    df_4_test['1 Year % Diff'] = df_4_test['Rolling 1 Year %'] - df_4_test['1 Year Return']
    df_4_test['3 Year % Diff'] = df_4_test['Rolling 3 Year %'] - df_4_test['3 Year Return']
    df_4_test['1 Year Rank Diff'] = df_4_test['Rolling 1 Year Rank'] - df_4_test['1 Year Rank']
    df_4_test['3 Year Rank Diff'] = df_4_test['Rolling 3 Year Rank'] - df_4_test['3 Year Rank']
    df_4_test.sort_values(['Option Name', 'Product Type', 'SR Index'])
    df_4_test_SR_1_Year = df_4_test.groupby(['SR Index', 'Date'])['Rolling 1 Year %'].count()
    df_4_test_AS_1_Year = df_4_test.groupby(['SR Index', 'Date'])['1 Year Return'].count()
    df_4_test_SR_3_Year = df_4_test.groupby(['SR Index', 'Date'])['Rolling 3 Year %'].count()
    df_4_test_AS_3_Year = df_4_test.groupby(['SR Index', 'Date'])['3 Year Return'].count()
    df_4_test_SR_AS = pd.concat([df_4_test_SR_1_Year, df_4_test_AS_1_Year, df_4_test_SR_3_Year, df_4_test_AS_3_Year], axis=1).reset_index(drop=False)
    df_4_test_SR_AS['Diff 1 Year'] = df_4_test_SR_AS['Rolling 1 Year %'] - df_4_test_SR_AS['1 Year Return']
    df_4_test_SR_AS['Diff 3 Year'] = df_4_test_SR_AS['Rolling 3 Year %'] - df_4_test_SR_AS['3 Year Return']
    df_4_test.to_csv('C:/Users/mnguyen/LGSS/Investments Team - SandPits - SandPits/data/output/vendors/superratings/SuperRatingsTest.csv', index=False)
    df_4_test_SR_AS.to_csv('C:/Users/mnguyen/LGSS/Investments Team - SandPits - SandPits/data/output/vendors/superratings/SuperRatingsTestSampleSize.csv', index=False)

    # df_5 = df_4[df_4['Date'].isin([dt.datetime(2021, 4, 30)])].sort_values(['Option Name', '1 Year Return'])

    df_6 = df_3[['Option Name', 'Date', 'SR Index', '2 Year Return', '2 Year Rank']].rename(columns={'2 Year Return': 'Rolling 2 Year %', '2 Year Rank': 'Rolling 2 Year Rank'})

    df_6['Rolling 2 Year %'] = [round(x*100, 2) for x in df_6['Rolling 2 Year %']]

    df_6.to_csv('C:/Users/mnguyen/LGSS/Investments Team - SandPits - SandPits/data/input/vendors/superratings/superratings_2years.csv', index=False)

    df_7 = pd.merge(
        left=df_0_a,
        right=df_6,
        left_on=['Option Name', 'SR Index', 'Date'],
        right_on=['Option Name', 'SR Index', 'Date'],
        how='inner'
    )

    df_8 = df_7[df_7['Option Name'].isin(select_list)].reset_index(drop=True)

    df_8_HG = df_8[df_8['Option Name'].isin(['Active Super - High Growth'])][['Option Name', 'Rolling 7 Year %', 'Rolling 7 Year Rank']].rename(columns={'Rolling 7 Year %': 'Return', 'Rolling 7 Year Rank': 'Rank'})

    df_8_BG = df_8[df_8['Option Name'].isin(['Active Super - Balanced Growth'])][['Option Name', 'Rolling 5 Year %', 'Rolling 5 Year Rank']].rename(columns={'Rolling 5 Year %': 'Return', 'Rolling 5 Year Rank': 'Rank'})

    df_8_BA = df_8[df_8['Option Name'].isin(['Active Super - Balanced'])][['Option Name', 'Rolling 3 Year %', 'Rolling 3 Year Rank']].rename(columns={'Rolling 3 Year %': 'Return', 'Rolling 3 Year Rank': 'Rank'})

    df_8_CO = df_8[df_8['Option Name'].isin(['Active Super - Conservative'])][['Option Name', 'Rolling 2 Year %', 'Rolling 2 Year Rank']].rename(columns={'Rolling 2 Year %': 'Return', 'Rolling 2 Year Rank': 'Rank'})

    df_8_MC = df_8[df_8['Option Name'].isin(['Active Super - Managed Cash'])][['Option Name', 'Rolling 2 Year %', 'Rolling 2 Year Rank']].rename(columns={'Rolling 2 Year %': 'Return', 'Rolling 2 Year Rank': 'Rank'})

    df_9 = pd.concat([df_8_HG, df_8_BG, df_8_BA, df_8_CO, df_8_MC]).reset_index(drop=True)

    horizons = ['7 Year', '5 Year', '3 Year', '2 Year', '2 Year']

    df_10 = df_9.copy()

    df_10['Horizon'] = horizons

    df_10 = df_10[['Option Name', 'Horizon', 'Return', 'Rank']]

    option_names = [x.split("-")[1] for x in df_10['Option Name']]

    df_10['Option Name'] = option_names

    with open('C:/Users/mnguyen/LGSS/Investments Team - SandPits - SandPits/data/output/lgs/reports/superratings/summary/summary.tex','w') as tf:

        tf.write(df_10.to_latex(index=False, na_rep='', multicolumn_format='c', escape=False, float_format="{:0.2f}".format))

    """
    df_11 = df_10[['Option Name', 'Return']].set_index('Option Name')

    df_11['Return'] = [string_to_float(str(x)) for x in df_11['Return']]

    df_11 = df_11.sort_values(['Return'], ascending=[True])

    df_11.plot.barh(color=[middlegreen])
    """

