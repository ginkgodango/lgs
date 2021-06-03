import datetime as dt
import math as math
import numpy as np
import os as os
import pandas as pd


def string_to_float(x):

    return 0 if x == '-' else float(x.replace(',', ''))


if __name__ == '__main__':

    folderPath = 'C:/Users/mnguyen/LGSS/Investments Team - SandPits - SandPits/data/input/vendors/superratings/All/'

    df_1 = pd.read_csv(folderPath + 'SuperRatings-FCRS-202104.csv', parse_dates=['Date'])[['Option Name', 'Product Type', 'Date', 'SR Index', 'Rolling 3 Year %']]

    df_2 = pd.read_csv(folderPath + 'SuperRatings-FCRS-201904.csv', parse_dates=['Date'])[['Option Name', 'Product Type', 'Date', 'SR Index', 'Rolling 1 Year %']]

    df_3 = pd.merge(
        left=df_1,
        right=df_2,
        left_on=['Option Name', 'Product Type', 'SR Index'],
        right_on=['Option Name', 'Product Type', 'SR Index'],
        how='inner'
    )

    df_3['R_0_3'] = [string_to_float(str(x)) / 100 for x in df_3['Rolling 3 Year %']]

    df_3['R_2_3'] = [string_to_float(str(x)) / 100 for x in df_3['Rolling 1 Year %']]

    df_3['(1 + R_0_3)^3'] = [(1 + x)**3 for x in df_3['R_0_3']]

    df_3['(1 + R_2_3)^1'] = [(1 + x) ** 1 for x in df_3['R_2_3']]

    df_3['(1 + R_0_2)^2'] = df_3['(1 + R_0_3)^3'] / df_3['(1 + R_2_3)^1']

    df_3['R_0_2'] = [math.sqrt(x) - 1 for x in df_3['(1 + R_0_2)^2']]

    df_3['2 Year Rank'] = df_3.groupby(['SR Index'])['R_0_2'].rank(ascending=False)

    df_3.to_csv('C:/Users/mnguyen/LGSS/Investments Team - SandPits - SandPits/data/output/vendors/superratings/SolveSuperRatingsTest.csv', index=False)
