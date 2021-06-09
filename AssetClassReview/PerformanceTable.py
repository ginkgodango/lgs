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


if __name__ == "__main__":
    folder_path = "C:/Users/mnguyen/LGSS/Investments Team - SandPits - SandPits/data/output/lgs/investment/checker/"
    file_path = folder_path + "lgs_combined.csv"
    df_0 = pd.read_csv(file_path, parse_dates=['Date'])

    df_1 = df_0[
        [
            'Manager',
            'Date',
            'LGS Asset Class Level 1',
            'LGS Asset Class Level 2',
            'FYTD_Return',
            'FYTD_Benchmark',
            '12_Benchmark',
            '36_Return',
            '36_Benchmark',
            '60_Return',
            '60_Benchmark',
            '84_Return',
            '84_Benchmark'
        ]
    ]

    df_2 = df_1[df_1['Date'] == dt.datetime(2021, 4, 30)]

    groupby_dict = dict(list(df_2.groupby(['LGS Asset Class Level 1'])))

    df_2 = df_2[df_2['LGS Asset Class Level 1'].isin(['AR'])]

