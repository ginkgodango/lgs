import datetime as dt
import os as os
import pandas as pd
import numpy as np


def file_name_to_date(s):
    string_date = s.split("_")[1]
    int_year = int(string_date[0:4])
    int_month = int(string_date[4:6])
    int_day = int(string_date[6:8])
    return dt.datetime(int_year, int_month, int_day).date()


path = "C:/Users/Mnguyen/LGSS/Investments Team - SandPits - SandPits/data/input/vendors/Inalytics/"

file_names = sorted(os.listdir(path))

list_df = []

for file_name in file_names:

    df_0 = pd.read_csv(path + file_name, skiprows=[0, 1, 2, 3])

    df_0.insert(0, "Date", file_name_to_date(file_name))

    list_df.append(df_0)

df_1 = pd.concat(list_df)

longview = "EBV53"

df_2 = df_1[df_1["Account Number"].isin([longview])]

df_2.to_csv("C:/Users/Mnguyen/LGSS/Investments Team - SandPits - SandPits/data/output/vendors/inalytics/sample_portfolio.csv", index=False)
