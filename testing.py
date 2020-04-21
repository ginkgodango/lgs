import os
import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

# START USER INPUT DATA
jpm_main_filepath = 'U:/CIO/#Data/input/jpm/performance/2020/03/Historical Time Series - Monthly - Main Returns and Benchmarks.xlsx'

FYTD = 9
report_date = dt.datetime(2020, 3, 31)
# END USER INPUT DATA
use_managerid = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12]
use_accountid = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
footnote_rows = 28
df_jpm_main = pd.read_excel(
        pd.ExcelFile(jpm_main_filepath),
        sheet_name='Sheet1',
        skiprows=use_managerid,
        skipfooter=footnote_rows,
        header=1
        )
# df_jpm_main = df_jpm_main.rename(columns={'Unnamed: 0': 'Date'})
# df_jpm_main = df_jpm_main.set_index('Date')
# df_jpm_main = df_jpm_main.transpose()
# df_jpm_main = df_jpm_main.reset_index(drop=False)
# df_jpm_main = df_jpm_main.rename(columns={'index': 'Manager'})
# df_jpm_main = pd.melt(df_jpm_main, id_vars=['Manager'], value_name='Values')
# df_jpm_main = df_jpm_main.sort_values(['Manager', 'Date'])
# df_jpm_main = df_jpm_main.reset_index(drop=True)
# df_jpm_main = df_jpm_main.replace('-', np.NaN)