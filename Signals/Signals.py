import datetime as dt
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

folder_path = "C:/Users/Mnguyen/LGSS/Investments Team - SandPits - SandPits/data/input/lgs/signal/"
filename = "signal_data.xlsx"
start_date = dt.datetime(2006, 6, 30)

df_index_0 = pd.read_excel(pd.ExcelFile(folder_path + filename), sheet_name="Sheet1")

df_data_0 = pd.read_excel(pd.ExcelFile(folder_path + filename), sheet_name="Sheet2")

date_list = ['Date']

signal_list = [
    'PMI', '10yr_Yield', 'A_Credit_Spread', 'CPI_yoy',
    'AUD_Rate', 'Oil_Price', 'Ted_Spread', 'VIX',
    'AE_Valuation_I', 'BO_Valuation_I', 'FX_Valuation_I', 'IE_Valuation_I',
    'Yield_Curve', 'Excess_Liquidity', 'High_Yield_Spread'
]

signal_change_list = [
    'PMI', '10yr_Yield', 'A_Credit_Spread', 'CPI_yoy',
    'AUD_Rate', 'Oil_Price', 'Ted_Spread', 'VIX',
    'High_Yield_Spread'
]

signal_change_diff_list = [column + "_Diff" for column in signal_change_list]

signal_change_diff_lag_list = [column + "_Lag" for column in signal_change_diff_list]

signal_level_list = [
    'AE_Valuation_I', 'BO_Valuation_I', 'FX_Valuation_I', 'IE_Valuation_I',
    'Yield_Curve', 'Excess_Liquidity'
]

signal_level_lag_list = [column + "_Lag" for column in signal_level_list]

signal_final_list = signal_level_lag_list + signal_change_diff_lag_list

df_data_change_0 = df_data_0[date_list + signal_change_list]

df_data_change_1 = df_data_change_0.diff()

df_data_change_2 = df_data_change_1.copy()[signal_change_list]

df_data_change_2.columns = signal_change_diff_list

df_data_level_0 = df_data_0[date_list + signal_level_list]

df_data_1 = pd.concat([df_data_level_0, df_data_change_2], axis=1)

df_data_2 = df_data_1.copy()

df_data_2['Date'] = df_data_2['Date'].shift(-1)

df_data_2.columns = date_list + signal_final_list

df_signal_0 = pd.merge(
    left=df_index_0,
    right=df_data_2,
    left_on=['Date'],
    right_on=['Date'],
    how='outer'
)

df_signal_1 = df_signal_0[df_signal_0['Date'] >= start_date].reset_index(drop=True)

df_signal_ae = df_signal_1[date_list + ["ASX200_Return"] + signal_final_list]


def positive_indicator(y, signal):

    return y if signal >= 0 else 0


def negative_indicator(y, signal):

    return y if signal < 0 else 0


for column in signal_final_list:
    temp_list_positive = []
    temp_list_negative = []
    for i in range(len(df_signal_ae)):
        y = df_signal_ae['ASX200_Return'][i]
        signal = df_signal_ae[column][i]
        temp_list_positive.append(positive_indicator(y, signal))
        temp_list_negative.append(negative_indicator(y, signal))
    df_signal_ae[column + "_P"] = temp_list_positive
    df_signal_ae[column + "_N"] = temp_list_negative
    df_signal_ae[column + "LP_SN"] = df_signal_ae[column + "_P"] - df_signal_ae[column + "_N"]
    df_signal_ae[column + "LN_SP"] = df_signal_ae[column + "_N"] - df_signal_ae[column + "_P"]

