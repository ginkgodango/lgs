import pandas as pd
import numpy as np
import datetime as dt

current_month = dt.datetime(2019, 9, 30)
nof_filepath = 'U:/CIO/#Investment_Report/Data/input/returns/20190930 Historical Time Series.xlsx'
gof_filepath = 'U:/CIO/#Investment_Report/Data/input/returns/20190930 Historical Time Series GOF.xlsx'

# Imports the JPM time-series.
nof_xlsx = pd.ExcelFile(nof_filepath)
gof_xlsx = pd.ExcelFile(gof_filepath)
use_managerid = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11]
use_accountid = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14]
footnote_rows = 28

df_nof = pd.read_excel(
        nof_xlsx,
        sheet_name='Sheet1',
        skiprows=use_accountid,
        skipfooter=footnote_rows,
        header=1
        )

df_gof = pd.read_excel(
        gof_xlsx,
        sheet_name='Sheet1',
        skiprows=use_accountid,
        skipfooter=footnote_rows,
        header=1
        )

df_link = pd.read_csv('U:/CIO/#Investment_Report/Data/input/link/assetclass_dictionary.csv')

df_link = df_link[['Book Name', 'Account Id']]

def ts_to_panel(df, return_type):
    # Reshapes the time-series into a panel.
    df = df.rename(columns={'Unnamed: 0': 'Date'})
    df = df.set_index('Date')
    df = df.transpose()
    df = df.reset_index(drop=False)
    df = df.rename(columns={'index': 'Account Id'})
    df = pd.melt(df, id_vars=['Account Id'], value_name=return_type)
    df = df.sort_values(['Account Id', 'Date'])
    df = df.reset_index(drop=True)
    df = df.replace('-', np.nan)
    return df


df_nof = ts_to_panel(df_nof, 'GOF')
df_gof = ts_to_panel(df_gof, 'NOF')

df_cost = pd.merge(
    left=df_gof,
    right=df_nof,
    left_on=['Account Id', 'Date'],
    right_on=['Account Id', 'Date'],
    how='inner'
)

df_cost['Fee'] = df_cost['NOF'] - df_cost['GOF']

df_cost = pd.merge(
    left=df_link,
    right=df_cost,
    left_on=['Account Id'],
    right_on=['Account Id'],
    how='inner'
)

df_cost.to_csv('U:/CIO/#Investment_Report/Data/output/fees/fees_' + str(current_month.date()) + '.csv', index=False)

df_cost_current_month = df_cost[df_cost['Date'] == current_month].reset_index(drop=True)

df_cost_current_month = df_cost_current_month.dropna().reset_index(drop=True)

df_cost_current_month.to_csv('U:/CIO/#Investment_Report/Data/output/fees/fees_current_' + str(current_month.date()) + '.csv', index=False)

