# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 16:02:28 2019

@author: MerrillN
"""
import pandas as pd

# reads the jpm holdings csv into a dataframe df
df = pd.read_csv('U:/CIO/#Holdings/Data/input/holdings/jpm/2020/01/Priced Positions - All.csv', header=3)

# Subset the dataframe to only have Account Name, Security Name, and ISIN
df_isin = df[['Account Name', 'Security Name', 'ISIN']]

# Places each dataframe from the groupby into a dictionary structure
account_to_dataframe_dict = dict(list(df_isin.groupby('Account Name')))

# Saves the dataframes into separate files on our disk
for account, dataframe in account_to_dataframe_dict.items():

    dataframe = dataframe[['ISIN']]

    dataframe.to_csv('U:/CIO/#Holdings/Data/output/msci/' + account + '.csv', index=False)
