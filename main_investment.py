import datetime as dt
import pandas as pd
import investment.extraction

directory = 'D:/automation/final/investment/2019/04/'
update_filename = 'LGSS Preliminary Performance April 2019_Addkeys.xlsx'
returns_filename = 'returns_NOF_201903.csv'
market_values_filename = 'market_values_NOF_201903.csv'
table_filename = 'table_NOF_201903.csv'

report_date = dt.datetime(2019, 4, 30)
MTD = 10
days_in_month = 30
SSgACUSTOM_Index = 2.32
EOAS_Index = 3.36
MXEF_Index = 3.05


"""
Loads the new JPM monthly report
"""
df_update = investment.extraction.load_update(directory + update_filename)
df_update = investment.extraction.add_report_date(df_update, report_date)
df_update = investment.extraction.clean(df_update)
# df_update.to_csv(directory + 'df_update_' + str(report_date.date()) + '.csv', index=True)

"""
Adds new month returns to existing return time-series. 
"""
df_returns = investment.extraction.load_returns(directory + returns_filename)
missing_returns_list = investment.extraction.update_check_missing_returns(df_returns, df_update)
new_returns_list = investment.extraction.update_check_new_returns(df_returns, df_update)
update_dict = investment.extraction.create_update_dict(df_update, days_in_month, SSgACUSTOM_Index, EOAS_Index, MXEF_Index)
df_updater = investment.extraction.update_dict_to_df(update_dict, report_date)
df_returns = investment.extraction.apply_update_to_df_returns(df_returns, df_updater)
# df_returns.to_csv(directory + 'returns_NOF_201904.csv')

"""
Adds new month market values to existing market values time-series. 
"""
df_market_values = investment.extraction.load_market_values(directory + market_values_filename)
missing_market_values_list = investment.extraction.update_check_missing_market_values(df_market_values, df_update)
new_market_values_list = investment.extraction.update_check_new_market_values(df_market_values, df_update)
update_market_values_dict = investment.extraction.create_update_market_value_dict(df_update)
df_updater_market_values = investment.extraction.update_market_values_dict_to_df(update_market_values_dict, report_date)
df_market_values = investment.extraction.apply_update_to_df_market_values(df_market_values, df_updater_market_values)
# df_market_values.to_csv(directory + 'market_values_NOF_201904.csv')

"""
Adds FX sector return and market value
"""
df_returns['FX'] = df_returns['IECurrencyOverlay_IE']
df_market_values['FX'] = df_market_values['IECurrencyOverlay_IE']

