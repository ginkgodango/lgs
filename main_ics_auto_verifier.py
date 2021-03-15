import os
import pandas as pd
import numpy as np

# Reads in JPM to ICS map excel file in the map_file_path folder location and uses the sheet name map_sheet_name
map_sheet_name = 'JPM_HLD_IA'
map_file_path = 'C:/Users/Mnguyen/LGSS/Investments Team - SandPits - SandPits/Email Messages/edm/holdings/map/'
map_file_names = sorted(os.listdir(map_file_path))
df_map = pd.read_excel(pd.ExcelFile(map_file_path + map_file_names[0]), sheet_name=map_sheet_name, skiprows=[0, 1, 2])
df_map_active_columns = df_map[~df_map['Source Column'].isin([np.nan]) & ~df_map['Destination Column'].isin([np.nan])]

# Reads in JPM files in the jpm_file_path folder location and then concatenates the JPM files into a single data frame.
jpm_file_path = 'C:/Users/Mnguyen/LGSS/Investments Team - SandPits - SandPits/Email Messages/edm/holdings/jpm/'
jpm_file_names = sorted(os.listdir(jpm_file_path))
df_jpm = pd.concat([pd.read_csv(jpm_file_path + jpm_file_name) for jpm_file_name in jpm_file_names])

# Reads in ICS files in the ics_file_path folder location and then concatenates the ICS files into a single data frame.
ics_file_path = 'C:/Users/Mnguyen/LGSS/Investments Team - SandPits - SandPits/Email Messages/edm/holdings/ics/'
ics_file_names = sorted(os.listdir(ics_file_path))
df_ics = pd.concat([pd.read_excel(pd.ExcelFile(ics_file_path + ics_file_name)) for ics_file_name in ics_file_names])


df_jpm_1 = df_jpm.pivot_table(df_jpm, index=['Valuation Date', 'Portfolio ID', 'Security ID']).reset_index(drop=False)
df_jpm_2 = pd.melt(df_jpm.pivot_table(df_jpm, index=['Valuation Date', 'Portfolio ID', 'Security ID']).reset_index(drop=False), index)
df_jpm_3 = pd.melt(df_jpm.pivot_table(df_jpm, index=['Valuation Date', 'Portfolio ID', 'Security ID']).T)
