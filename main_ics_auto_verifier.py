import os
import pandas as pd
import numpy as np
from dateutil import parser

# User variable set up.
# Note: The script auto reads in all files in the file_path location. There is no need to specify filename.
map_file_path = 'C:/Users/Mnguyen/LGSS/Investments Team - SandPits - SandPits/Email Messages/edm/holdings/map/'
jpm_file_path = 'C:/Users/Mnguyen/LGSS/Investments Team - SandPits - SandPits/Email Messages/edm/holdings/jpm/'
ics_file_path = 'C:/Users/Mnguyen/LGSS/Investments Team - SandPits - SandPits/Email Messages/edm/holdings/ics/'
out_file_path = 'C:/Users/Mnguyen/LGSS/Investments Team - SandPits - SandPits/Email Messages/edm/holdings/out/'
map_sheet_name = 'JPM_HLD_IA'
jpm_keys = ['Valuation Date', 'Portfolio ID', 'SecurityCode', 'SourceCode', 'ValuationTypeId']
ics_keys = ['AsAtDate', 'PortfolioCode', 'SecurityCode', 'SourceCode', 'ValuationTypeId']
tolerance_level = 0.01
# End user variable set up

# Reads in JPM to ICS map excel file in the map_file_path folder location and uses the sheet name map_sheet_name
map_file_names = sorted(os.listdir(map_file_path))
df_map = pd.read_excel(pd.ExcelFile(map_file_path + map_file_names[0]), sheet_name=map_sheet_name, skiprows=[0, 1, 2])
df_map_active_columns = df_map[~df_map['Source Column'].isin([np.nan]) & ~df_map['Destination Column'].isin([np.nan])][['Destination Column', 'Source Column']]

# Reads in JPM files in the jpm_file_path folder location and then concatenates the JPM files into a single data frame.
jpm_file_names = sorted(os.listdir(jpm_file_path))
df_jpm = pd.concat([pd.read_csv(jpm_file_path + jpm_file_name) for jpm_file_name in jpm_file_names])

# Reads in ICS files in the ics_file_path folder location and then concatenates the ICS files into a single data frame.
ics_file_names = sorted(os.listdir(ics_file_path))
df_ics = pd.concat([pd.read_excel(pd.ExcelFile(ics_file_path + ics_file_name)) for ics_file_name in ics_file_names])

# START ICS CALCULATED VALUES
# ISNULL(IIF([Category Code] = 'ZF', [Security ID] + ISNULL(SUBSTRING([Security Name],1,1),''),
# IIF([Category Code] = 'ZL', ISNULL([Symbol],[Security ID]),[Security ID])),'TBA')
security_code_calculated = []
for i in range(len(df_jpm)):
    if df_jpm['Category Code'][i] == 'ZF':
        if str(df_jpm['Security Name'][i]) != 'nan':
            security_code_calculated.append(df_jpm['Security ID'][i] + df_jpm['Security Name'][i][:1])
        else:
            security_code_calculated.append(df_jpm['Security ID'][i])
    elif df_jpm['Category Code'][i] == 'ZL':
        if str(df_jpm['Symbol'][i]) != 'nan':
            security_code_calculated.append(df_jpm['Symbol'][i])
        else:
            security_code_calculated.append(df_jpm['Security ID'][i])
    elif str(df_jpm['Security ID'][i]) == 'nan' and str(df_jpm['Symbol'][i]) == 'nan':
        security_code_calculated.append('TBA')
    else:
        security_code_calculated.append(df_jpm['Security ID'][i])
df_jpm['SecurityCode'] = security_code_calculated

# "IIF([Category Code] = 'FI',[Sector Level 0 Name],
# IIF([Category Subtype Code] = 'ZL01',[Sector Level 0 Name],
# IIF([Category Subtype Code] = 'CO00',NULL,
# IIF([Category Subtype Code] = 'PO00',NULL,
# IIF([Category Code] = 'DS',[Security Type 1],
# IIF([Sector Level 0 Name] IN ('Hedge Fund','Private Equity','Unit Trust','Unlisted Prop Trust'),[Sector Level 0 Name],[Category Subtype Code]))))))"
security_instrument_type_code = []
for i in range(len(df_jpm)):
    if df_jpm['Category Code'][i] == 'FI':
        security_instrument_type_code.append(df_jpm['Sector Level 0 Name'][i])
    elif df_jpm['Category Subtype Code'][i] == 'ZL01':
        security_instrument_type_code.append(df_jpm['Sector Level 0 Name'][i])
    elif df_jpm['Category Subtype Code'][i] in ['CO00', 'PO00']:
        security_instrument_type_code.append(np.nan)
    elif df_jpm['Category Code'][i] == 'DS':
        security_instrument_type_code.append(df_jpm['Security Type 1'][i])
    elif df_jpm['Sector Level 0 Name'][i] in ['Hedge Fund', 'Private Equity', 'Unit Trust', 'Unlisted Prop Trust']:
        security_instrument_type_code.append(df_jpm['Sector Level 0 Name'][i])
    else:
        security_instrument_type_code.append(df_jpm['Category Subtype Code'][i])
df_jpm['SecurityInstrumentTypeCode'] = security_instrument_type_code
# END ICS CALCULATED VALUES


# START ICS TRANSFORMATION
# CASE WHEN SecurityInstrumentTypeCode = 'Unit Trust' and left(SecurityCode,2) = 'LG' and right(SecurityCode,2) = 'UP' then left(SecurityCode,4) else SecurityCode end
security_code_transformed = []
for i in range(len(df_jpm)):
    if df_jpm['SecurityInstrumentTypeCode'][i] == 'Unit Trust' and df_jpm['SecurityCode'][i][:2] == 'LG' and df_jpm['SecurityCode'][i][-2:] == 'UP':
        security_code_transformed.append(df_jpm['SecurityCode'][i][:4])
    else:
        security_code_transformed.append(df_jpm['SecurityCode'][i])
df_jpm['SecurityCode'] = security_code_transformed

#CASE WHEN LoadText07 LIKE ('%HC%') THEN 'JPMIA_HC' ELSE 'JPMIA' END
source_code = []
for i in range(len(df_jpm)):
    if 'HC' in df_jpm['Valuation Type'][i]:
        source_code.append('JPMIA_HC')
    else:
        source_code.append('JPMIA')
df_jpm['SourceCode'] = source_code

# CASE WHEN LoadText07 LIKE ('%HC%') THEN '1' ELSE '0' END
valuation_type_id = []
for i in range(len(df_jpm)):
    if 'HC' in df_jpm['Valuation Type'][i]:
        valuation_type_id.append(1)
    else:
        valuation_type_id.append(0)
df_jpm['ValuationTypeId'] = valuation_type_id
# END TRANSFORMATION

# Remove whitespaces in ICS column headers
ics_columns_remove_whitespace = [x.replace(" ", "") for x in df_ics.columns]
df_ics.columns = ics_columns_remove_whitespace

# Checks length of dataframes between JPM and ICS
print('JPM has', len(df_jpm) - len(df_ics), 'more rows then ICS')

# Pivots on the keys and melts the dataframe from wide to long shape.
df_jpm_1 = pd.melt(df_jpm.pivot_table(df_jpm, index=jpm_keys).reset_index(drop=False), id_vars=jpm_keys)
df_ics_1 = pd.melt(df_ics.pivot_table(df_ics, index=ics_keys).reset_index(drop=False), id_vars=ics_keys)

# Keeps on columns used in 'Source Column' and 'Destination Column'.
df_jpm_2 = df_jpm_1[df_jpm_1['variable'].isin(df_map_active_columns['Source Column'])]
df_ics_2 = df_ics_1[df_ics_1['variable'].isin(df_map_active_columns['Destination Column'])]

# Converts dates to same format
df_jpm_2['Valuation Date'] = [parser.parse(str(x)).date() for x in df_jpm_2['Valuation Date']]
df_ics_2['AsAtDate'] = [parser.parse(str(x)).date() for x in df_ics_2['AsAtDate']]

# Merges keys onto df_jpm
df_jpm_map = pd.merge(
    left=df_jpm_2,
    right=df_map_active_columns,
    left_on='variable',
    right_on='Source Column',
    how='inner'
)

# Merges JPM and ICS dataframes
df_jpm_map_ics = pd.merge(
    left=df_jpm_map,
    right=df_ics_2,
    left_on=jpm_keys + ['Destination Column'],
    right_on=ics_keys + ['variable'],
    how='outer',
    indicator=True
)

df_jpm_map_ics_merge_sucess = df_jpm_map_ics[df_jpm_map_ics['_merge'].isin(['both'])].reset_index(drop=True)
df_jpm_map_ics_merge_fail = df_jpm_map_ics[~df_jpm_map_ics['_merge'].isin(['both'])]
df_jpm_map_ics_merge_fail_unique_keys = df_jpm_map_ics_merge_fail[jpm_keys].drop_duplicates().reset_index(drop=True)
print('Merge sucess is', str(round(len(df_jpm_map_ics_merge_sucess)/min(len(df_jpm_map), len(df_ics_2)) * 100, 2)) + '% of maximum merge sucess possible.')

# Checks for deviations between JPM and ICS for the sucessfully merged dataframe
value_match = []
absolute_deviation = []
for i in range(len(df_jpm_map_ics_merge_sucess)):
    jpm_value = df_jpm_map_ics_merge_sucess['value_x'][i]
    ics_value = df_jpm_map_ics_merge_sucess['value_y'][i]
    if isinstance(jpm_value, (int, float)) and isinstance(ics_value, (int, float)):
        if abs(jpm_value - ics_value) >= tolerance_level:
            value_match.append(0)
            absolute_deviation.append(abs(jpm_value - ics_value))
        else:
            value_match.append(1)
            absolute_deviation.append(abs(jpm_value - ics_value))
    else:
        if str(jpm_value) == str(ics_value):
            value_match.append(1)
            absolute_deviation.append(np.nan)
        else:
            value_match.append(0)
            absolute_deviation.append(np.nan)
print("Value match accuracy for sucessfully merged rows is:", str(round(np.average(value_match) * 100, 2)) + '%')

# Adds the value match column to df_jpm_key_ics_merge_sucess.
df_jpm_map_ics_merge_sucess['value_match'] = value_match
df_jpm_map_ics_merge_sucess['absolute_deivation'] = absolute_deviation

# Finds rows that did not have a value match but were sucessfully merged between JPM and ICS.
df_jpm_map_ics_merge_sucess_value_match_fail = df_jpm_map_ics_merge_sucess[df_jpm_map_ics_merge_sucess['value_match'].isin([0])]

# Writes dataframes to excel
writer = pd.ExcelWriter(out_file_path + 'auto_verifier.xlsx', engine='xlsxwriter')
df_jpm_map_ics_merge_sucess_value_match_fail.to_excel(writer, sheet_name='jpm_ics_merge_sucess_value_fail', index=False)
df_jpm_map_ics_merge_sucess.to_excel(writer, sheet_name='jpm_ics_merge_sucess', index=False)
df_jpm_map_ics_merge_fail_unique_keys.to_excel(writer, sheet_name='jpm_ics_merge_fail_unique_keys', index=False)
df_jpm_map_ics_merge_fail.to_excel(writer, sheet_name='jpm_ics_merge_fail', index=False)
df_jpm_map.to_excel(writer, sheet_name='jpm_key', index=False)
df_jpm_2.to_excel(writer, sheet_name='jpm', index=False)
df_ics_2.to_excel(writer, sheet_name='ics', index=False)
df_map.to_excel(writer, sheet_name='map', index=False)
writer.save()
writer.close()
