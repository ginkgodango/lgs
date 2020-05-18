import datetime as dt
import numpy as np
import pandas as pd

lgs_filepath = 'U:/CIO/#Data/output/investment/checker/lgs_table.csv'
jpm_filepath = 'U:/CIO/#Data/input/jpm/report/investment/LGSS Preliminary Performance 202004.xlsx'
FYTD = 10
report_date = dt.datetime(2020, 4, 30)

# Reads LGS Tables
df_lgs = pd.read_csv(lgs_filepath)

# Reads JPM Performance Report
df_jpm = pd.DataFrame()
sheet_to_columns_dict = {
    'Page 3 NOF': 'A:N',
    'Page 5 NOF': 'B:O',
    'Page 6 NOF': 'B:O',
    'Page 7 NOF': 'B:O',
    'Page 8': 'D:O'
}
for sheet, columns in sheet_to_columns_dict.items():
    print('Accessing:', sheet)
    df_sheet = pd.read_excel(
        pd.ExcelFile(jpm_filepath),
        sheet_name=sheet,
        usecols=columns,
        skiprows=[0, 1, 2]
    )
    df_sheet = df_sheet.rename(
        columns={
            'Unnamed: 0': 'ModelCode',
            'Unnamed: 1': 'JPM ReportName',
            'Unnamed: 2': 'JPM ReportName',
        }
    )

    if sheet == 'Page 8':
        df_sheet = df_sheet.rename(
            columns={
                'Unnamed: 0': 'ModelCode',
                'Unnamed: 4': 'JPM ReportName',
            }
        )

    df_jpm = pd.concat([df_jpm, df_sheet], sort=False)

df_jpm = df_jpm.reset_index(drop=True)
df_jpm = df_jpm.replace('-', np.nan)
df_jpm = df_jpm.drop(columns=['ModelCode'], axis=1)
df_jpm['Market Value'] = (df_jpm['Market Value']/1000000).round(2)

# Reads footers and removes them
df_footers = pd.read_excel('U:/CIO/#Investment_Report/Data/input/testing/20191031 Footers.xlsx')

remove_items = list(df_footers['Footers']) + [np.nan, 'Excess return']
df_jpm = df_jpm[~df_jpm['JPM ReportName'].isin(remove_items)].reset_index(drop=True)

df_lgs_jpm = pd.merge(
    left=df_lgs,
    right=df_jpm,
    on=['JPM ReportName'],
    how='outer'
)

df_later = df_lgs_jpm[df_lgs_jpm['Manager'].isin([np.nan])].reset_index(drop=True)
df_lgs_jpm = df_lgs_jpm[~df_lgs_jpm['Manager'].isin([np.nan])].reset_index(drop=True)

lgscolumn_to_jpmcolumn_dict = {
    'Market Value_x': 'Market Value_y',
    '1_Return': '1 Month',
    '3_Return': '3 Months',
    'FYTD_Return': 'FYTD',
    '12_Return': '1 Year',
    '36_Return': '3 Years',
    '60_Return': '5 Years',
    '84_Return': '7 Years'
}

df_deviations = pd.DataFrame()
deviants = []
columns = []
deviations = []
jpm_missing = []
lgs_missing = []
total_count = 0
deviant_count = 0
for lgscolumn, jpmcolumn in lgscolumn_to_jpmcolumn_dict.items():

    for i in range(0, len(df_lgs_jpm)):
        deviation = df_lgs_jpm[lgscolumn][i] - df_lgs_jpm[jpmcolumn][i]

        if deviation >= 0.01:
            deviants.append(df_lgs_jpm['Manager'][i])
            columns.append(jpmcolumn)
            deviations.append(deviation)
            deviant_count += 1

        if (not pd.isna(df_lgs_jpm[jpmcolumn][i])) and (pd.isna(df_lgs_jpm[lgscolumn][i])):
            lgs_missing.append((df_lgs_jpm['Manager'][i], lgscolumn))

        if (pd.isna(df_lgs_jpm[jpmcolumn][i])) and (not pd.isna(df_lgs_jpm[lgscolumn][i])):
            jpm_missing.append((df_lgs_jpm['JPM ReportName'][i], jpmcolumn))

        total_count += 1

# Fixes the column names
columns_fix = []
for column in columns:
    if column == 'Market Value_y':
        columns_fix.append('Market Value')
    else:
        columns_fix.append(column)

df_deviations['Manager'] = deviants
df_deviations['Column'] = columns_fix
df_deviations['Deviations'] = deviations
df_lgs_missing = pd.DataFrame(lgs_missing, columns=['Manager', 'Column'])
df_jpm_missing = pd.DataFrame(jpm_missing, columns=['Manager', 'Column'])

accuracy = round(((total_count - deviant_count)/total_count)*100, 2)

print('\nMissing from LGS', lgs_missing)
print('\nMissing from JPM', jpm_missing)
print('\nThe deviants are:\n')
print(df_deviations, '\n')
print('Total Count: ', total_count, 'Deviant Count: ', deviant_count, 'Accuracy: ', accuracy, '%')



# Import JPM_IAP, Accounts; By ID; Include Closed Accounts; Select All; Mode: Portfolio Only
# jpm_iap_filepath = 'U:/CIO/#Investment_Report/Data/input/testing/jpm_iap/'
# jpm_iap_filenames = sorted(os.listdir(jpm_iap_filepath))
# df_jpm_iap = pd.DataFrame()
# for filename in jpm_iap_filenames:
#     jpm_iap_xlsx = pd.ExcelFile(jpm_iap_filepath + filename)
#     df_jpm_iap_temp = pd.read_excel(
#         jpm_iap_xlsx,
#         sheet_name='Sheet1',
#         skiprows=[0, 1],
#         header=0
#     )
#     df_jpm_iap_temp['Date'] = dt.datetime(int(filename[:4]), int(filename[4:6]), int(filename[6:8]))
#     df_jpm_iap = pd.concat([df_jpm_iap, df_jpm_iap_temp], sort=False)
#
# df_jpm_iap = df_jpm_iap.rename(columns={'Account Id': 'Manager'}).reset_index(drop=True)
# df_jpm_iap = df_jpm_iap[['Manager', 'Date', 'Market Value']]
#
# # Merges the market values from JPM IAP with JPM HTS
# df_jpm_main = pd\
#     .merge(
#         left=df_jpm_iap,
#         right=df_jpm,
#         left_on=['Manager', 'Date'],
#         right_on=['Manager', 'Date'],
#         how='right'
#     )\
#     .sort_values(['Manager', 'Date'])\
#     .reset_index(drop=True)