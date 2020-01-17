import datetime as dt
import numpy as np
import pandas as pd

lgs_filepath = 'U:/CIO/#Investment_Report/Data/output/testing/checker/lgs_table.csv'
jpm_filepath = 'U:/CIO/#Investment_Report/Data/input/performance_report/LGSS Preliminary Performance 102019_AddKeys.xlsx'
FYTD = 4
report_date = dt.datetime(2019, 10, 31)

# Reads LGS Tables
df_lgs = pd.read_csv(lgs_filepath)

# Reads JPM Performance Report
df_jpm = pd.DataFrame()
sheet_to_columns_dict = {
    'Page 3 NOF': 'A:N',
    'Page 5 NOF': 'B:O',
    'Page 6 NOF': 'B:O',
    'Page 7 NOF': 'B:O',
    # 'Page 8': 'D:O'
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

df_check = pd.DataFrame()
deviants = []
columns = []
deviations = []
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

        total_count += 1

# Fixes the column names
columns_fix = []
for column in columns:
    if column == 'Market Value_y':
        columns_fix.append('Market Value')
    else:
        columns_fix.append(column)

df_check['Manager'] = deviants
df_check['Column'] = columns_fix
df_check['Deviations'] = deviations
accuracy = round(((total_count - deviant_count)/total_count)*100, 2)

print('\nThe deviants are:\n')
print(df_check, '\n')
print('Total Count: ', total_count, 'Deviant Count: ', deviant_count, 'Accuracy: ', accuracy, '%')
