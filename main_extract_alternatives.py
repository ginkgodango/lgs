import os
import datetime
import calendar
import pandas as pd
import numpy as np

input_directory = 'U:/CIO/#Investment_Report/Data/input/performance_report_alts/'
output_directory = 'U:/CIO/#Investment_Report/Data/input/alternatives/'
input_filenames = sorted(os.listdir(input_directory))

# Creates an empty unified dataframe for the alternatives data
df_jpm = pd.DataFrame()

# Loops over the filenames in the directory and for each file opens the file as df, renames the columns, extracts the
# alternatives data in Page 8 of each file and concatenates it to df_jpm. df_jpm is then output as alternatives_YYYY-MM-DD.csv.
for filename in input_filenames:
    xlsx = pd.ExcelFile(input_directory + filename)

    df = pd.read_excel(
        xlsx,
        sheet_name='Page 8',
        usecols='E:O',
        skiprows=[0, 1, 2]
    )
    df = df.rename(
        columns={
            'Unnamed: 4': 'Manager',
            'Market Value': 'Market Value',
            '1 Month': '1 Month',
            '3 Months': '3 Month',
            'FYTD': 'FYTD',
            '1 Year': '1 Year',
            '3 Years': '3 Year',
            '5 Years': '5 Year',
            '7 Years': '7 Year'

        }
    )
    df = df.drop(columns=['Unnamed: 6', '2 Years'], axis=1)

    # Removes NaN rows and last 2 rows which are footnotes
    df = df[df['Manager'].notnull()][:-2].reset_index(drop=True)

    df = df.replace('-', np.nan)

    year_month = filename[:6]
    year = filename[:4]
    month = filename[4:6]
    day = str(calendar.monthrange(int(year), int(month))[1])
    date = datetime.datetime.strptime(year + month + day, "%Y%m%d").date()
    df.insert(loc=0, column='Date', value=date)
    print(date)
    df_jpm = pd.concat([df_jpm, df], sort=False).reset_index(drop=True)

# Outputs df_jpm into a csv
df_jpm.to_csv(output_directory + 'alternatives_' + str(date) + '.csv', index=False)
