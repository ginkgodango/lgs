import os
import datetime
import calendar
import pandas as pd
import numpy as np
input_directory = 'D:/data/LGS/JPM/monthly/'
output_directory = 'D:/data/output/LGS/alternatives/'
input_filenames = sorted(os.listdir(input_directory))

df_jpm = pd.DataFrame()
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


df_jpm.to_csv(output_directory + 'alternatives_' + str(date) + '.csv', index=False)
