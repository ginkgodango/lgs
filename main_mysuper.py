from datetime import datetime
import pandas as pd


input_directory = 'U:/CIO/#Investment_Report/Data/input/mysuper/'
output_directory = 'U:/CIO/#Investment_Report/Data/output/mysuper/'
filename = 'MySuper FUM 20190909.xlsx'
report_date = datetime(2019, 8, 31)

df_mysuper = pd.read_excel(input_directory + filename, sheet_name='Sheet1')

df_mysuper = df_mysuper[['MySuper Option', 'Amount', 'Number of Mbrs']]
df_mysuper.columns = ['MySuper', 'Market Value', 'Member Count']

option_to_name_dict = {
        'MySuper High Growth': 'High Growth',
        'MySuper Balanced Grw': 'Balanced Growth',
        'MySuper Balanced': 'Balanced',
        'MySuper Conservative': 'Conservative',
        'Grand Total': 'TOTAL'
        }

df_mysuper['MySuper'] = [
        option_to_name_dict[df_mysuper['MySuper'][i]]
        for i in range(0,len(df_mysuper))
        ]

df_mysuper['Market Value'] = [
        round(df_mysuper['Market Value'][i]/1000000,2)
        for i in range(0,len(df_mysuper))
        ]

df_mysuper.set_index('MySuper', inplace=True)

df_mysuper = df_mysuper.reindex([
        'High Growth',
        'Balanced Growth',
        'Balanced',
        'Conservative',
        'TOTAL'
        ])

df_mysuper = df_mysuper.reset_index(drop=False)

df_mysuper.to_csv(output_directory + 'output.csv', index=False)

with open(output_directory + 'MySuper.tex', 'w') as tf:
    tf.write(df_mysuper.to_latex(index=False))
