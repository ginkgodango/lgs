import os
import pandas as pd
import numpy as np
import win32com.client
import datetime
import calendar
import attribution.extraction


directory = 'D:/automation/final/attribution/extraction/'
filenames = sorted(os.listdir(directory))

asset_allocations = ['High Growth', 'Balanced Growth', 'Balanced', 'Conservative', 'Growth', 'Employer Reserve']
sheet_numbers = [17, 17, 18, 18, 19, 19]
start_cells = ['C:8', 'C:35', 'C:8', 'C:35', 'C:8', 'C:35']
end_cells = ['G:22', 'G:49', 'G:22', 'G:49', 'G:22', 'G:50']

excel = win32com.client.Dispatch("Excel.Application")
df_asset_allocations = pd.DataFrame()
for filename in filenames:
    for i in range(0, len(asset_allocations)):
        df = attribution.extraction.load_asset_allocation(
            directory + filename,
            sheet_numbers[i],
            start_cells[i],
            end_cells[i],
            excel
        )
        year_month = filename[:6]
        year = filename[:4]
        month = filename[4:6]
        day = str(calendar.monthrange(int(year), int(month))[1])
        date = datetime.datetime.strptime(year + month + day, "%Y%m%d")
        df.insert(loc=0, column='Date', value=date)
        print(date)
        df_asset_allocations = pd.concat([df_asset_allocations, df], sort=True).reset_index(drop=True)

excel.Quit()
df_asset_allocations.to_csv(directory + 'asset_allocations_201904.csv', index=False)