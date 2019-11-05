import os
import pandas as pd
import daymove.extraction
import daymove.calculation
import daymove.format

password = "lgsinvestops@LGS"
filepath = 'U:/CIO/#Daymove/files/2019/11/'
# filepath = 'U:/CIO/#Daymove/files/2019/test/'
filenames = sorted(os.listdir(filepath))
output_directory = 'U:/CIO/#Daymove/tables/'

df_main = pd.DataFrame()

for filename in filenames:
    df = daymove.extraction.load(filepath + filename, password)
    df = daymove.extraction.clean(df)
    df = daymove.extraction.match_manager_to_sector(df)
    df = daymove.extraction.match_manager_to_benchmarks(df)
    df = daymove.extraction.remove_nan_and_header_funds(df)
    df = daymove.calculation.calculate_returns(df)
    df = daymove.calculation.calculate_active_returns(df)

    df_main = pd.concat([df_main, df]).reset_index(drop=True)

df_main = daymove.calculation.calculate_mtd_returns(df_main)

df_today = daymove.format.select_rows_today(df_main)
df_today = daymove.format.select_columns(df_today)
df_today = daymove.format.format_output(df_today)
df_today = daymove.format.filter_misc_funds(df_today)
df_today = daymove.format.rename_funds(df_today)
df_today = daymove.format.rename_benchmarks(df_today)
# df_today = daymove.format.aggregate_southpeak(df_today)
df_today = daymove.format.aggregate_aqr(df_today)
df_today = daymove.format.fillna(df_today)

df_inactive = daymove.format.collect_inactive_funds(df_today)

df_today = daymove.format.filter_inactive_funds(df_today)

df_sectors = daymove.format.collect_sectors(df_today)
df_sectors = daymove.format.reorder_sectors(df_sectors)
df_sectors = daymove.format.rename_to_latex_headers(df_sectors)


sectors = [
    'CASH',
    'INDEXED LINKED BONDS',
    'BONDS',
    'AUSTRALIAN EQUITIES',
    'PROPERTY',
    'GLOBAL PROPERTY',
    'INTERNATIONAL EQUITY',
    'ABSOLUTE RETURN',
    'ACTIVE COMMODITIES SECTOR',
    'LGSS AE OPTION OVERLAY SECTOR',
    'LGSS LEGACY PE SECTOR',
    "LGSS LEGACY DEF'D PESECTOR",
    'LGS PRIVATE EQUITY SECTOR',
    'LGS OPPORTUNISTIC ALTERNATIVES SECTOR',
    'LGS DEFENSIVE ALTERNATIVES SECTOR',
    'TOTAL LGS (LGSMAN)'
]

container = {}
for sector in sectors:
    df_sector = df_today[df_today['Sector'] == sector].reset_index(drop=True)
    df_sector = df_sector[['Fund', 'Benchmark Name', 'Market Value', 'MTD', 'Benchmark', 'Active']]
    df_sector = daymove.format.rename_to_latex_headers(df_sector)
    container[sector] = df_sector

    with open(output_directory + str(sector) + '.tex', 'w') as tf:
        latex_table1 = daymove.format.create_latex_table(df_sector)
        tf.write(latex_table1)

with open(output_directory + 'ALL_SECTORS.tex', 'w') as tf:
    latex_table2 = daymove.format.create_latex_table(df_sectors)
    tf.write(latex_table2)

with open(output_directory + 'INACTIVE_FUNDS.tex', 'w') as tf:
    tf.write(df_inactive.to_latex(index=False))

with open(output_directory + 'REPORT_DATE.tex', 'w') as tf:
    date_today = df_main['Date'].max()
    tf.write(date_today.strftime('%d %B %Y'))

df_output = df_main.sort_values(['Sector', 'FUND', 'Date'])

#df_output.to_csv('D:/automation/final/daymove/daymove_data.csv', index=False)