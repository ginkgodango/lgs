import pandas as pd
import matplotlib.pyplot as plt

df_fred_data = pd.read_csv('U:/CIO/#Data/input/fred/FRED_MD/fred-md/monthly/2020-01.csv', parse_dates=['sasdate'])
df_fred_dict = pd.read_csv('U:/CIO/#Data/input/fred/fred_dictionary_20200229v2.csv', encoding= 'unicode_escape')

match_count = 0
for i in range(0, len(df_fred_dict)):
    if df_fred_dict['fred'][i] in df_fred_data.columns:
        match_count += 1

    else:
        print('missing:', df_fred_dict['fred'][i])
print(match_count)

df_fred_data = df_fred_data.rename(columns={'sasdate': 'date'})
df_fred_data = df_fred_data.set_index('date')
df_fred_data = df_fred_data[1:-1]

for i in range(0, len(df_fred_dict)):

    if df_fred_dict['fred'][i] in df_fred_data.columns:

        fig = df_fred_data[df_fred_dict['fred'][i]].plot(linewidth=1)
        fig.set_title(df_fred_dict['description'][i])
        fig.set_ylabel(df_fred_dict['gsi:description'][i])
        plt.axhline(y=0, linestyle=':', linewidth=1, color='k', )
        filename = df_fred_dict['description'][i].replace('/', '').replace(':', '')
        plt.tight_layout()
        plt.savefig('U:/CIO/#Data/output/fred/charts/' + filename + '.png')
        plt.clf()
