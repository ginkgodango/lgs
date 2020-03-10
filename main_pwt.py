import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_legend = pd.read_excel(
    pd.ExcelFile('U:/CIO/#Data/input/pwt/pwt91.xlsx'),
    sheet_name='Legend'
)

legend_headers = []
for i in range(0, len(df_legend)):
    if pd.isna(df_legend['Variable definition'][i]):
        legend_header = df_legend['Variable name'][i]
        legend_headers.append(legend_header)
    else:
        legend_headers.append(legend_header)

df_legend['Header'] = legend_headers

df_legend_headers = df_legend[df_legend['Variable definition'].isin([np.nan])].reset_index(drop=True)
df_legend_identifiers = df_legend[df_legend['Variable definition'].isin(['Identifier variables', 'Data information variables'])].reset_index(drop=True)
df_legend = df_legend[~df_legend['Variable definition'].isin([np.nan])].reset_index(drop=True)
df_legend = df_legend[~df_legend['Header'].isin([np.nan, 'Identifier variables', 'Data information variables'])].reset_index(drop=True)

df_data = pd.read_excel(
    pd.ExcelFile('U:/CIO/#Data/input/pwt/pwt91.xlsx'),
    sheet_name='Data'
)

df_lgs = pd.read_excel(
    pd.ExcelFile('U:/CIO/#Data/input/pwt/lgs_pwt_dictionary.xlsx'),
    sheet_name='Sheet1'
)

df_data = pd.merge(
    left=df_lgs,
    right=df_data,
    left_on=['countrycode', 'country', 'currency_unit'],
    right_on=['countrycode', 'country', 'currency_unit'],
    how='inner'
)

df_data = df_data[df_data['g20'].isin([1])].reset_index(drop=True)

for i in range(0, len(df_legend)):
    variable_name = df_legend['Variable name'][i]
    variable_definition = df_legend['Variable definition'][i]
    variable_header = df_legend['Header'][i]

    df_data[variable_name + '_lag1'] = df_data.groupby('country')[variable_name].shift(1)
    df_data['g_' + variable_name] = (df_data[variable_name] - df_data[variable_name + '_lag1'])/df_data[variable_name + '_lag1']
    df_data = df_data.drop(columns=[variable_name + '_lag1'], axis=0)

for prefix in ['', 'g_']:

    for i in range(0, len(df_legend)):
        variable_name = prefix + df_legend['Variable name'][i]
        variable_definition = df_legend['Variable definition'][i]
        variable_header = df_legend['Header'][i]

        df_temp_chart = df_data[['country', 'year', variable_name]]
        df_temp_chart = df_temp_chart.pivot(index='year', columns='country', values=variable_name)
        fig = df_temp_chart.plot(linewidth=1, figsize=(12.80, 7.2))
        fig.set_ylabel(variable_name)

        if variable_name[:2] == 'g_':
            folder = 'growth/'
            filename = ('g_' + variable_definition).replace('/', '').replace(':', '')
            fig.set_title('Growth in ' + variable_definition)
            plt.axhline(y=0, linestyle=':', linewidth=1, color='k', )
        else:
            folder = 'level/'
            filename = variable_definition.replace('/', '').replace(':', '')
            fig.set_title(variable_definition)

        plt.tight_layout()
        plt.savefig('U:/CIO/#Data/output/pwt/charts/' + folder + str(i) + '. ' + filename + '.png')
        plt.clf()
        plt.close()

