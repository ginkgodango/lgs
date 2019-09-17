import pandas as pd

jpm_directory = 'U:/CIO/#Holdings/Data/input/holdings/jpm/2019/07/'
jpm_filename = 'Priced Positions - All.csv'
jpm_filepath = jpm_directory + jpm_filename

dict_directory = 'U:/CIO/#Holdings/Data/input/dictionary/2019/07/'
dict_filename = 'jpm_dictionary.csv'
dict_filepath = dict_directory + dict_filename

df_jpm = pd.read_csv(jpm_filepath, header=3)

df_dict = pd.read_csv(dict_filepath, header=0)

df_jpm = pd.merge(
    left=df_jpm,
    right=df_dict,
    left_on=['Account Number', 'Account Name'],
    right_on=['Account Number', 'Account Name']
)

df_bonds = df_jpm[df_jpm['Sector Code'].isin(['AF', 'IF'])].reset_index(drop=True)
df_bonds = df_bonds.groupby(['Bloomberg Industry Sector']).sum()
df_bonds = df_bonds[['Total Market Value (Base)']]
df_bonds.to_csv(jpm_directory + 'bonds.csv', index=True)

