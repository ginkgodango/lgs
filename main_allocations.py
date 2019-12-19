import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

input_directory = 'U:/CIO/#Investment_Report/Data/input/allocations/'
output_directory = 'U:/CIO/#Investment_Report/Data/output/allocations/'

allocations_filename = 'asset_allocations_2019-11-31.csv'
dictionary_filename = 'allocations_dictionary_2019-11-30.csv'

df_allocations = pd.read_csv(
    input_directory + allocations_filename,
    parse_dates=['Date'],
    infer_datetime_format=True,
    float_precision='round_trip'
)

df_dictionary = pd.read_csv(input_directory + dictionary_filename)

df_allocation = df_allocations[df_allocations['Date'] == df_allocations['Date'].max()].reset_index(drop=True)

df_allocation['Deviation'] = df_allocation['Portfolio'] - df_allocation['Benchmark']

df_allocation = df_allocation.round(1)

filter_list = ['Australian Listed Property', 'Legacy Private Equity', 'Option Overlay', 'Forwards', 'Total']

df_allocation = df_allocation[~df_allocation['Asset Class'].isin(filter_list)].reset_index(drop=True)

df_allocation = pd.merge(
    left=df_allocation,
    right=df_dictionary,
    left_on=['Strategy', 'Asset Class'],
    right_on=['Strategy', 'Asset Class'],
    how='inner'
)

df_allocation = df_allocation.rename(columns={'Portfolio': 'Actual AA', 'Benchmark': 'Target AA'})

select_columns = [
    'Strategy',
    'Asset Class',
    'Actual AA',
    'Target AA',
    'Deviation',
    'Min AA',
    'Max AA'
]

df_allocation = df_allocation[select_columns]

jpm_to_lgs_dict = {
    'Australian Equity': 'Australian Equities',
    'International Equity': 'International Equities',
    'Property': 'Australian Property',
    'Global Property': 'International Property',
    'Bonds': 'Bonds',
    'Absolute Return': 'Absolute Return',
    'Cash': 'Managed Cash',
    'Commodities': 'Commodities',
    'Private Equity': 'Private Equity',
    'Opportunistic Alternatives': 'Opportunistic Alternatives',
    'Defensive Alternatives': 'Defensive Alternatives'
}

df_allocation['Asset Class'] = [jpm_to_lgs_dict[df_allocation['Asset Class'][i]] for i in range(0, len(df_allocation))]

strategy_to_dataframe_dict = dict()

treegreen = (75/256, 120/256, 56/256)
lightgreen = (175/256, 215/256, 145/256)

fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(12.80,7.20))

strategy_to_axes_dict = {
    'High Growth': axes[0, 0],
    'Growth': axes[0, 1],
    'Balanced Growth': axes[0, 2],
    'Balanced': axes[1, 0],
    'Conservative': axes[1, 1],
    'Employer Reserve': axes[1, 2]
}


def draw_box(df, column):
    deviations = []
    for i in range(0, len(df)):
        if df[column][i] >= 1:
            value = '\\fcolorbox{red}{white}{' + str(df[column][i]) + '}'
            deviations.append(value)

        elif df[column][i] <= -1:
            value = '\\fcolorbox{red}{white}{' + '{\\color{red}' + str(df[column][i]) + '}}'
            deviations.append(value)

        else:
            deviations.append(df[column][i])

    df[column] = deviations
    return df


for strategy, axes in strategy_to_axes_dict.items():

    df = df_allocation[df_allocation['Strategy'] == strategy].reset_index(drop=True)
    df = df.set_index('Asset Class')

    df_plot = df[['Actual AA', 'Target AA']]
    df_plot.plot.bar(ax=axes, color=[treegreen, lightgreen])
    axes.set_title(strategy)
    axes.set_xlabel('')
    axes.set_ylabel('Asset Allocation %')
    fig.tight_layout()
    plt.show()

    df = df[['Actual AA', 'Deviation', 'Min AA', 'Max AA']]

    df = draw_box(df, 'Deviation')

    strategy_to_dataframe_dict[strategy] = df

df_aa1 = pd.concat(
    [strategy_to_dataframe_dict['High Growth'], strategy_to_dataframe_dict['Growth'], strategy_to_dataframe_dict['Balanced Growth']],
    axis=1,
    keys=['High Growth', 'Growth', 'Balanced Growth'],
    sort=False
)

df_aa1 = df_aa1.reset_index(drop=False)

df_aa1 = df_aa1.rename(columns={'index': 'Asset Class'})

df_aa2 = pd.concat(
    [strategy_to_dataframe_dict['Balanced'], strategy_to_dataframe_dict['Conservative'], strategy_to_dataframe_dict['Employer Reserve']],
    axis=1,
    keys=['Balanced', 'Conservative', 'Employer Reserve'],
    sort=False
)

df_aa2 = df_aa2.reset_index(drop=False)

df_aa2 = df_aa2.rename(columns={'index': 'Asset Class'})

with open(output_directory + 'AA1.tex', 'w') as tf:
    latex_string1 = (
        df_aa1
        .to_latex(index=False, escape=False)
        .replace('\\multicolumn{4}{l}', '\\multicolumn{4}{c}')
        .replace('{lrlllrlllrlll}', '{lRRRR|RRRR|RRRR}')
        .replace('{lrrrrrlrrrlrr}', '{lRRRR|RRRR|RRRR}')
        .replace('{lrlrrrlrrrlrr}', '{lRRRR|RRRR|RRRR}')
        .replace('\\midrule', '\\specialrule{.05em}{.05em}{.05em}')
        .replace('\\bottomrule', '\\specialrule{.08em}{.05em}{.05em}')
    )
    latex_string1 = latex_string1[:199] + '\\cmidrule{2-5}\\cmidrule{6-9}\\cmidrule{10-13}' + latex_string1[199:]
    tf.write(latex_string1)

with open(output_directory + 'AA2.tex', 'w') as tf:
    latex_string2 = (
        df_aa2
        .to_latex(index=False, escape=False)
        .replace('\\multicolumn{4}{l}', '\\multicolumn{4}{c}')
        .replace('{lrlllrlllrlll}', '{lRRRR|RRRR|RRRR}')
        .replace('{lrlrrrlrrrlrr}', '{lRRRR|RRRR|RRRR}')
        .replace('{lrrrrrlrrrlrr}', '{lRRRR|RRRR|RRRR}')
        .replace('\\midrule', '\\specialrule{.05em}{.05em}{.05em}')
        .replace('\\bottomrule', '\\specialrule{.08em}{.05em}{.05em}')
    )
    latex_string2 = latex_string2[:203] + '\\cmidrule{2-5}\\cmidrule{6-9}\\cmidrule{10-13}' + latex_string2[203:]
    tf.write(latex_string2)

fig.savefig(output_directory + 'allocations.png', dpi=300)
