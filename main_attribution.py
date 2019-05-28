"""
Attribution
"""
import pandas as pd
import attribution.extraction


directory = 'D:/automation/final/attribution/2019/04/'
returns_filename = 'returns_NOF_201904.csv'
market_values_filename = 'market_values_NOF_201904.csv'
performance_report_filepath = 'D:/automation/final/investment/2019/04/LGSS Preliminary Performance April 2019_Addkeys.xlsx'

# Loads returns
df_returns = attribution.extraction.load_returns(directory + returns_filename)

# Reshapes returns dataframe from wide to long
df_returns = df_returns.transpose().reset_index(drop=False).rename(columns={'index': 'Manager'})
df_returns = pd.melt(df_returns, id_vars=['Manager'], value_name='1_r')

# Selects returns for this month
df_returns = df_returns[df_returns['Date'] == df_returns['Date'].max()].reset_index(drop=True)

# Loads market values
df_market_values = attribution.extraction.load_market_values(directory + market_values_filename)

# Reshapes market values dataframe from wide to long
df_market_values = df_market_values.transpose().reset_index(drop=False).rename(columns={'index': 'Manager'})
df_market_values = pd.melt(df_market_values, id_vars=['Manager'], value_name='Market Value')

# Selects market values for this month
df_market_values = df_market_values[df_market_values['Date'] == df_market_values['Date'].max()].reset_index(drop=True)

# Joins 1 month returns with market values
df_main = pd.merge(df_returns, df_market_values, how='outer', on=['Manager', 'Date'])

# Loads strategy asset allocations
asset_allocations = ['High Growth', 'Balanced Growth', 'Balanced', 'Conservative', 'Growth', 'Employer Reserve']
sheet_numbers = [17, 17, 18, 18, 19, 19]
start_cells = ['C:8', 'C:35', 'C:8', 'C:35', 'C:8', 'C:35']
end_cells = ['G:22', 'G:49', 'G:22', 'G:50', 'G:22', 'G:50']

df_asset_allocations = pd.DataFrame()
for i in range(0, len(asset_allocations)):
    df = attribution.extraction.load_asset_allocation(
        performance_report_filepath,
        sheet_numbers[i],
        start_cells[i],
        end_cells[i]
    )

    df_asset_allocations = pd.concat([df_asset_allocations, df], sort=True).reset_index(drop=True)

"""
Test for High Growth
April 2019
manager/sector weight * asset/total weight
"""
ae_large = ['BT_AE', 'Ubique_AE', 'DNR_AE', 'Blackrock_AE', 'SSgA_AE', 'DSRI_AE']
ae_small = ['WSCF_AE', 'ECP_AE']
ae = ae_large + ae_small

ie_developed = ['LSV_IE', 'MFS_IE', 'Hermes_IE', 'Longview_IE', 'AQR_IE', 'Impax_IE']
ie_emerging = ['WellingtonEMEquity_IE', 'Delaware_IE']
ie = ie_developed + ie_emerging

ilp = ['ILP']

dp = ['DP']

ae_market_value = 0
for manager in ae:
    for i in range(0, len(df_main)):
        if df_main['Manager'][i] == manager:
            ae_market_value += df_main['Market Value'][i]

sector_weight = dict()
manager_weight = dict()
for manager in ae:
    for i in range(0,len(df_main)):
        if df_main['Manager'][i] == manager:
            sector_weight[manager] = df_main['Market Value'][i]/ae_market_value

            for i in range(0,len(df_asset_allocations)):
                if df_asset_allocations['Asset Class'][i] == 'Australian Equity':
                    print(df_asset_allocations['Strategy'][i], manager, round(sector_weight[manager] * df_asset_allocations['Portfolio'][i]/100,2))


ie_market_value = 0
for manager in ie:
    for i in range(0, len(df_main)):
        if df_main['Manager'][i] == manager:
            ie_market_value += df_main['Market Value'][i]

