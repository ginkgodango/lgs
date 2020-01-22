import os
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import statsmodels.api as sm

# START USER INPUT DATA
jpm_main_filepath = 'U:/CIO/#Data/input/jpm/performance/2019/12/Historical Time Series - Monthly - Main Returns and Benchmarks.xlsx'
jpm_alts_filepath = 'U:/CIO/#Data/input/jpm/performance/2019/12/Historical Time Series - Monthly - Alternatives Returns and Benchmarks.xlsx'
jpm_iap_filepath = 'U:/CIO/#Investment_Report/Data/input/testing/jpm_iap/'
jpm_mv_filepath = 'U:/CIO/#Data/input/jpm/performance/2019/12/Historical Time Series - Monthly - Main Market Values.xlsx'
jpm_mv_alts_filepath = 'U:/CIO/#Data/input/jpm/performance/2019/12/Historical Time Series - Monthly - Alternatives Market Values.xlsx'
lgs_dictionary_filepath = 'U:/CIO/#Investment_Report/Data/input/testing/20191031 New Dictionary.xlsx'
FYTD = 6
report_date = dt.datetime(2019, 12, 31)
# END USER INPUT DATA

# Imports the JPM time-series.
use_managerid = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 14, 15]
use_accountid = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15]
footnote_rows = 28

df_jpm_main = pd.read_excel(
        pd.ExcelFile(jpm_main_filepath),
        sheet_name='Sheet1',
        skiprows=use_accountid,
        skipfooter=footnote_rows,
        header=1
        )

# Reshapes the time-series into a panel.
df_jpm_main = df_jpm_main.rename(columns={'Unnamed: 0': 'Date'})
df_jpm_main = df_jpm_main.set_index('Date')
df_jpm_main = df_jpm_main.transpose()
df_jpm_main = df_jpm_main.reset_index(drop=False)
df_jpm_main = df_jpm_main.rename(columns={'index': 'Manager'})
df_jpm_main = pd.melt(df_jpm_main, id_vars=['Manager'], value_name='JPM_Return')
df_jpm_main = df_jpm_main.sort_values(['Manager', 'Date'])
df_jpm_main = df_jpm_main.reset_index(drop=True)
df_jpm_main = df_jpm_main.replace('-', np.NaN)

df_jpm_alts = pd.read_excel(
        pd.ExcelFile(jpm_alts_filepath),
        sheet_name='Sheet1',
        skiprows=use_accountid,
        skipfooter=footnote_rows,
        header=1
        )

df_jpm_alts = df_jpm_alts.rename(columns={'Unnamed: 0': 'Date'})
df_jpm_alts = df_jpm_alts.set_index('Date')
df_jpm_alts = df_jpm_alts.transpose()
df_jpm_alts = df_jpm_alts.reset_index(drop=False)
df_jpm_alts = df_jpm_alts.rename(columns={'index': 'Manager'})
df_jpm_alts = pd.melt(df_jpm_alts, id_vars=['Manager'], value_name='JPM_Return')
df_jpm_alts = df_jpm_alts.sort_values(['Manager', 'Date'])
df_jpm_alts = df_jpm_alts.reset_index(drop=True)
df_jpm_alts = df_jpm_alts.replace('-', np.NaN)

# df_jpm = pd.concat([df_jpm_main, df_jpm_alts], axis=0).reset_index(drop=True)

df_jpm = df_jpm_main
# df_jpm = df_jpm_alts

# Converts the returns to percentage.
df_jpm['JPM_Return'] = df_jpm['JPM_Return']/100

# df_jpm = df_jpm[~df_jpm.Manager.str.endswith('.2')].reset_index(drop=True)

# Splits df_jpm into a returns and benchmarks
df_jpm_returns = df_jpm[~df_jpm.Manager.str.endswith('.1')].reset_index(drop=True)
df_jpm_benchmarks = df_jpm[df_jpm.Manager.str.endswith('.1')].reset_index(drop=True)
df_jpm_benchmarks['Manager'] = [df_jpm_benchmarks['Manager'][i][:-2] for i in range(0, len(df_jpm_benchmarks))]
df_jpm_benchmarks = df_jpm_benchmarks.rename(columns={'JPM_Return': 'JPM_Benchmark'})

# Creates Rf from Cash Aggregate Benchmark
df_jpm_rf = df_jpm_benchmarks[df_jpm_benchmarks['Manager'].isin(['CLFACASH', 'Cash Aggregate'])].reset_index(drop=True)
df_jpm_rf = df_jpm_rf.rename(columns={'JPM_Benchmark': 'JPM_Rf'})

rf_values = []
new_cash_benchmark_date = dt.datetime(2019, 11, 30)
for i in range(0, len(df_jpm_rf)):
    if df_jpm_rf['Date'][i] >= new_cash_benchmark_date:
        rf_values.append(df_jpm_rf['JPM_Rf'][i] - (((1+0.002)**(1/12))-1))
    else:
        rf_values.append(df_jpm_rf['JPM_Rf'][i])
df_jpm_rf['JPM_Rf'] = rf_values
df_jpm_rf = df_jpm_rf.drop(columns=['Manager'], axis=1)

# Create ASX300 for regression
df_jpm_asx300 = df_jpm_benchmarks[df_jpm_benchmarks['Manager'].isin(['CEIAETOT', 'Australian Equities Aggregate'])].reset_index(drop=True)
df_jpm_asx300 = df_jpm_asx300.rename(columns={'JPM_Benchmark': 'JPM_ASX300'})
df_jpm_asx300 = df_jpm_asx300.drop(columns=['Manager'], axis=1)

# Merges returns and benchmarks
df_jpm = pd.merge(
        left=df_jpm_returns,
        right=df_jpm_benchmarks,
        left_on=['Manager', 'Date'],
        right_on=['Manager', 'Date'],
        how='inner'
)


# Merges returns, benchmarks and Rf
df_jpm = pd.merge(
        left=df_jpm,
        right=df_jpm_rf,
        left_on=['Date'],
        right_on=['Date'],
        how='inner'
)


df_jpm = pd.merge(
        left=df_jpm,
        right=df_jpm_asx300,
        left_on=['Date'],
        right_on=['Date'],
        how='inner'
)

# Deletes the redundant dataframes.
# del df_jpm_returns
# del df_jpm_benchmarks


# Reads LGS's dictionary
df_lgs = pd.read_excel(
    pd.ExcelFile(lgs_dictionary_filepath),
    sheet_name='Sheet1',
    header=0
)
df_lgs = df_lgs.rename(columns={'JPM Account Id': 'Manager'})

df_jpm = pd.merge(
        left=df_jpm,
        right=df_lgs,
        left_on=['Manager'],
        right_on=['Manager'],
        how='inner'
)

# Keep only open accounts
df_jpm = df_jpm[df_jpm['LGS Open'] == 1].reset_index(drop=True)
df_jpm = df_jpm.drop(columns=['LGS Open'], axis=1)


# df_jpm = df_jpm.sort_values(['LGS Asset Class Order', 'LGS Manager Order', 'Date'], ascending=[True, True, True]).reset_index(drop=True)

# Sets the dictionary for the holding period returns.
horizon_to_period_dict = {
        '1_': 1,
        '3_': 3,
        'FYTD_': FYTD,
        '12_': 12,
        '24_': 24,
        '36_': 36,
        '60_': 60,
        '84_': 84
}


# Calculates the holding period returns and annualises for periods greater than 12 months.
for horizon, period in horizon_to_period_dict.items():

    for column in ['Return', 'Benchmark', 'Rf']:

        column_name = horizon + column
        return_type = 'JPM_' + column

        if period <= 12:
            df_jpm[column_name] = (
                df_jpm
                .groupby(['Manager'])[return_type]
                .rolling(period)
                .apply(lambda r: np.prod(1+r)-1, raw=False)
                .reset_index(drop=False)[return_type]
            )

        elif period > 12:
            df_jpm[column_name] = (
                df_jpm
                .groupby(['Manager'])[return_type]
                .rolling(period)
                .apply(lambda r: (np.prod(1+r)**(12/period))-1, raw=False)
                .reset_index(drop=False)[return_type]
            )

    df_jpm[horizon + 'Excess'] = df_jpm[horizon + 'Return'] - df_jpm[horizon + 'Benchmark']

# Calculates volatility, tracking error, sharpe ratio, information ratio
df_jpm['36_Volatility'] = (
    df_jpm
    .groupby(['Manager'])['1_Return']
    .rolling(36)
    .apply(lambda r: np.std(r, ddof=1)*np.sqrt(12), raw=False)
    .reset_index(drop=False)['1_Return']
)

df_jpm['36_Tracking_Error'] = (
    df_jpm
    .groupby(['Manager'])['1_Excess']
    .rolling(36)
    .apply(lambda r: np.std(r, ddof=1)*np.sqrt(12), raw=False)
    .reset_index(drop=False)['1_Excess']
)

df_jpm['36_Sharpe_Ratio'] = (df_jpm['36_Return'] - df_jpm['36_Rf']) / df_jpm['36_Volatility']

df_jpm['36_Information_Ratio'] = df_jpm['36_Excess'] / df_jpm['36_Tracking_Error']

# Calculates rolling betas
def rolling_ols(indices, result, ycol, xcols):
    roll_df = df.loc[indices] # get relevant data frame subset
    result[indices[-1]] = (sm.OLS(roll_df[ycol], sm.add_constant(roll_df[xcols]), hasconst=True).fit().params)[-1]
    return 0

# Creates container and writes results of regression beta to result: idx: beta
kwargs = {
    "xcols": ['JPM_ASX300'],
    "ycol": '1_Return',
    "result": {}
}

# iterate id's sub data frames and call rolling_ols for rolling windows
df = df_jpm.copy()
df["identifier"] = df.index
for idx, sub_df in df.groupby("Manager"):
    sub_df["identifier"].rolling(36).apply(rolling_ols, kwargs=kwargs, raw=True)

# write results back to original df
df_jpm['36_Beta'] = pd.Series(kwargs["result"])
# End calculation of rolling beta

# Imports the JPM time-series of market values.
df_jpm_mv = pd.read_excel(
        pd.ExcelFile(jpm_mv_filepath),
        sheet_name='Sheet1',
        skiprows=use_accountid,
        skipfooter=footnote_rows,
        header=1
        )

# Reshapes the time-series into a panel.
df_jpm_mv = df_jpm_mv.rename(columns={'Unnamed: 0': 'Date'})
df_jpm_mv = df_jpm_mv.set_index('Date')
df_jpm_mv = df_jpm_mv.transpose()
df_jpm_mv = df_jpm_mv.reset_index(drop=False)
df_jpm_mv = df_jpm_mv.rename(columns={'index': 'Manager'})
df_jpm_mv = pd.melt(df_jpm_mv, id_vars=['Manager'], value_name='Market Value')
df_jpm_mv = df_jpm_mv.sort_values(['Manager', 'Date'])
df_jpm_mv = df_jpm_mv.reset_index(drop=True)
df_jpm_mv = df_jpm_mv.replace('-', np.NaN)

# Imports the JPM time-series of market values.
df_jpm_mv_alts = pd.read_excel(
        pd.ExcelFile(jpm_mv_alts_filepath),
        sheet_name='Sheet1',
        skiprows=use_accountid,
        skipfooter=footnote_rows,
        header=1
        )

# Reshapes the time-series into a panel.
df_jpm_mv_alts = df_jpm_mv_alts.rename(columns={'Unnamed: 0': 'Date'})
df_jpm_mv_alts = df_jpm_mv_alts.set_index('Date')
df_jpm_mv_alts = df_jpm_mv_alts.transpose()
df_jpm_mv_alts = df_jpm_mv_alts.reset_index(drop=False)
df_jpm_mv_alts = df_jpm_mv_alts.rename(columns={'index': 'Manager'})
df_jpm_mv_alts = pd.melt(df_jpm_mv_alts, id_vars=['Manager'], value_name='Market Value')
df_jpm_mv_alts = df_jpm_mv_alts.sort_values(['Manager', 'Date'])
df_jpm_mv_alts = df_jpm_mv_alts.reset_index(drop=True)
df_jpm_mv_alts = df_jpm_mv_alts.replace('-', np.NaN)

df_jpm_mv = pd.concat([df_jpm_mv, df_jpm_mv_alts]).reset_index(drop=True)

df_jpm_main = pd\
    .merge(
        left=df_jpm_mv,
        right=df_jpm,
        left_on=['Manager', 'Date'],
        right_on=['Manager', 'Date'],
        how='right'
    )\
    .sort_values(['Manager', 'Date'])\
    .reset_index(drop=True)

# ACTIVE CONTRIBUTION
remove_double_count = []
df_jpm_main['12_Average_Market_Value'] = (
    df_jpm_main[~df_jpm_main['Manager'].isin([remove_double_count])]
    .groupby(['Manager'])['Market Value']
    .rolling(12)
    .mean()
    .reset_index(drop=True)
)

df_jpm_main['Total Market_Value'] = (
    df_jpm_main[~df_jpm_main['LGS Sector Aggregate'].isin([1])]
    .groupby(['Date'])['Market Value']
    .transform('sum')
    .reset_index(drop=True)
)

df_jpm_main['12_Active_Contribution'] = (
        (df_jpm_main['12_Average_Market_Value'] / df_jpm_main['Total Market_Value']) * (df_jpm_main['12_Excess'])
)

# SUSTAINABILITY
df_jpm_main['Sustainability Value'] = df_jpm_main['LGS Sustainability Weight'] * df_jpm_main['Market Value']


# SWITCHES MANAGER WITH LGS NAME, ALSO RENAMES RISK METRICS
df_jpm_main = df_jpm_main.drop(['Manager'], axis=1)
df_jpm_main = df_jpm_main.rename(
        columns={
                'LGS Name': 'Manager',
                '36_Tracking_Error': 'Tracking Error',
                '36_Volatility': 'Volatility',
                '36_Information_Ratio': 'Information Ratio',
                '36_Sharpe_Ratio': 'Sharpe Ratio',
                '36_Beta': 'Beta'
        }
)

df_jpm_main = df_jpm_main.sort_values(['LGS Asset Class Order', 'LGS Manager Order']).reset_index(drop=True)

# Deletes duplicates
df_jpm_main_duplicated = df_jpm_main[df_jpm_main.duplicated(['Manager', 'Date'])]
df_jpm_main = df_jpm_main[~df_jpm_main.duplicated(['Manager', 'Date'])].reset_index(drop=True)


# CREATES LATEX TABLES AND CHARTS
# Selects rows as at report date
df_jpm_table = df_jpm_main[df_jpm_main['Date'] == report_date].reset_index(drop=True)

# Sets list of columns for each table
columns_lead = ['Manager', 'Market Value']
columns_indicators = ['LGS Asset Class', 'LGS Sector Aggregate', 'JPM ReportName']
columns_performance = []
for horizon, period in horizon_to_period_dict.items():
    for column in ['Return', 'Excess']:
        columns_performance.append(horizon + column)
columns_risk = ['Tracking Error', 'Volatility', 'Information Ratio', 'Sharpe Ratio', 'Beta']
columns_active_contribution = ['12_Active_Contribution']
columns_millions = ['Market Value']
columns_decimal = columns_performance + columns_risk[:2] + columns_active_contribution
columns_round = columns_millions + columns_decimal + columns_risk + columns_active_contribution

# Selects columns for Latex Tables
df_jpm_table = df_jpm_table[
        columns_lead +
        columns_performance +
        columns_risk +
        columns_active_contribution +
        columns_indicators
]

# Converts market value into millions and decimal into percentage
df_jpm_table[columns_millions] = df_jpm_table[columns_millions] / 1000000
df_jpm_table[columns_decimal] = df_jpm_table[columns_decimal] * 100
df_jpm_table[columns_round] = df_jpm_table[columns_round].round(2)

# Creates column hierarchy for performance table
columns_performance_lead_multilevel1 = pd.MultiIndex.from_product([[''], ['Manager', 'LGS Asset Class']], names=['horizon', 'type'])
columns_performance_lead_multilevel2 = pd.MultiIndex.from_product([['Market Value'], ['$Mills']], names=['horizon', 'type'])
columns_performance_performance_multilevel = pd.MultiIndex.from_product(
    [['1 Month', '3 Month', 'FYTD', '1 Year', '2 Year', '3 Year', '5 Year', '7 Year'], ['LGS', 'Active']],
    names=['horizon', 'type']
)

# Creates performance tables
df_jpm_table_performance_lead1 = df_jpm_table[['Manager', 'LGS Asset Class']]
df_jpm_table_performance_lead2 = df_jpm_table[['Market Value']]
df_jpm_table_performance_lead1.columns = columns_performance_lead_multilevel1
df_jpm_table_performance_lead2.columns = columns_performance_lead_multilevel2
df_jpm_table_performance_performance = df_jpm_table[columns_performance]
df_jpm_table_performance_performance.columns = columns_performance_performance_multilevel
df_jpm_table_performance = pd.concat(
    [
        df_jpm_table_performance_lead1,
        df_jpm_table_performance_lead2,
        df_jpm_table_performance_performance
        ],
    axis=1
)

del df_jpm_table_performance_lead1
del df_jpm_table_performance_lead2
del df_jpm_table_performance_performance

asset_class_to_performance_dict = dict(list(df_jpm_table_performance.groupby([('', 'LGS Asset Class')])))
for asset_class, df_temp in asset_class_to_performance_dict.items():
    df_temp = df_temp.drop(('', 'LGS Asset Class'), axis=1)
    df_temp.to_csv('U:/CIO/#Data/output/investment/returns/' + str(asset_class) + '.csv', index=False)

    with open('U:/CIO/#Data/output/investment/returns/' + str(asset_class) + '.tex', 'w') as tf:
        latex_string_temp = (
                df_temp
                .to_latex(index=False, na_rep='', multicolumn_format='c', column_format='lRRRRRRRRRRRRRRRRR')
                .replace('-0.00', '0.00')
        )
        tf.write(latex_string_temp)


# Creates risk table
df_jpm_table_risk = df_jpm_table[columns_lead + columns_risk + ['LGS Asset Class']]

asset_class_to_risk_dict = dict(list(df_jpm_table_risk.groupby(['LGS Asset Class'])))
for asset_class, df_temp in asset_class_to_risk_dict.items():
    df_temp = df_temp.drop('LGS Asset Class', axis=1)
    df_temp.to_csv('U:/CIO/#Data/output/investment/risk/' + str(asset_class) + '.csv', index=False)

    with open('U:/CIO/#Data/output/investment/risk/' + str(asset_class) + '.tex', 'w') as tf:
        latex_string_temp = (
                df_temp
                .to_latex(index=False, na_rep='', multicolumn_format='c')
                .replace('-0.00', '0.00')
        )
        tf.write(latex_string_temp)


# Creates active contribution table
df_jpm_table_active_contribution = df_jpm_table[columns_lead[:1] + columns_active_contribution]
df_jpm_table_active_contribution = df_jpm_table_active_contribution.sort_values(['12_Active_Contribution'], ascending=False).reset_index(drop=True)
df_jpm_table_active_contribution_missing = df_jpm_table_active_contribution[df_jpm_table_active_contribution['12_Active_Contribution'].isin([np.nan])]
df_jpm_table_active_contribution = df_jpm_table_active_contribution[~df_jpm_table_active_contribution['12_Active_Contribution'].isin([np.nan])]

df_jpm_table_active_contribution_top = df_jpm_table_active_contribution[:10].reset_index(drop=True)
df_jpm_table_active_contribution_top = df_jpm_table_active_contribution_top.rename(columns={'12_Active_Contribution': 'Contribution'})
df_jpm_table_active_contribution_bottom = df_jpm_table_active_contribution[-10:].reset_index(drop=True)
df_jpm_table_active_contribution_bottom = df_jpm_table_active_contribution_bottom.rename(columns={'12_Active_Contribution': 'Detraction'})

df_jpm_table_active_contribution_combined = pd.concat([df_jpm_table_active_contribution_top, df_jpm_table_active_contribution_bottom], axis=1, sort=False)
df_jpm_table_active_contribution_combined.to_csv('U:/CIO/#Data/output/investment/contributors/contributors.csv', index=False)
with open('U:/CIO/#Data/output/investment/contributors/contributors.tex', 'w') as tf:
    tf.write(df_jpm_table_active_contribution_combined.to_latex(index=False, na_rep=''))


# Creates charts
df_jpm_chart_12_excess = df_jpm_main[['Manager', 'Date', '12_Excess', 'LGS Sector Aggregate', 'LGS Asset Class']]
df_jpm_chart_12_excess = df_jpm_chart_12_excess[df_jpm_chart_12_excess['LGS Sector Aggregate'] == 0].reset_index(drop=True)
df_jpm_chart_12_excess = df_jpm_chart_12_excess.drop(columns=['LGS Sector Aggregate'], axis=1)
df_jpm_chart_12_excess['12_Excess'] = df_jpm_chart_12_excess['12_Excess']*100
asset_class_to_chart_12_dict = dict(list(df_jpm_chart_12_excess.groupby(['LGS Asset Class'])))

df_jpm_chart_60_excess = df_jpm_main[['Manager', 'Date', '60_Excess', 'LGS Sector Aggregate', 'LGS Asset Class']]
df_jpm_chart_60_excess = df_jpm_chart_60_excess[df_jpm_chart_60_excess['LGS Sector Aggregate'] == 1].reset_index(drop=True)
df_jpm_chart_60_excess = df_jpm_chart_60_excess.drop(columns=['LGS Sector Aggregate'], axis=1)
df_jpm_chart_60_excess['60_Excess'] = df_jpm_chart_60_excess['60_Excess']*100
asset_class_to_chart_60_dict = dict(list(df_jpm_chart_60_excess.groupby(['LGS Asset Class'])))

df_jpm_chart_mv = df_jpm_main[['Manager', 'Date', 'Market Value', 'LGS Sector Aggregate', 'LGS Asset Class']]
df_jpm_chart_mv = df_jpm_chart_mv[df_jpm_chart_mv['Date'] == report_date].reset_index(drop=True)
df_jpm_chart_mv = df_jpm_chart_mv[df_jpm_chart_mv['LGS Sector Aggregate'] == 0].reset_index(drop=True)
df_jpm_chart_mv = df_jpm_chart_mv.drop(columns=['LGS Sector Aggregate'], axis=1)
asset_class_to_chart_mv_dict = dict(list(df_jpm_chart_mv.groupby(['LGS Asset Class'])))

asset_classes1 = list(asset_class_to_chart_12_dict.keys())
asset_classes2 = list(asset_class_to_chart_60_dict.keys())
# assert asset_classes1 == asset_classes2
asset_classes = asset_classes1
for asset_class in asset_classes:

    fig, axes = plt.subplots(
            nrows=1,
            ncols=3,
            sharex=False,
            sharey=False,
            gridspec_kw={'width_ratios': [2, 2, 1]},
            figsize=(16.8, 3.6)
            )

    df_chart_12_temp = asset_class_to_chart_12_dict[asset_class]
    df_chart_12_temp = df_chart_12_temp.pivot(index='Date', columns='Manager', values='12_Excess')[-60:]
    df_chart_12_temp.plot(ax=axes[0], linewidth=1)
    axes[0].set_title('Manager 1-yr Rolling Active Returns')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('Active Return %')
    axes[0].margins(x=0)
    axes[0].axhline(y=0, linestyle=':', linewidth=1, color='k',)
    axes[0].legend(loc='lower left', title='')

    df_chart_60_temp = asset_class_to_chart_60_dict[asset_class]
    df_chart_60_temp = df_chart_60_temp.pivot(index='Date', columns='Manager', values='60_Excess')[-60:]
    df_chart_60_temp.plot(ax=axes[1], linewidth=1)
    axes[1].set_title('Sector Annualized 5-yr Rolling Active Returns')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('Active Return %')
    axes[1].margins(x=0)
    axes[1].axhline(y=0, linestyle=':', linewidth=1, color='k',)
    axes[1].legend(loc='lower left', title='')

    df_chart_mv_temp = asset_class_to_chart_mv_dict[asset_class]
    df_chart_mv_temp = df_chart_mv_temp.set_index('Manager')
    axes[2] = df_chart_mv_temp.plot(kind='pie', ax=axes[2], y='Market Value', autopct='%1.0f%%', legend=None, title='Holdings')
    my_circle = plt.Circle((0, 0), 0.75, color='white')
    axes[2].add_artist(my_circle)
    axes[2].set_ylabel(None)

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    fig.savefig('U:/CIO/#Data/output/investment/charts/' + str(asset_class) + '.png', dpi=300)

# Creates sustainability table
df_jpm_main_esg = df_jpm_main[
    [
        'Manager',
        'Market Value',
        'LGS Sustainability Weight',
        'Sustainability Value',
        '1_Return',
        '1_Excess',
        '3_Return',
        '3_Excess',
        '12_Return',
        '12_Excess',
        '36_Return',
        '36_Excess'
    ]
]

df_jpm_main_esg = df_jpm_main_esg[(df_jpm_main_esg['LGS Sustainability Weight'] != 0) & (df_jpm_main['Date'] == report_date)].reset_index(drop=True)

df_jpm_table.to_csv('U:/CIO/#Data/output/investment/checker/lgs_table.csv', index=False)
