import pandas as pd
import numpy as np
import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
FYTD = 5
report_date = dt.datetime(2019, 11, 30)
darkgreen = (75/256, 120/256, 56/256)
middlegreen = (141/256, 177/256, 66/256)
lightgreen = (175/256, 215/256, 145/256)

# UNIT PRICES
# Imports unit price panel
df_up = pd.read_csv('U:/CIO/#Data/input/lgs/unitprices/20191031 Unit Prices.csv', parse_dates=['Date'])
df_up_unique = df_up[df_up['Date'] == report_date]
# df_up_unique.to_csv('U:/CIO/#Research/product_unique.csv', index=False)

df_up = df_up.rename(
    columns={
        'InvestmentOption.InvestmentComponent.Name': 'Component',
        'InvestmentOption.InvestmentOptionType.Name': 'OptionType'
    }
)

# Renames columns
df_up = (
    df_up[
        (df_up['Component'].isin(['Accumulation Scheme', 'Defined Benefit Scheme'])) |
        ((df_up['Component'].isin(['Contributor Financed Benefit'])) & (df_up['OptionType'].isin(['Growth'])))
    ]
    .drop(columns=['Component'], axis=1)
    .sort_values(['OptionType', 'Date'])
    .reset_index(drop=True)
)



# Removes SAS
df_up_SAS = df_up[df_up['OptionType'] == 'Sustainable Australian Shares']
df_up = df_up[~df_up['OptionType'].isin(['Sustainable Australian Shares'])].reset_index(drop=True)

# Removes rows after reporting date
df_up_future = df_up[df_up['Date'] > report_date]
df_up = df_up[df_up['Date'] <= report_date].reset_index(drop=True)

# Checks for unit prices equal to zero and filters it
df_up_zeros = df_up[df_up['Unit Price'] == 0]
df_up = df_up[df_up['Unit Price'] != 0].reset_index(drop=True)

# Down-samples the data using hypothetical month end
df_up_monthly = df_up.set_index('Date').groupby('OptionType').resample('M').pad().drop(columns='OptionType', axis=1).reset_index(drop=False)

# Creates month and day indicators
df_up_monthly['Year'] = [df_up_monthly['Date'][i].year for i in range(0, len(df_up_monthly))]
df_up_monthly['Month'] = [df_up_monthly['Date'][i].month for i in range(0, len(df_up_monthly))]
df_up_monthly['Day'] = [df_up_monthly['Date'][i].day for i in range(0, len(df_up_monthly))]

# Down-samples the data using actual dates
# df_up_monthly = df_up_monthly.groupby(['OptionType', 'Year', 'Month']).tail(1).reset_index(drop=True)
# Checks for months which ended before 28 days
# df_up_monthly_not_full_month = df_up_monthly[df_up_monthly['Day'] < 28]

# Calculates 1 month return
df_up_monthly['Unit Price Lag 1'] = df_up_monthly.groupby('OptionType')['Unit Price'].shift(1)
df_up_monthly['Return'] = (df_up_monthly['Unit Price'] - df_up_monthly['Unit Price Lag 1']) / df_up_monthly['Unit Price Lag 1']


# STRATEGY BENCHMARKS
df_cpi = pd.read_csv(
    'U:/CIO/#Data/input/rba/inflation/20190930 g1-data.csv',
    header=10,
    usecols=['Series ID', 'GCPIAGQP'],
    parse_dates=['Series ID'],
    index_col=['Series ID']
)

df_cpi = df_cpi.rename(columns={'GCPIAGQP': 'Inflation'})
df_cpi = df_cpi[df_cpi.index >= dt.datetime(1999, 1, 1)]
df_cpi['Inflation'] = df_cpi['Inflation']/100
df_cpi['Inflation'] = ((1 + df_cpi['Inflation']) ** (1/3)) - 1
df_cpi = df_cpi.resample('M').pad()
df_cpi['Inflation'] = df_cpi['Inflation'].shift(-2).ffill()
df_cpi = df_cpi.reset_index(drop=False)
df_cpi = df_cpi.rename(columns={'Series ID': 'Date'})

df_cpi.to_csv('U:/CIO/#Research/inflation_output.csv', index=False)

df_up_monthly = pd.merge(
    left=df_up_monthly,
    right=df_cpi,
    left_on=['Date'],
    right_on=['Date']
)

employer_reserve_benchmark_monthly = ((1 + 0.0575)**(1/12) - 1)
core_benchmark = []
for i in range(0, len(df_up_monthly)):
    if df_up_monthly['OptionType'][i] == 'Defined Benefit Strategy':
        core_benchmark.append(employer_reserve_benchmark_monthly)
    else:
        core_benchmark.append(df_up_monthly['Inflation'][i])
df_up_monthly['Core'] = core_benchmark

df_up_monthly = df_up_monthly.sort_values(['OptionType', 'Date']).reset_index(drop=True)

# Sets the dictionary for the holding period returns.
horizon_to_period_dict = {
        '1_': 1,
        '3_': 3,
        str(FYTD) + '_': FYTD,
        '12_': 12,
        '24_': 24,
        '36_': 36,
        '60_': 60,
        '84_': 84
}

product_to_hurdle_dict = {
    'High Growth': 0.035,
    'Growth': 0.03,
    'Balanced Growth': 0.03,
    'Balanced': 0.02,
    'Conservative': 0.015,
    'Managed Cash': 0.0025,
    'Defined Benefit Strategy': 0
}

product_to_tax_dict = {
    'High Growth': 0.08,
    'Growth': 0.08,
    'Balanced Growth': 0.085,
    'Balanced': 0.10,
    'Conservative': 0.115,
    'Managed Cash': 0.15,
    'Defined Benefit Strategy': 0.08
}

df_up_monthly['Hurdle'] = [product_to_hurdle_dict[df_up_monthly['OptionType'][i]] for i in range(0, len(df_up_monthly))]
df_up_monthly['Hurdle'] = [(1 + df_up_monthly['Hurdle'][i])**(1/12) - 1 for i in range(0, len(df_up_monthly))]
df_up_monthly['Tax'] = [product_to_tax_dict[df_up_monthly['OptionType'][i]] for i in range(0, len(df_up_monthly))]
df_up_monthly['Tax'] = [(1 + df_up_monthly['Tax'][i])**(1/12) - 1 for i in range(0, len(df_up_monthly))]

# Calculates the holding period returns and annualises for periods greater than 12 months.
for horizon, period in horizon_to_period_dict.items():

    for return_type in ['Return', 'Core', 'Hurdle', 'Tax']:

        column_name = str(period) + '_' + return_type
        return_name = str(period) + '_Return'
        core_name = str(period) + '_Core'
        hurdle_name = str(period) + '_Hurdle'
        tax_name = str(period) + '_Tax'
        objective_name = str(period) + '_Objective'
        excess_name = str(period) + '_Excess'

        if period <= 12:
            df_up_monthly[column_name] = (
                df_up_monthly
                .groupby(['OptionType'])[return_type]
                .rolling(period)
                .apply(lambda r: np.prod(1+r)-1, raw=False)
                .reset_index(drop=False)[return_type]
            )

        elif period > 12:
            df_up_monthly[column_name] = (
                df_up_monthly
                .groupby(['OptionType'])[return_type]
                .rolling(period)
                .apply(lambda r: (np.prod(1+r)**(12/period))-1, raw=False)
                .reset_index(drop=False)[return_type]
            )

    df_up_monthly[objective_name] = (df_up_monthly[core_name] + df_up_monthly[hurdle_name]) * (1 - df_up_monthly[tax_name])
    df_up_monthly[excess_name] = df_up_monthly[return_name] - df_up_monthly[objective_name]

# Creates Risk of Loss Years table
df_risk_of_loss_years_F = df_up_monthly[df_up_monthly['Date'] == max(df_up_monthly['Date'])]
df_risk_of_loss_years_F = df_risk_of_loss_years_F[['OptionType', 'Date', str(FYTD) + '_Return']].dropna()
df_risk_of_loss_years_F = (df_risk_of_loss_years_F.pivot_table(index='Date', columns='OptionType', values=str(FYTD) + '_Return').sort_index(ascending=False)).round(4)*100

df_risk_of_loss_years_12 = df_up_monthly[df_up_monthly['Month'] == 6]
df_risk_of_loss_years_12 = df_risk_of_loss_years_12[['OptionType', 'Date', '12_Return']].dropna()
df_risk_of_loss_years_12 = (df_risk_of_loss_years_12.pivot_table(index='Date', columns='OptionType', values='12_Return').sort_index(ascending=False)).round(4)*100

df_risk_of_loss_years = pd.concat([df_risk_of_loss_years_F, df_risk_of_loss_years_12], axis=0)
del df_risk_of_loss_years_F
del df_risk_of_loss_years_12

df_risk_of_loss_years = df_risk_of_loss_years[
    [
        'High Growth',
        'Growth',
        'Balanced Growth',
        'Balanced',
        'Conservative',
        'Managed Cash',
        'Defined Benefit Strategy'
    ]
]


# Creates Product Performance table
# df_product = df_up_monthly[df_up_monthly['Date'] == report_date]
# df_product = df_product.set_index('OptionType')
#
# product_to_horizon_dict = {
#     'High Growth': 84,
#     'Growth': 60,
#     'Balanced Growth': 60,
#     'Balanced': 36,
#     'Conservative': 24,
#     'Managed Cash': 24,
#     'Defined Benefit Strategy': 60
# }
#
# product_name = list()
# product_performance = list()
# product_objective = list()
# product_excess = list()
# for product, horizon in product_to_horizon_dict.items():
#     product_name.append(product)
#     product_performance.append(df_product[str(horizon) + '_Return'][product])
#     product_objective.append(df_product[str(horizon) + '_Objective'][product])
#     product_excess.append(df_product[str(horizon) + '_Excess'][product])
#
# product_zipped = list(zip(product_performance, product_objective, product_excess))
# df_product = pd.DataFrame(product_zipped, index=product_name, columns=['Performance', 'Objective', 'Active']).round(4).T*100
# df_product_chart = df_product[:-1].T
# fig = df_product_chart.plot(kind='bar', color=[darkgreen, lightgreen])
# fig.set_title('LGS Annualised Product Performance')
# fig.set_ylabel('Performance %')
# fig.set_xlabel('')
# plt.tight_layout()


# Charting
# for horizon, period in horizon_to_period_dict.items():
#
#     for column in ['Excess']:
#
#         column_name = horizon + column
#         return_type = column
#
#         df_chart_temp = df_up_monthly[['OptionType', 'Date', column_name]]
#         df_chart_temp[column_name] = df_chart_temp[column_name]*100
#         df_chart_temp = df_chart_temp.pivot_table(index='Date', columns='OptionType', values=column_name)
#         if period >= 12:
#             period = int(period/12)
#             time_category = 'Year'
#         else:
#             time_category = 'Month'
#
#         chart_title = str(period) + ' ' + str(time_category) + ' Rolling Return for Each Strategy Since Inception'
#         ax = df_chart_temp.plot(title=chart_title, figsize=(16.8, 7.2))
#         ax.set_ylabel('Return %')
#         ax.axhline(y=0, linestyle=':', linewidth=1, color='k')
#         ax.legend(loc='lower left', title='')
#         fig = ax.get_figure()
#         fig.savefig('U:/CIO/#Investment_Report/Data/output/testing/product/' + horizon + 'chart.png')



# LIFECYCLES
df_lc = pd.read_excel(pd.ExcelFile('U:/CIO/#Data/input/jana/lifecycles/20191031 Lifecycles.xlsx'), sheet_name='Sheet2')
df_lc = df_lc.fillna(0)
df_lc = df_lc.set_index(['Lifecycle', 'OptionType']).T.unstack().reset_index(drop=False)
df_lc = df_lc.rename(columns={'level_2': 'Age', 0: 'Weight'})
df_lc['Weight'] = df_lc['Weight'] / 100

# Make data smaller
# df_lc = df_lc[df_lc['Lifecycle'] == 'Lifecycle 1'].reset_index(drop=True)

# Merges age cohorts with lifecycle portfolios
df_age = pd.merge(
    left=df_up_monthly,
    right=df_lc,
    left_on=['OptionType'],
    right_on=['OptionType']
)


# df_age['12_Weighted_Return'] = df_age['12_Return'] * df_age['Weight']

df_age['1_Weighted_Return'] = df_age['1_Return'] * df_age['Weight']
df_age['1_Weighted_Objective'] = df_age['1_Objective'] * df_age['Weight']

# df_age_final = df_age.groupby(['Age', 'Lifecycle', 'Date'])['12_Weighted_Return'].sum().reset_index(drop=False)

df_age_final = df_age.groupby(['Age', 'Lifecycle', 'Date'])['1_Weighted_Return', '1_Weighted_Objective'].sum().reset_index(drop=False)

# member_horizon_period_dict = {'2 Year': 2, '3 Year': 3, '5 Year': 5, '7 Year': 7}

member_horizon_period_dict = {
    '1 Year': 12,
    '2 Year': 24,
    '3 Year': 36,
    '4 Year': 48,
    '5 Year': 60,
    '6 Year': 72,
    '7 Year': 84
}

# Calculates Life Cycle Returns
for horizon, period in member_horizon_period_dict.items():

    for return_type in ['Weighted_Return', 'Weighted_Objective']:

        column_name = str(period) + '_' + return_type
        return_name = str(period) + '_Weighted_Return'
        objective_name = str(period) + '_Weighted_Objective'
        excess_name = str(period) + '_Weighted_Excess'

        df_age_final[column_name] = (
            df_age_final
            .groupby(['Age', 'Lifecycle'])['1_' + return_type]
            .rolling(period)
            .apply(lambda r: (np.prod(1+r)**(12/period))-1, raw=False)
            .reset_index(drop=False)['1_' + return_type]
        )

    df_age_final[excess_name] = df_age_final[return_name] - df_age_final[objective_name]

df_age_final = df_age_final.sort_values(['Lifecycle', 'Age', 'Date'])

df_age_final.to_csv('U:/CIO/#Research/MySuper Lifecycles.csv', index=False)

df_age_final['Year'] = [df_age_final['Date'][i].year for i in range(0, len(df_age_final))]
df_age_final['Month'] = [df_age_final['Date'][i].month for i in range(0, len(df_age_final))]


simulate_columns = [
    'Lifecycle',
    'Date',
    'Year',
    'Month',
    'Age',
    '12_Weighted_Return',
    '12_Weighted_Objective'
]

df_simulate = df_age_final[simulate_columns]
df_simulate = df_simulate.sort_values(['Lifecycle', 'Month', 'Year', 'Age']).reset_index(drop=True)
# df_simulate = df_simulate[df_simulate['Lifecycle'] == 'Lifecycle 1'].reset_index(drop=True)

lifespan = 5
weighted_return_lag_columns = list()
weighted_objective_lag_columns = list()
for year in range(0, lifespan):
    date_forward_column = 'Date Forward ' + str(year)
    weighted_return_lag_column = '12_Weighted_Return Lag ' + str(year)
    weighted_objective_lag_column = '12_Weighted_Objective Lag ' + str(year)
    weighted_return_lag_columns.append(weighted_return_lag_column)
    weighted_objective_lag_columns.append(weighted_objective_lag_column)

    df_simulate_temp = df_simulate[['Lifecycle', 'Age', 'Date', '12_Weighted_Return', '12_Weighted_Objective']]
    if year == 0:
        df_simulate_temp[date_forward_column] = df_simulate_temp['Date']
    else:
        df_simulate_temp[date_forward_column] = df_simulate_temp['Date'].apply(lambda date: date + pd.DateOffset(years=year))

    df_simulate_temp = df_simulate_temp.drop(columns=['Date'], axis=1)
    df_simulate_temp = df_simulate_temp.rename(
        columns={
            '12_Weighted_Return': weighted_return_lag_column,
            '12_Weighted_Objective': weighted_objective_lag_column
        }
    )

    df_simulate = pd.merge(
        left=df_simulate,
        right=df_simulate_temp,
        left_on=['Lifecycle', 'Age', 'Date'],
        right_on=['Lifecycle', 'Age', date_forward_column],
        how='left'
    )

    df_simulate[weighted_return_lag_column] = df_simulate.groupby(['Lifecycle', 'Date'])[weighted_return_lag_column].shift(year)
    df_simulate[weighted_objective_lag_column] = df_simulate.groupby(['Lifecycle', 'Date'])[weighted_objective_lag_column].shift(year)

    df_simulate = df_simulate.drop(columns=[date_forward_column], axis=1)

 # Converts df_simulate into panel
df_simulate = df_simulate.sort_values(['Lifecycle', 'Date', 'Age'])
df_simulate = df_simulate.drop(columns=['Year', 'Month', '12_Weighted_Return', '12_Weighted_Objective'], axis=1)
df_simulate = pd.wide_to_long(df_simulate, stubnames=['12_Weighted_Return Lag ', '12_Weighted_Objective Lag '], i=['Lifecycle', 'Date', 'Age'], j='Lag')
df_simulate = df_simulate.reset_index(drop=False)
df_simulate = df_simulate.sort_values(['Lifecycle', 'Date', 'Age', 'Lag'], ascending=[True, True, True, False]).reset_index(drop=True)
df_simulate = df_simulate.rename(columns={'12_Weighted_Return Lag ': '12_Weighted_Return', '12_Weighted_Objective Lag ': '12_Weighted_Objective'})

for return_type in ['12_Weighted_Return', '12_Weighted_Objective']:
    column_name = str(lifespan*12) + '_Lifecycle_' + return_type[12:]
    df_simulate[column_name] = (
        df_simulate
            .groupby(['Lifecycle', 'Date', 'Age'])[return_type]
            .rolling(lifespan)
            .apply(lambda r: (np.prod(1 + r) ** (1 /lifespan)) - 1, raw=False)
            .reset_index(drop=False)[return_type]
    )

df_simulate['Age Lag'] = df_simulate['Age'] - df_simulate['Lag']

df_simulate.to_csv('U:/CIO/#Research/lifecycle_simulation_5years_panel.csv', index=False)

df_simulate_chart_bar_summary = df_simulate[(df_simulate['Date'] == report_date) & (df_simulate['Lag'] == 0)]
df_simulate_chart_bar_summary = df_simulate_chart_bar_summary[df_simulate_chart_bar_summary['Age'].isin([50, 55, 60, 65])]
df_simulate_chart_bar_summary = ((df_simulate_chart_bar_summary[['Age', '60_Lifecycle_Return', '60_Lifecycle_Objective']].set_index('Age'))*100).round(2)
df_simulate_chart_bar_summary = df_simulate_chart_bar_summary.rename(columns={'60_Lifecycle_Return': 'Return', '60_Lifecycle_Objective': 'Objective'})
fig_simulate_chart_bar_summary = df_simulate_chart_bar_summary.plot(kind='bar', color=[darkgreen, lightgreen])
fig_simulate_chart_bar_summary.set_title('5 Year Return for Each Age Cohort')
fig_simulate_chart_bar_summary.set_ylabel('Return (%)')
fig_simulate_chart_bar_summary.set_xlabel('Age Cohort (Year)')
plt.tight_layout()


df_simulate_chart_cross_section = df_simulate[df_simulate['Date'] == report_date]
df_simulate_chart_cross_section = df_simulate_chart_cross_section.drop(columns=['Date', 'Lag', '60_Lifecycle_Return', '60_Lifecycle_Objective'], axis=1)
df_simulate_chart_cross_section = df_simulate_chart_cross_section[df_simulate_chart_cross_section['Age'].isin([50, 55, 60, 65])]

lifecycle_to_cross_section_dict = dict(list(df_simulate_chart_cross_section.groupby(['Lifecycle'])))
for lifecycle, df_cross_section in lifecycle_to_cross_section_dict.items():
    df_cross_section = df_cross_section.drop(columns=['Lifecycle'], axis=1)
    age_to_cross_section2_dict = dict(list(df_cross_section.groupby(['Age'])))

    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=True, figsize=(12.8, 7.2))
    i = 0
    j = 0
    for age, df_cross_section2 in age_to_cross_section2_dict.items():
        df_cross_section2 = df_cross_section2.drop(columns=['Age'], axis=1)
        df_cross_section2 = df_cross_section2.set_index('Age Lag')
        df_cross_section2 = df_cross_section2.rename(columns={'12_Weighted_Return': 'Return', '12_Weighted_Objective': 'Objective'})
        df_cross_section2 = (df_cross_section2 * 100).round(2)
        df_cross_section2.plot(ax=axes[i, j], kind='bar', color=[darkgreen, lightgreen])
        axes[i, j].set_title('1 Year Return at Each Age for a ' + str(age) + ' Year Old')
        axes[i, j].set_ylabel('Return (%)')
        axes[i, j].set_xlabel('Age (Years)')
        axes[i, j].legend(loc='upper left', title='')
        if i == 0 and j == 0:
            j += 1
        elif i == 0 and j == 1:
            i += 1
            j -= 1
        elif i == 1 and j == 0:
            j += 1

    fig.suptitle(lifecycle)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    fig.savefig('U:/CIO/#Investment_Report/Data/output/testing/charts/' + str(lifecycle) + '.png', dpi=300)



# CASH BENCHMARK
# df_r = pd.read_csv(
#         'U:/CIO/#Investment_Report/Data/input/returns/returns_2019-06-30.csv',
#         index_col='Date',
#         parse_dates=['Date'],
#         infer_datetime_format=True,
#         float_precision='round_trip',
#         usecols=['Date', 'AUBI_Index']
#         )
#
# df_r['2_Year'] = df_r['AUBI_Index'].rolling(24).apply(lambda r: (np.prod(1+r)**(1/2))-1, raw=True)
# df_benchmark['Cash'] = (df_r['2_Year'] + 0.0025) * (1 - 0.15)

# df_benchmark.to_csv('U:/CIO/#Investment_Report/Data/input/product/strategy_benchmarks_201905.csv')
