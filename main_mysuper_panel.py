import pandas as pd
import numpy as np
import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
FYTD = 10
report_date = dt.datetime(2020, 4, 30)
darkgreen = (75/256, 120/256, 56/256)
middlegreen = (141/256, 177/256, 66/256)
lightgreen = (175/256, 215/256, 145/256)

lgs_unit_prices_filepath = 'U:/CIO/#Data/input/lgs/unitprices/20200430 Unit Prices.csv'
rba_cpi_filepath = 'U:/CIO/#Data/input/rba/inflation/20200430 g1-data.csv'

use_managerid = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12]
use_accountid = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
footnote_rows = 28

# UNIT PRICES
# Imports unit price panel
df_up = pd.read_csv(lgs_unit_prices_filepath, parse_dates=['Date'])
df_up_unique = df_up[df_up['Date'] == report_date]
# df_up_unique.to_csv('U:/CIO/#Research/product_unique.csv', index=False)

# Renames columns
df_up = df_up.rename(
    columns={
        'InvestmentOption.InvestmentComponent.Name': 'Component',
        'InvestmentOption.InvestmentOptionType.Name': 'OptionType'
    }
)

df_up = (
    df_up[
        (df_up['Component'].isin(['Accumulation Scheme', 'Defined Benefit Scheme'])) |
        ((df_up['Component'].isin(['Deferred Benefit'])) & (df_up['OptionType'].isin(['Growth'])))
    ]
    .drop(columns=['Component'], axis=1)
    .sort_values(['OptionType', 'Date'])
    .reset_index(drop=True)
)


# Removes SAS which is a closed account
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

# Down-samples the data using actual dates and checks for months which ended before 28 days
df_up_monthly_check = df_up.copy()
df_up_monthly_check['Year'] = [df_up_monthly_check['Date'][i].year for i in range(0, len(df_up_monthly_check))]
df_up_monthly_check['Month'] = [df_up_monthly_check['Date'][i].month for i in range(0, len(df_up_monthly_check))]
df_up_monthly_check['Day'] = [df_up_monthly_check['Date'][i].day for i in range(0, len(df_up_monthly_check))]
df_up_monthly_check = df_up_monthly_check.groupby(['OptionType', 'Year', 'Month']).tail(1).reset_index(drop=True)
df_up_monthly_check = df_up_monthly_check[df_up_monthly_check['Day'] < 28]
df_up_monthly_check = df_up_monthly_check.to_csv('U:/CIO/#Data/output/investment/mysuper/df_up_monthly_check.csv', index=False)

# Calculates 1 month return
df_up_monthly['Unit Price Lag 1'] = df_up_monthly.groupby('OptionType')['Unit Price'].shift(1)
df_up_monthly['Return'] = (df_up_monthly['Unit Price'] - df_up_monthly['Unit Price Lag 1']) / df_up_monthly['Unit Price Lag 1']

# STRATEGY BENCHMARKS
df_cpi = pd.read_csv(
    rba_cpi_filepath,
    header=10,
    usecols=['Series ID', 'GCPIAGQP'],
    parse_dates=['Series ID'],
    index_col=['Series ID']
)

# Converts quarterly inflation into monthly
df_cpi = df_cpi.rename(columns={'GCPIAGQP': 'Inflation'})
df_cpi = df_cpi[df_cpi.index >= dt.datetime(1999, 1, 1)]
df_cpi['Inflation'] = df_cpi['Inflation']/100
df_cpi['Inflation'] = ((1 + df_cpi['Inflation']) ** (1/3)) - 1
df_cpi = df_cpi.resample('M').pad()
df_cpi['Inflation'] = df_cpi['Inflation'].shift(-2).ffill()
df_cpi = df_cpi.reset_index(drop=False)
df_cpi = df_cpi.rename(columns={'Series ID': 'Date'})

df_cpi.to_csv('U:/CIO/#Research/inflation_output.csv', index=False)

# Merges monthly unit price returns with inflation
df_up_monthly = pd.merge(
    left=df_up_monthly,
    right=df_cpi,
    left_on=['Date'],
    right_on=['Date']
).reset_index(drop=True)

# Creates core_benchmark. For High Growth, Balanced Growth, Balanced, Conservative, and Growth is inflation.
# core_benchmark for employer reserve is a constant 5.75% pa.
employer_reserve_benchmark_monthly = ((1 + 0.0575)**(1/12) - 1)
core_benchmark = []
for i in range(0, len(df_up_monthly)):
    if df_up_monthly['OptionType'][i] == 'Defined Benefit Strategy':
        core_benchmark.append(employer_reserve_benchmark_monthly)
    else:
        core_benchmark.append(df_up_monthly['Inflation'][i])
df_up_monthly['Core'] = core_benchmark

# Changes leap years to normal years for merging on month-end
non_leap_dates = []
for i in range(0, len(df_up_monthly)):
    if df_up_monthly['Date'][i].month == 2 and df_up_monthly['Date'][i].day == 29:
        non_leap_dates.append(df_up_monthly['Date'][i] - pd.Timedelta(days=1))
    else:
        non_leap_dates.append(df_up_monthly['Date'][i])
df_up_monthly['Date'] = non_leap_dates

# Sorts dataframe for calculation
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
        '84_': 84,
        '120_': 120,
}

# Sets product to hurdle dictionary for MySuper Chart
product_to_hurdle_dict = {
    'High Growth': 0.03,
    'Growth': 0.03,
    'Balanced Growth': 0.03,
    'Balanced': 0.03,
    'Conservative': 0.03,
    'Managed Cash': 0.03,
    'Defined Benefit Strategy': 0
}


# Sets product to tax dictionary
product_to_tax_dict = {
    'High Growth': 0,
    'Growth': 0,
    'Balanced Growth': 0,
    'Balanced': 0,
    'Conservative': 0,
    'Managed Cash': 0,
    'Defined Benefit Strategy': 0
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

df_up_monthly.to_csv('U:/CIO/#Data/output/investment/mysuper/df_up_monthly_mysuper.csv', index=False)

# LIFECYCLES
df_lc = pd.read_excel(pd.ExcelFile('U:/CIO/#Data/input/jana/lifecycles/20200131 Lifecycles.xlsx'), sheet_name='Sheet2')
df_lc = df_lc.fillna(0)
df_lc = df_lc.set_index(['Lifecycle', 'OptionType']).T.unstack().reset_index(drop=False)
df_lc = df_lc.rename(columns={'level_2': 'Age', 0: 'Weight'})
df_lc['Weight'] = df_lc['Weight'] / 100

# Select only certain lifecycles
df_lc = df_lc[df_lc['Lifecycle'].isin(['Lifecycle 1'])].reset_index(drop=True)

# Merges age cohorts with lifecycle portfolios
df_age = pd.merge(
    left=df_up_monthly,
    right=df_lc,
    left_on=['OptionType'],
    right_on=['OptionType']
)

df_age['1_Weighted_Return'] = df_age['1_Return'] * df_age['Weight']
df_age['1_Weighted_Objective'] = df_age['1_Objective'] * df_age['Weight']

df_age_final = df_age.groupby(['Age', 'Lifecycle', 'Date'])['1_Weighted_Return', '1_Weighted_Objective'].sum().reset_index(drop=False)

# Creates horizon dictionary to loop over for horizon calculation
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

df_age_final.to_csv('U:/CIO/#Data/output/investment/mysuper/MySuper Lifecycles.csv', index=False)

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

lifespan = 10
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

df_simulate.to_csv('U:/CIO/#Data/output/investment/mysuper/lifecycle_simulation_5years_panel.csv', index=False)

df_simulate_chart_bar_summary = df_simulate[(df_simulate['Date'] == report_date) & (df_simulate['Lag'] == 0)]
df_simulate_chart_bar_summary = df_simulate_chart_bar_summary[df_simulate_chart_bar_summary['Age'].isin([50, 55, 60, 65])]
df_simulate_chart_bar_summary = ((df_simulate_chart_bar_summary[['Age', '120_Lifecycle_Return', '120_Lifecycle_Objective']].set_index('Age'))*100).round(2)
df_simulate_chart_bar_summary = df_simulate_chart_bar_summary.rename(columns={'120_Lifecycle_Return': 'Return', '120_Lifecycle_Objective': 'Objective'})
fig_simulate_chart_bar_summary = df_simulate_chart_bar_summary.plot(kind='bar', color=[darkgreen, lightgreen])
# fig_simulate_chart_bar_summary.set_title('5 Year Return for Each Age Cohort')
fig_simulate_chart_bar_summary.set_ylabel('Return (%)')
fig_simulate_chart_bar_summary.set_xlabel('Age Cohort (Year)')
plt.tight_layout()
fig_bar = fig_simulate_chart_bar_summary.get_figure()
fig_bar.savefig('U:/CIO/#Data/output/investment/mysuper/monitor.PNG', dpi=300)

df_simulate_chart_cross_section = df_simulate[df_simulate['Date'] == report_date]
df_simulate_chart_cross_section = df_simulate_chart_cross_section.drop(columns=['Date', 'Lag', '120_Lifecycle_Return', '120_Lifecycle_Objective'], axis=1)
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
    fig.savefig('U:/CIO/#Data/output/investment/mysuper/' + str(lifecycle) + '.png', dpi=300)

