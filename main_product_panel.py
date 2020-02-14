import pandas as pd
import numpy as np
import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
FYTD = 6
report_date = dt.datetime(2019, 12, 31)
darkgreen = (75/256, 120/256, 56/256)
middlegreen = (141/256, 177/256, 66/256)
lightgreen = (175/256, 215/256, 145/256)

# UNIT PRICES
# Imports unit price panel
df_up = pd.read_csv('U:/CIO/#Data/input/lgs/unitprices/20191231 Unit Prices.csv', parse_dates=['Date'])
df_up_unique = df_up[df_up['Date'] == report_date]
# df_up_unique.to_csv('U:/CIO/#Research/product_unique.csv', index=False)

# Renames columns
df_up = df_up.rename(
    columns={
        'InvestmentOption.InvestmentComponent.Name': 'Component',
        'InvestmentOption.InvestmentOptionType.Name': 'OptionType'
    }
)

# Selects unit prices of strategies that are Accumulation Scheme or Defined Benefit
df_up = (
    df_up[
        (df_up['Component'].isin(['Accumulation Scheme', 'Defined Benefit Scheme'])) |
        ((df_up['Component'].isin(['Contributor Financed Benefit'])) & (df_up['OptionType'].isin(['Growth'])))
    ]
    .drop(columns=['Component'], axis=1)
    .sort_values(['OptionType', 'Date'])
    .reset_index(drop=True)
)

lgs_strategy_name_list = []
for i in range(0,len(df_up)):
    if df_up['OptionType'][i] == 'Defined Benefit Strategy':
        lgs_strategy_name_list.append('Employer Reserve')
    else:
        lgs_strategy_name_list.append(df_up['OptionType'][i])
df_up['OptionType'] = lgs_strategy_name_list

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
df_up_monthly_check = df_up_monthly_check.to_csv('U:/CIO/#Investment_Report/Data/output/testing/mysuper/df_up_monthly_check.csv', index=False)

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
    if df_up_monthly['OptionType'][i] == 'Employer Reserve':
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
    'Employer Reserve': 0
}

product_to_tax_dict = {
    'High Growth': 0,
    'Growth': 0,
    'Balanced Growth': 0,
    'Balanced': 0,
    'Conservative': 0,
    'Managed Cash': 0,
    'Employer Reserve': 0
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

df_up_monthly.to_csv('U:/CIO/#Investment_Report/Data/output/testing/mysuper/df_up_monthly.csv', index=False)

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
        'Employer Reserve'
    ]
]


# Creates Product Performance table
df_product = df_up_monthly[df_up_monthly['Date'] == report_date]
df_product = df_product.set_index('OptionType')

product_to_horizon_dict = {
    'High Growth': 84,
    'Growth': 60,
    'Balanced Growth': 60,
    'Balanced': 36,
    'Conservative': 24,
    'Managed Cash': 24,
    'Employer Reserve': 60
}

product_name = list()
product_performance = list()
product_objective = list()
product_excess = list()
for product, horizon in product_to_horizon_dict.items():
    product_name.append(product)
    product_performance.append(df_product[str(horizon) + '_Return'][product])
    product_objective.append(df_product[str(horizon) + '_Objective'][product])
    product_excess.append(df_product[str(horizon) + '_Excess'][product])

product_zipped = list(zip(product_performance, product_objective, product_excess))
df_product = pd.DataFrame(product_zipped, index=product_name, columns=['Performance', 'Objective', 'Active']).round(4).T*100
df_product_chart = df_product[:-1].T
fig = df_product_chart.plot(kind='bar', color=[darkgreen, lightgreen])
fig.set_title('LGS Annualised Product Performance')
fig.set_ylabel('Performance %')
fig.set_xlabel('')
plt.tight_layout()


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