import pandas as pd
import numpy as np
import datetime as dt
FYTD = 4
report_date = dt.datetime(2019, 10, 31)

# UNIT PRICES
# Imports unit price panel
df_up = pd.read_csv('U:/CIO/#Investment_Report/Data/input/testing/20191031 Unit Prices.csv', parse_dates=['Date'])
df_up = df_up.rename(
    columns={
        'InvestmentOption.InvestmentComponent.Name': 'Component',
        'InvestmentOption.InvestmentOptionType.Name': 'OptionType'
    }
)

# Renames columns
df_up = (
    df_up[df_up['Component'].isin(['Accumulation Scheme', 'Defined Benefits'])]
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
    'U:/CIO/#Investment_Report/Data/input/product/g1-data_201909.csv',
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

product_to_objective_dict = {
    'High Growth': 0.035,
    'Growth': 0.03,
    'Balanced Growth': 0.03,
    'Balanced': 0.02,
    'Conservative': 0.015,
    'Managed Cash': 0.0025
}

product_to_tax_dict = {
    'High Growth': 0.08,
    'Growth': 0.08,
    'Balanced Growth': 0.085,
    'Balanced': 0.10,
    'Conservative': 0.115,
    'Managed Cash': 0.15
}

df_up_monthly['Objective'] = [product_to_objective_dict[df_up_monthly['OptionType'][i]] for i in range(0, len(df_up_monthly))]
df_up_monthly['Objective'] = [(1 + df_up_monthly['Objective'][i])**(1/12) - 1 for i in range(0, len(df_up_monthly))]
df_up_monthly['Tax'] = [product_to_tax_dict[df_up_monthly['OptionType'][i]] for i in range(0, len(df_up_monthly))]
df_up_monthly['Tax'] = [(1 + df_up_monthly['Tax'][i])**(1/12) - 1 for i in range(0, len(df_up_monthly))]

# Calculates the holding period returns and annualises for periods greater than 12 months.
for horizon, period in horizon_to_period_dict.items():

    for return_type in ['Return', 'Inflation', 'Objective', 'Tax']:

        column_name = str(period) + '_' + return_type
        return_name = str(period) + '_Return'
        inflation_name = str(period) + '_Inflation'
        objective_name = str(period) + '_Objective'
        tax_name = str(period) + '_Tax'
        benchmark_name = str(period) + '_Benchmark'
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

    df_up_monthly[benchmark_name] = (df_up_monthly[inflation_name] + df_up_monthly[objective_name]) * (1 - df_up_monthly[tax_name])
    df_up_monthly[excess_name] = df_up_monthly[return_name] - df_up_monthly[benchmark_name]

df_up_risk_of_loss_years = df_up_monthly[(df_up_monthly['Month'] == 6) | (df_up_monthly['Date'] == max(df_up_monthly['Date']))]

df_up_bar_chart = df_up_monthly[df_up_monthly['Date']==report_date]



"""
# Charting
for horizon, period in horizon_to_period_dict.items():

    for column in ['Return']:

        column_name = horizon + column
        return_type = column

        df_chart_temp = df_up_monthly[['OptionType', 'Date', column_name]]
        df_chart_temp[column_name] = df_chart_temp[column_name]*100
        df_chart_temp = df_chart_temp.pivot_table(index='Date', columns='OptionType', values=column_name)
        if period >= 12:
            period = int(period/12)
            time_category = 'Year'
        else:
            time_category = 'Month'

        chart_title = str(period) + ' ' + str(time_category) + ' Rolling Return for Each Strategy Since Inception'
        ax = df_chart_temp.plot(title=chart_title, figsize=(16.8, 7.2))
        ax.set_ylabel('Return %')
        ax.axhline(y=0, linestyle=':', linewidth=1, color='k')
        ax.legend(loc='lower left', title='')
        fig = ax.get_figure()
        fig.savefig('U:/CIO/#Investment_Report/Data/output/testing/product/' + horizon + 'chart.png')
"""

"""
# LIFECYCLES
df_lc = pd.read_excel(pd.ExcelFile('U:/CIO/#Investment_Report/Data/input/testing/20191031 Lifecycles.xlsx'), sheet_name='Sheet2')
df_lc = df_lc.fillna(0)
df_lc = df_lc.set_index(['Lifecycle', 'OptionType']).T.unstack().reset_index(drop=False)
df_lc = df_lc.rename(columns={'level_2': 'Age', 0: 'Weight'})
df_lc['Weight'] = df_lc['Weight'] / 100


# df_age = pd.merge(
#     left=df_up_risk_of_loss_years,
#     right=df_lc,
#     left_on=['OptionType'],
#     right_on=['OptionType']
# )

df_age = pd.merge(
    left=df_up_monthly,
    right=df_lc,
    left_on=['OptionType'],
    right_on=['OptionType']
)


# df_age['12_Weighted_Return'] = df_age['12_Return'] * df_age['Weight']

df_age['1_Weighted_Return'] = df_age['1_Return'] * df_age['Weight']

# df_age_final = df_age.groupby(['Age', 'Lifecycle', 'Date'])['12_Weighted_Return'].sum().reset_index(drop=False)

df_age_final = df_age.groupby(['Age', 'Lifecycle', 'Date'])['1_Weighted_Return'].sum().reset_index(drop=False)

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

for horizon, period in member_horizon_period_dict.items():

    df_age_final[horizon] = (
        df_age_final
        .groupby(['Age', 'Lifecycle'])['1_Weighted_Return']
        .rolling(period)
        .apply(lambda r: (np.prod(1+r)**(12/period))-1, raw=False)
        .reset_index(drop=False)['1_Weighted_Return']
    )

df_age_final = df_age_final.sort_values(['Lifecycle', 'Age', 'Date'])

df_age_final.to_csv('U:/CIO/#Research/MySuper Lifecycles.csv', index=False)

df_age_final['Month'] = [df_age_final['Date'][i].month for i in range(0, len(df_age_final))]
"""



# STRATEGY BENCHMARKS
# df_cpi = pd.read_csv(
#     'U:/CIO/#Investment_Report/Data/input/product/g1-data_201906.csv',
#     header=10,
#     usecols=['Series ID', 'GCPIAGQP'],
#     parse_dates=['Series ID'],
#     index_col=['Series ID']
# )
#
# df_cpi = df_cpi.rename(columns={'GCPIAGQP': 'Inflation'})
# df_cpi['Inflation'] = df_cpi['Inflation']/100
# # df_cpi = df_cpi.reset_index(drop=True)
#
# df_cpi = df_cpi[df_cpi.index >= dt.datetime(1999, 1, 1)]
#
#
# years = [1, 2, 3, 4, 5, 6, 7]
# for year in years:
#     column_name = str(year) + '_Year'
#     quarters = year*4
#     df_cpi[column_name] = df_cpi['Inflation'].rolling(quarters).apply(lambda r: (np.prod(1+r)**(1/year))-1, raw=True)
#
# df_benchmark = pd.DataFrame()
# df_benchmark['High Growth'] = (df_cpi['7_Year'] + 0.035) * (1 - 0.08)
# df_benchmark['Growth'] = (df_cpi['5_Year'] + 0.03) * (1 - 0.08)
# df_benchmark['Balanced Growth'] = (df_cpi['5_Year'] + 0.03) * (1 - 0.085)
# df_benchmark['Balanced'] = (df_cpi['3_Year'] + 0.02) * (1 - 0.1)
# df_benchmark['Conservative'] = (df_cpi['2_Year'] + 0.015) * (1 - 0.115)
# df_benchmark['Employer Reserve'] = (0.0575) * (1 - 0.08)
# df_benchmark = df_benchmark.resample('M').pad()


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
