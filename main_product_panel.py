import pandas as pd
import numpy as np
import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
import requests
from pandas import pandas as pd
from bs4 import BeautifulSoup

# Start User Input
FYTD = 10
report_date = dt.datetime(2020, 4, 30)
darkgreen = (75/256, 120/256, 56/256)
middlegreen = (141/256, 177/256, 66/256)
lightgreen = (175/256, 215/256, 145/256)

lgs_unit_prices_filepath = 'U:/CIO/#Data/input/lgs/unitprices/20200430 Unit Prices.csv'
jpm_main_benchmarks_filepath = 'U:/CIO/#Data/input/jpm/performance/2020/04/Historical Time Series - Monthly - Main Benchmarks.xlsx'
rba_cpi_filepath = 'U:/CIO/#Data/input/rba/inflation/20200430 g1-data.csv'
lgs_website_return_acc_filepath = 'U:/CIO/#Data/input/lgs/website/investment_returns/2020/04/InvestmentReturns_acc.csv'
lgs_website_return_dbg_filepath = 'U:/CIO/#Data/input/lgs/website/investment_returns/2020/04/InvestmentReturns_dbg.csv'
lgs_website_return_dbs_filepath = 'U:/CIO/#Data/input/lgs/website/investment_returns/2020/04/InvestmentReturns_dbs.csv'

use_managerid = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12]
use_accountid = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
footnote_rows = 28
# End User Input


# UNIT PRICES
# Imports unit price panel
df_up = pd.read_csv(lgs_unit_prices_filepath, parse_dates=['Date'])


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

# Creates year, month, and day indicators
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
df_up_monthly_check.to_csv('U:/CIO/#Data/output/investment/product/df_up_monthly_check_product.csv', index=False)

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

df_cpi.to_csv('U:/CIO/#Data/output/investment/product/inflation_output.csv', index=False)

# Merges monthly unit price returns with inflation
df_up_monthly = pd.merge(
    left=df_up_monthly,
    right=df_cpi,
    left_on=['Date'],
    right_on=['Date']
).reset_index(drop=True)


# Creates core_benchmark for managed cash
df_jpm_main_benchmarks = pd.read_excel(
    pd.ExcelFile(jpm_main_benchmarks_filepath),
    sheet_name='Sheet1',
    skiprows=use_accountid,
    skipfooter=footnote_rows,
    header=1
)


# Reshapes the time-series into a panel.
df_jpm_main_benchmarks = df_jpm_main_benchmarks.rename(columns={'Unnamed: 0': 'Date'})
df_jpm_main_benchmarks = df_jpm_main_benchmarks.set_index('Date')
df_jpm_main_benchmarks = df_jpm_main_benchmarks.transpose()
df_jpm_main_benchmarks = df_jpm_main_benchmarks.reset_index(drop=False)
df_jpm_main_benchmarks = df_jpm_main_benchmarks.rename(columns={'index': 'Manager'})
df_jpm_main_benchmarks = pd.melt(df_jpm_main_benchmarks, id_vars=['Manager'], value_name='Values')
df_jpm_main_benchmarks = df_jpm_main_benchmarks.sort_values(['Manager', 'Date'])
df_jpm_main_benchmarks = df_jpm_main_benchmarks.reset_index(drop=True)
df_jpm_main_benchmarks = df_jpm_main_benchmarks.replace('-', np.NaN)

# Creates Rf from Cash Aggregate Benchmark
df_jpm_rf = df_jpm_main_benchmarks[df_jpm_main_benchmarks['Manager'].isin(['CLFACASH', 'Cash Aggregate'])].reset_index(drop=True)
df_jpm_rf = df_jpm_rf.rename(columns={'Values': 'JPM_Rf'})
df_jpm_rf['JPM_Rf'] = df_jpm_rf['JPM_Rf']/100

# # Infers the risk-free rate from the Cash +0.2% benchmark, the +0.2% benchmark started November 2019.
# rf_values = []
# new_cash_benchmark_date = dt.datetime(2019, 11, 30)
# for i in range(0, len(df_jpm_rf)):
#     if df_jpm_rf['Date'][i] >= new_cash_benchmark_date:
#         rf_values.append(df_jpm_rf['JPM_Rf'][i] - (((1+0.002)**(1/12))-1))
#     else:
#         rf_values.append(df_jpm_rf['JPM_Rf'][i])
# df_jpm_rf['JPM_Rf'] = rf_values
# df_jpm_rf = df_jpm_rf.drop(columns=['Manager'], axis=1)

df_up_monthly = pd.merge(
    left=df_up_monthly,
    right=df_jpm_rf,
    left_on=['Date'],
    right_on=['Date'],
    how='outer'
).reset_index(drop=True)


# Creates core_benchmark. For High Growth, Balanced Growth, Balanced, Conservative, and Growth is inflation.
# core_benchmark for managed cash is Inflation, employer reserve is a constant 5.75% pa.
employer_reserve_benchmark_monthly = ((1 + 0.0575)**(1/12) - 1)
core_benchmark = []
for i in range(0, len(df_up_monthly)):
    if df_up_monthly['OptionType'][i] == 'Managed Cash':
        core_benchmark.append(df_up_monthly['JPM_Rf'][i])
    elif df_up_monthly['OptionType'][i] == 'Defined Benefit Strategy':
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

# Sets product to hurdle dictionary for Products
product_to_hurdle_dict = {
    'High Growth': 0.035,
    'Growth': 0.03,
    'Balanced Growth': 0.03,
    'Balanced': 0.02,
    'Conservative': 0.015,
    'Managed Cash': 0,
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

df_up_monthly.to_csv('U:/CIO/#Data/output/investment/product/df_up_monthly_product.csv', index=False)

# Creates Risk of Loss Years table
df_risk_of_loss_years_FYTD = df_up_monthly[df_up_monthly['Date'] == max(df_up_monthly['Date'])]
df_risk_of_loss_years_FYTD = df_risk_of_loss_years_FYTD[['OptionType', 'Date', str(FYTD) + '_Return']].dropna()
df_risk_of_loss_years_FYTD = (df_risk_of_loss_years_FYTD.pivot_table(index='Date', columns='OptionType', values=str(FYTD) + '_Return').sort_index(ascending=False)).round(4)*100

df_risk_of_loss_years_12 = df_up_monthly[df_up_monthly['Month'] == 6]
df_risk_of_loss_years_12 = df_risk_of_loss_years_12[['OptionType', 'Date', '12_Return']].dropna()
df_risk_of_loss_years_12 = (df_risk_of_loss_years_12.pivot_table(index='Date', columns='OptionType', values='12_Return').sort_index(ascending=False)).round(4)*100

df_risk_of_loss_years = pd.concat([df_risk_of_loss_years_FYTD, df_risk_of_loss_years_12], axis=0)
del df_risk_of_loss_years_FYTD
del df_risk_of_loss_years_12

df_risk_of_loss_years = df_risk_of_loss_years[
    [
        'High Growth',
        'Balanced Growth',
        'Balanced',
        'Conservative',
        'Managed Cash',
        'Growth',
        'Defined Benefit Strategy'
    ]
]

df_risk_of_loss_years = df_risk_of_loss_years.reset_index(drop=False)
df_risk_of_loss_years['Year'] = [int(df_risk_of_loss_years['Date'][i].year) for i in range(0, len(df_risk_of_loss_years))]
df_risk_of_loss_years = df_risk_of_loss_years.set_index('Year')
df_risk_of_loss_years = df_risk_of_loss_years.rename(index={2020: 'FYTD'})
df_risk_of_loss_years = df_risk_of_loss_years.reset_index(drop=False)
df_risk_of_loss_years = df_risk_of_loss_years.drop(columns=('Date'), axis=1)

# df_risk_of_loss_years.to_latex('U:/CIO/#Data/output/investment/product/product_risk_years.tex')

# Creates Product Performance table
df_product = df_up_monthly[df_up_monthly['Date'] == report_date]
df_product = df_product.set_index('OptionType')

product_to_horizon_dict = {
    'High Growth': 84,
    'Balanced Growth': 60,
    'Balanced': 36,
    'Conservative': 24,
    'Managed Cash': 24,
    'Growth': 60,
    'Defined Benefit Strategy': 60
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
with open('U:/CIO/#Data/output/investment/product/product_performance.tex', 'w') as tf:
    tf.write(df_product.to_latex(index=True, na_rep='', multicolumn_format='c', column_format='lRRRRRRR'))

# Create product performance chart
df_product_chart = df_product[:-1].T
fig = df_product_chart.plot(kind='bar', color=[darkgreen, lightgreen])
fig.set_title('LGS Annualised Product Performance')
fig.set_ylabel('Performance %')
fig.set_xlabel('')
plt.tight_layout()
fig.get_figure().savefig('U:/CIO/#Data/output/investment/product/product_performance.png', dpi=300)


# Create Risk Table and Risk Chart
# df_risk_of_loss_years = df_risk_of_loss_years.set_index('Year')
# loss_years = df_risk_of_loss_years.lt(0).sum()
# gain_years = df_risk_of_loss_years.gt(0).sum()
#
# df_loss_years_percentage = pd.DataFrame((loss_years/(loss_years+gain_years))).T.rename(index={0: 'Actual loss years (%)'})
# df_loss_years = pd.DataFrame(loss_years).T.rename(index={0: 'No. loss years'})
# df_target_loss_years = pd.DataFrame()
# df_forecast_loss_years = pd.DataFrame()
# # df_expected_loss_years = pd.DataFrame((loss_years/(loss_years+gain_years))*20).T.rename(index={0: 'No. Expected loss years per 20 years'})
# df_total_years = pd.DataFrame(loss_years+gain_years).T.rename(index={0: 'No. years'})
#
# df_product_risk = pd.concat(
#     [
#         df_loss_years_percentage,
#         df_loss_years,
#         df_target_loss_years,
#         df_forecast_loss_years,
#         # df_expected_loss_years,
#         df_total_years
#     ],
#     axis=0
# ).round(2)
#
# df_risk_of_loss_years = df_risk_of_loss_years.reset_index(drop=False)


# Creates Checker
df_lgs_website_return_acc = pd.read_csv(lgs_website_return_acc_filepath, skiprows=[0, 1, 2, 3, 4, 5, 6])
df_lgs_website_return_dbg = pd.read_csv(lgs_website_return_dbg_filepath, skiprows=[0, 1, 2, 3, 4, 5, 6])
df_lgs_website_return_dbs = pd.read_csv(lgs_website_return_dbs_filepath, skiprows=[0, 1, 2, 3, 4, 5, 6])

# Removes SAS from Accumulation and Selects only Growth from Deferred Benefits
df_lgs_website_return_acc = df_lgs_website_return_acc[~df_lgs_website_return_acc['Investment Option'].isin(['Sustainable Australian Shares'])].reset_index(drop=True)
df_lgs_website_return_dbg = df_lgs_website_return_dbg[df_lgs_website_return_dbg['Investment Option'].isin(['Growth'])].reset_index(drop=True)

# Concatenates the website returns into a single dataframe
df_lgs_website_return = pd\
    .concat(
    [
        df_lgs_website_return_acc,
        df_lgs_website_return_dbg,
        df_lgs_website_return_dbs
    ], axis=0)\
    .sort_values('Investment Option')\
    .reset_index(drop=True)

website_return_to_internal_return_dictionary = {
    '1mth': '1_Return',
    '3mths': '3_Return',
    'FYTD': str(FYTD) + '_Return',
    '12mts': '12_Return',
    '3yrs': '36_Return',
    '5yrs': '60_Return',
    '7yrs': '84_Return',
    '10yrs': '120_Return'
}

website_return_column_list = list(website_return_to_internal_return_dictionary.keys())
internal_return_column_list = list(website_return_to_internal_return_dictionary.values())

df_up_monthly_current = df_up_monthly[df_up_monthly['Date'].isin([report_date])].reset_index(drop=True)
df_up_monthly_current = df_up_monthly_current[
    [
        'OptionType',
        '1_Return',
        '3_Return',
        str(FYTD) + '_Return',
        '12_Return',
        '36_Return',
        '60_Return',
        '84_Return',
        '120_Return'
    ]
].sort_values('OptionType').round(4)

# df_lgs_website_return[website_return_column_list] = df_lgs_website_return[website_return_column_list]/100
df_up_monthly_current[internal_return_column_list] = df_up_monthly_current[internal_return_column_list]*100

df_check_returns1 = pd.merge(
    left=df_up_monthly_current,
    right=df_lgs_website_return,
    left_on=['OptionType'],
    right_on=['Investment Option']
)

df_check_returns2 = pd.DataFrame()
strategy_deviation_list = []
month_deviation_list = []
value_deviation_list = []

for website_return, internal_return in website_return_to_internal_return_dictionary.items():
    deviation_column_name = internal_return[:-7] + '_Deviation'
    df_check_returns1[deviation_column_name] = (df_check_returns1[internal_return] - df_check_returns1[website_return]).round(2)

    for i in range(0, len(df_check_returns1)):
        if abs(df_check_returns1[deviation_column_name][i]) >= 0.01:
            strategy_deviation_list.append(df_check_returns1['OptionType'][i])
            month_deviation_list.append(deviation_column_name)
            value_deviation_list.append(df_check_returns1[deviation_column_name][i])

df_check_returns2['Strategy'] = strategy_deviation_list
df_check_returns2['Month'] = month_deviation_list
df_check_returns2['Value'] = value_deviation_list

print('This Impacts Table 3.1 at:\n')
df_check_returns3 = pd.DataFrame()
for i in range(0, len(df_check_returns2)):
    if (
            (df_check_returns2['Strategy'][i] == 'High Growth' and df_check_returns2['Month'][i][:2] == '84') or
            (df_check_returns2['Strategy'][i] == 'Balanced Growth' and df_check_returns2['Month'][i][:2] == '60') or
            (df_check_returns2['Strategy'][i] == 'Balanced' and df_check_returns2['Month'][i][:2] == '36') or
            (df_check_returns2['Strategy'][i] == 'Growth' and df_check_returns2['Month'][i][:2] == '60') or
            (df_check_returns2['Strategy'][i] == 'Defined Benefit Strategy' and df_check_returns2['Month'][i][:2] == '60')
    ):
        df_check_returns3 = df_check_returns3.append(df_check_returns2.loc[i])
        print(df_check_returns2.loc[i], '\n')


# Creates checker for yearly returns
# Request Allocation Table High Growth and creates a dataframe - as at 19/8/2019 Table number is 20
res = requests.get('https://www.lgsuper.com.au/investments/performance/accumulation-scheme/')
soup = BeautifulSoup(res.content, 'lxml')
table = soup.find_all('table')
df_lgs_website_yearly_return_acc = pd.read_html(str(table))[2]
df_lgs_website_yearly_return_acc = df_lgs_website_yearly_return_acc.set_index('Year')
df_lgs_website_yearly_return_acc = df_lgs_website_yearly_return_acc.drop(columns=['Sustainable Australian Shares (%)'], axis=1)

# Request Allocation Table Deferred benefit growth  - as at 19/8/2019 Table number is 20
res = requests.get('https://www.lgsuper.com.au/investments/performance/retirement-scheme/deferred-benefit/')
soup = BeautifulSoup(res.content, 'lxml')
table = soup.find_all('table')
df_lgs_website_yearly_return_dbg = pd.read_html(str(table))[2]
df_lgs_website_yearly_return_dbg = df_lgs_website_yearly_return_dbg.set_index('Year')
df_lgs_website_yearly_return_dbg = df_lgs_website_yearly_return_dbg.drop(columns=['High Growth (%)', 'Balanced (%)', 'Conservative (%)' , 'Managed Cash (%)', 'Balanced Growth (%)'], axis=1)

# Request Allocation Table Defined benefit  - as at 19/8/2019 Table number is 20
res = requests.get('https://www.lgsuper.com.au/investments/performance/defined-benefit-scheme/')
soup = BeautifulSoup(res.content, 'lxml')
table = soup.find_all('table')
df_lgs_website_yearly_return_dbs = pd.read_html(str(table))[2]
df_lgs_website_yearly_return_dbs = df_lgs_website_yearly_return_dbs.set_index('Year')

df_lgs_website_yearly_return = pd.concat(
    [
        df_lgs_website_yearly_return_acc,
        df_lgs_website_yearly_return_dbg,
        df_lgs_website_yearly_return_dbs
    ],
    axis=1
)

df_lgs_website_yearly_return = df_lgs_website_yearly_return.sort_index(ascending=False).reset_index(drop=False)
df_lgs_website_yearly_return['Year'] = [int(df_lgs_website_yearly_return['Year'][i]) for i in range(0, len(df_lgs_website_yearly_return))]
df_lgs_website_yearly_return = df_lgs_website_yearly_return.set_index('Year', drop=True)

# Adds the FYTD to Risk of Loss Years
df_lgs_website_return_FYTD = df_lgs_website_return[['Investment Option', 'FYTD']]
strategy_column_list = [df_lgs_website_return_FYTD['Investment Option'][i] + ' (%)' for i in range(0, len(df_lgs_website_return_FYTD))]
df_lgs_website_return_FYTD = df_lgs_website_return_FYTD.drop(columns=['Investment Option'], axis=1)
df_lgs_website_return_FYTD['Investment Option'] = strategy_column_list
df_lgs_website_return_FYTD = df_lgs_website_return_FYTD.set_index('Investment Option')
df_lgs_website_return_FYTD = df_lgs_website_return_FYTD.T
df_lgs_website_return_FYTD = df_lgs_website_return_FYTD.rename(index={'FYTD': 2020})
df_lgs_website_return_FYTD = df_lgs_website_return_FYTD.reset_index(drop=False)
df_lgs_website_return_FYTD = df_lgs_website_return_FYTD.rename(columns={'index': 'Year'})
df_lgs_website_return_FYTD = df_lgs_website_return_FYTD.set_index('Year')

df_lgs_website_yearly_return = pd.concat([df_lgs_website_yearly_return, df_lgs_website_return_FYTD], axis=0).sort_index(ascending=False)
df_lgs_website_yearly_return = df_lgs_website_yearly_return.rename(index={2020: 'FYTD'})
df_lgs_website_yearly_return = df_lgs_website_yearly_return.reset_index(drop=False)

# Create Risk Table and Risk Chart using website calculated data
df_lgs_website_yearly_return = df_lgs_website_yearly_return.set_index('Year')
loss_years = df_lgs_website_yearly_return.lt(0).sum()
gain_years = df_lgs_website_yearly_return.gt(0).sum()

df_loss_years_percentage = pd.DataFrame((loss_years/(loss_years+gain_years))).T.rename(index={0: 'Actual loss years (%)'})
df_loss_years = pd.DataFrame(loss_years).T.rename(index={0: 'No. loss years'})
# df_target_loss_years = pd.DataFrame()
# df_forecast_loss_years = pd.DataFrame()
# df_expected_loss_years = pd.DataFrame((loss_years/(loss_years+gain_years))*20).T.rename(index={0: 'No. Expected loss years per 20 years'})
df_total_years = pd.DataFrame(loss_years+gain_years).T.rename(index={0: 'No. years'})

df_product_risk = pd.concat(
    [
        df_loss_years_percentage,
        df_loss_years,
        # df_target_loss_years,
        # df_forecast_loss_years,
        # df_expected_loss_years,
        df_total_years
    ],
    axis=0
).round(2)
df_lgs_website_yearly_return = df_lgs_website_yearly_return.reset_index(drop=False)

# Outputs the risk of loss years table
with open('U:/CIO/#Data/output/investment/product/product_risk_years.tex', 'w') as tf:
    tf.write(df_lgs_website_yearly_return.to_latex(index=False, na_rep='', multicolumn_format='c', column_format='lRRRRRRR'))


df_check_yearly_returns1 = pd.merge(
    left=df_risk_of_loss_years,
    right=df_lgs_website_yearly_return,
    left_on=['Year'],
    right_on=['Year'],
    how='outer'
)
df_check_yearly_returns1 = df_check_yearly_returns1.reset_index(drop=False)

website_yearly_return_column_to_internal_yearly_return_column_dictionary = {
    'High Growth (%)': 'High Growth',
    'Balanced Growth (%)': 'Balanced Growth',
    'Balanced (%)': 'Balanced',
    'Conservative (%)': 'Conservative',
    'Managed Cash (%)': 'Managed Cash',
    'Defined Benefit Strategy (%)': 'Defined Benefit Strategy'
}

strategy_deviation_yearly_list = []
year_deviation_yearly_list = []
value_deviation_yearly_list = []
for website_column, internal_column in website_yearly_return_column_to_internal_yearly_return_column_dictionary.items():
    deviation_column = internal_column + '_Deviation'
    df_check_yearly_returns1[deviation_column] = df_check_yearly_returns1[internal_column] - df_check_yearly_returns1[website_column]

    for i in range(0, len(df_check_yearly_returns1)):
        if abs(df_check_yearly_returns1[deviation_column][i]) >= 0.01:
            strategy_deviation_yearly_list.append(internal_column)
            year_deviation_yearly_list.append(df_check_yearly_returns1['Year'][i])
            value_deviation_yearly_list.append(df_check_yearly_returns1[deviation_column][i])

df_check_yearly_returns2 = pd.DataFrame()
df_check_yearly_returns2['Strategy'] = strategy_deviation_yearly_list
df_check_yearly_returns2['Year'] = year_deviation_yearly_list
df_check_yearly_returns2['Value'] = value_deviation_yearly_list

writer = pd.ExcelWriter('U:/CIO/#Data/output/investment/product/product_checker.xlsx', engine='xlsxwriter')
df_check_returns1.to_excel(writer, sheet_name='Returns Details', index=False)
df_check_returns2.to_excel(writer, sheet_name='Returns Deviants', index=False)
df_check_returns3.to_excel(writer, sheet_name='Returns Deviants Table 3.1', index=False)
df_check_yearly_returns1.to_excel(writer, sheet_name='Yearly Details', index=False)
df_check_yearly_returns2.to_excel(writer, sheet_name='Yearly Deviants', index=False)
writer.save()




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
