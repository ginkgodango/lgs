import pandas as pd
import numpy as np
import datetime as dt
import matplotlib
import matplotlib.pyplot as plt


def read_up(path, date_column):

    return pd.read_csv(path, parse_dates=[date_column])


def process_up(df, reporting_date):

    df = df.rename(
        columns={
            'InvestmentOption.InvestmentComponent.Name': 'Component',
            'InvestmentOption.InvestmentOptionType.Name': 'OptionType'
        }
    )

    df = (
        df[
            (df['Component'].isin(['Accumulation Scheme'])) &
            (df['OptionType'].isin(['High Growth', 'Balanced Growth', 'Balanced', 'Conservative'])) &
            (df['Date'] <= reporting_date) &
            (df['Unit Price'] != 0)
        ]
        .drop(columns=['Component'], axis=1)
        .sort_values(['OptionType', 'Date'])
        .reset_index(drop=True)
        .set_index('Date')
        .groupby('OptionType')
        .resample('M')
        .pad()
        .drop(columns='OptionType', axis=1)
        .reset_index(drop=False)
    )

    df['Unit Price Lag 1'] = df.groupby('OptionType')['Unit Price'].shift(1)
    df['Return'] = (df['Unit Price'] - df['Unit Price Lag 1']) / df['Unit Price Lag 1']
    df = df.sort_values(['OptionType', 'Date']).reset_index(drop=True)

    columns = ['OptionType', 'Date', 'Unit Price', 'Return']

    return df[columns]


def read_ap(path, sheet):

    return pd.read_excel(pd.ExcelFile(path), sheet_name=sheet)


def process_ap(df, date_column):

    df = df.rename(columns={'Strategy': 'OptionType'})

    df[date_column] = [
        df[date_column][i] - pd.Timedelta(days=1) if (df[date_column][i].month == 2 and df[date_column][i].day == 29) else
        df[date_column][i]
        for i in range(len(df))
    ]

    columns = ['OptionType', 'Date', 'AA Version', 'BN Version', 'APRA']

    return df[columns]


def read_lc(path, sheet):

    return pd.read_excel(pd.ExcelFile(lc_path), sheet_name='Sheet2')


def process_lc(df):

    df = (df.fillna(0).set_index(['Lifecycle', 'OptionType']).T.unstack().reset_index(drop=False).rename(columns={'level_2': 'Age', 0: 'Weight'}))

    df['Weight'] = df['Weight'] / 100

    return df


darkgreen = (75/256, 120/256, 56/256)
middlegreen = (141/256, 177/256, 66/256)
lightgreen = (175/256, 215/256, 145/256)


if __name__ == '__main__':

    report_date = dt.datetime(2020, 9, 30)
    up_path = 'C:/Users/Mnguyen/LGSS/Investments Team - SandPits - SandPits/data/input/lgs/unitprices/20200930 Unit Prices.csv'
    ap_path = 'C:/Users/Mnguyen/LGSS/Investments Team - SandPits - SandPits/Email Messages/APRA_benchmark.xlsx'
    lc_path = 'C:/Users/Mnguyen/LGSS/Investments Team - SandPits - SandPits/data/input/vendors/jana/lifecycles/20200131 Lifecycles.xlsx'

    df_up = process_up(read_up(up_path, 'Date'), report_date)
    df_ap = process_ap(read_ap(ap_path, 'History'), 'Date')
    df_lc = process_lc(read_lc(lc_path, 'Sheet2'))

    df_merge1 = pd.merge(left=df_up, right=df_ap, on=['OptionType', 'Date'], how='inner')
    df_merge2 = pd.merge(left=df_lc, right=df_merge1, on=['OptionType'], how='inner')

    df_merge2['wR_p'] = df_merge2['Weight'] * df_merge2['Return']
    df_merge2['wR_b'] = df_merge2['Weight'] * df_merge2['APRA']

    df_groupby1 = df_merge2.groupby(['Age', 'Lifecycle', 'AA Version', 'BN Version', 'Date'])[['wR_p', 'wR_b']].sum().reset_index(drop=False)

    df_groupby1['wR_p_12'] = (
        df_groupby1
        .groupby(['Age', 'Lifecycle', 'AA Version', 'BN Version'])['wR_p']
        .rolling(12)
        .apply(lambda r: (np.prod(1+r))-1, raw=False)
        .reset_index(drop=False)['wR_p']
    )

    df_groupby1['wR_b_12'] = (
        df_groupby1
        .groupby(['Age', 'Lifecycle', 'AA Version', 'BN Version'])['wR_b']
        .rolling(12)
        .apply(lambda r: (np.prod(1 + r)) - 1, raw=False)
        .reset_index(drop=False)['wR_b']
    )

    df_groupby1['Year'] = [df_groupby1['Date'][i].year for i in range(len(df_groupby1))]
    df_groupby1['Month'] = [df_groupby1['Date'][i].month for i in range(len(df_groupby1))]


    # simulate_columns = [
    #     'Lifecycle',
    #     'Date',
    #     'Year',
    #     'Month',
    #     'Age',
    #     '12_Weighted_Return',
    #     '12_Weighted_Objective'
    # ]
    #
    # df_simulate = df_age_final[simulate_columns]
    # df_simulate = df_simulate.sort_values(['Lifecycle', 'Month', 'Year', 'Age']).reset_index(drop=True)
    #
    # lifespan = 10
    # weighted_return_lag_columns = list()
    # weighted_objective_lag_columns = list()
    # for year in range(0, lifespan):
    #     date_forward_column = 'Date Forward ' + str(year)
    #     weighted_return_lag_column = '12_Weighted_Return Lag ' + str(year)
    #     weighted_objective_lag_column = '12_Weighted_Objective Lag ' + str(year)
    #     weighted_return_lag_columns.append(weighted_return_lag_column)
    #     weighted_objective_lag_columns.append(weighted_objective_lag_column)
    #
    #     df_simulate_temp = df_simulate[['Lifecycle', 'Age', 'Date', '12_Weighted_Return', '12_Weighted_Objective']]
    #     if year == 0:
    #         df_simulate_temp[date_forward_column] = df_simulate_temp['Date']
    #     else:
    #         df_simulate_temp[date_forward_column] = df_simulate_temp['Date'].apply(lambda date: date + pd.DateOffset(years=year))
    #
    #     df_simulate_temp = df_simulate_temp.drop(columns=['Date'], axis=1)
    #     df_simulate_temp = df_simulate_temp.rename(
    #         columns={
    #             '12_Weighted_Return': weighted_return_lag_column,
    #             '12_Weighted_Objective': weighted_objective_lag_column
    #         }
    #     )
    #
    #     df_simulate = pd.merge(
    #         left=df_simulate,
    #         right=df_simulate_temp,
    #         left_on=['Lifecycle', 'Age', 'Date'],
    #         right_on=['Lifecycle', 'Age', date_forward_column],
    #         how='left'
    #     )
    #
    #     df_simulate[weighted_return_lag_column] = df_simulate.groupby(['Lifecycle', 'Date'])[weighted_return_lag_column].shift(year)
    #     df_simulate[weighted_objective_lag_column] = df_simulate.groupby(['Lifecycle', 'Date'])[weighted_objective_lag_column].shift(year)
    #
    #     df_simulate = df_simulate.drop(columns=[date_forward_column], axis=1)
    #
    # # Converts df_simulate into panel
    # df_simulate = df_simulate.sort_values(['Lifecycle', 'Date', 'Age'])
    # df_simulate = df_simulate.drop(columns=['Year', 'Month', '12_Weighted_Return', '12_Weighted_Objective'], axis=1)
    # df_simulate = pd.wide_to_long(df_simulate, stubnames=['12_Weighted_Return Lag ', '12_Weighted_Objective Lag '], i=['Lifecycle', 'Date', 'Age'], j='Lag')
    # df_simulate = df_simulate.reset_index(drop=False)
    # df_simulate = df_simulate.sort_values(['Lifecycle', 'Date', 'Age', 'Lag'], ascending=[True, True, True, False]).reset_index(drop=True)
    # df_simulate = df_simulate.rename(columns={'12_Weighted_Return Lag ': '12_Weighted_Return', '12_Weighted_Objective Lag ': '12_Weighted_Objective'})
    #
    # for return_type in ['12_Weighted_Return', '12_Weighted_Objective']:
    #     column_name = str(lifespan*12) + '_Lifecycle_' + return_type[12:]
    #     df_simulate[column_name] = (
    #         df_simulate
    #             .groupby(['Lifecycle', 'Date', 'Age'])[return_type]
    #             .rolling(lifespan)
    #             .apply(lambda r: (np.prod(1 + r) ** (1 /lifespan)) - 1, raw=False)
    #             .reset_index(drop=False)[return_type]
    #     )
    #
    # df_simulate['Age Lag'] = df_simulate['Age'] - df_simulate['Lag']
    #
    # df_simulate.to_csv('U:/CIO/#Data/output/investment/mysuper/lifecycle_simulation_5years_panel.csv', index=False)
    #
    # df_simulate_chart_bar_summary = df_simulate[(df_simulate['Date'] == report_date) & (df_simulate['Lag'] == 0)]
    # df_simulate_chart_bar_summary = df_simulate_chart_bar_summary[df_simulate_chart_bar_summary['Age'].isin([50, 55, 60, 65])]
    # df_simulate_chart_bar_summary = ((df_simulate_chart_bar_summary[['Age', '120_Lifecycle_Return', '120_Lifecycle_Objective']].set_index('Age'))*100).round(2)
    # df_simulate_chart_bar_summary = df_simulate_chart_bar_summary.rename(columns={'120_Lifecycle_Return': 'Return', '120_Lifecycle_Objective': 'Objective'})
    # fig_simulate_chart_bar_summary = df_simulate_chart_bar_summary.plot(kind='bar', color=[darkgreen, lightgreen])
    # # fig_simulate_chart_bar_summary.set_title('5 Year Return for Each Age Cohort')
    # fig_simulate_chart_bar_summary.set_ylabel('Return (%)')
    # fig_simulate_chart_bar_summary.set_xlabel('Age Cohort (Year)')
    # plt.tight_layout()
    # fig_bar = fig_simulate_chart_bar_summary.get_figure()
    # fig_bar.savefig('U:/CIO/#Data/output/investment/mysuper/monitor.PNG', dpi=300)
    #
    # df_simulate_chart_cross_section = df_simulate[df_simulate['Date'] == report_date]
    # df_simulate_chart_cross_section = df_simulate_chart_cross_section.drop(columns=['Date', 'Lag', '120_Lifecycle_Return', '120_Lifecycle_Objective'], axis=1)
    # df_simulate_chart_cross_section = df_simulate_chart_cross_section[df_simulate_chart_cross_section['Age'].isin([50, 55, 60, 65])]
    #
    # lifecycle_to_cross_section_dict = dict(list(df_simulate_chart_cross_section.groupby(['Lifecycle'])))
    # for lifecycle, df_cross_section in lifecycle_to_cross_section_dict.items():
    #     df_cross_section = df_cross_section.drop(columns=['Lifecycle'], axis=1)
    #     age_to_cross_section2_dict = dict(list(df_cross_section.groupby(['Age'])))
    #
    #     fig, axes = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=True, figsize=(12.8, 7.2))
    #     i = 0
    #     j = 0
    #     for age, df_cross_section2 in age_to_cross_section2_dict.items():
    #         df_cross_section2 = df_cross_section2.drop(columns=['Age'], axis=1)
    #         df_cross_section2 = df_cross_section2.set_index('Age Lag')
    #         df_cross_section2 = df_cross_section2.rename(columns={'12_Weighted_Return': 'Return', '12_Weighted_Objective': 'Objective'})
    #         df_cross_section2 = (df_cross_section2 * 100).round(2)
    #         df_cross_section2.plot(ax=axes[i, j], kind='bar', color=[darkgreen, lightgreen])
    #         axes[i, j].set_title('1 Year Return at Each Age for a ' + str(age) + ' Year Old')
    #         axes[i, j].set_ylabel('Return (%)')
    #         axes[i, j].set_xlabel('Age (Years)')
    #         axes[i, j].legend(loc='upper left', title='')
    #         if i == 0 and j == 0:
    #             j += 1
    #         elif i == 0 and j == 1:
    #             i += 1
    #             j -= 1
    #         elif i == 1 and j == 0:
    #             j += 1
    #
    #     fig.suptitle(lifecycle)
    #     fig.tight_layout()
    #     fig.subplots_adjust(top=0.9)
    #     fig.savefig('U:/CIO/#Data/output/investment/mysuper/' + str(lifecycle) + '.png', dpi=300)

