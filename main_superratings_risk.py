import pandas as pd
import numpy as np


def read_superratings(s):

    return pd.read_excel(pd.ExcelFile(s))


def colour_green_dark(x):

    return '\cellcolor{CT_green1}(' + str(int(x)) + ')'


def colour_green_light(x):

    return '\cellcolor{CT_green2}(' + str(int(x)) + ')'


def colour_red_light(x):

    return '(' + str(int(x)) + ')'


if __name__ == '__main__':

    file_path = 'C:/Users/mnguyen/LGSS/Investments Team - SandPits - SandPits/data/input/vendors/superratings/2021/03/SR Super Volatility and Risk-Adjusted Return Survey - March 2021.xlsx'

    lgs_fund_list = [
        'Local Government Super Accum - High Growth',
        'Local Government Super Accum - Balanced Growth',
        'Local Government Super Accum - Balanced',
        'Local Government Super Accum - Conservative',
        'Local Government Super Accum - Managed Cash'
    ]

    sr_index_list = [
        'SR50 Growth (77-90) Index',
        'SR50 Balanced (60-76) Index',
        'SR25 Conservative Balanced (41-59) Index',
        'SR50 Capital Stable (20-40) Index',
        'SR50 Cash Index'
    ]

    sr_index_50 = [
        'SR50 Growth (77-90) Index',
        'SR50 Balanced (60-76) Index',
        'SR50 Capital Stable (20-40) Index',
        'SR50 Cash Index'
    ]

    sr_index_25 = [
        'SR25 Conservative Balanced (41-59) Index',
    ]

    comparison_list = [
        'Local Government Super',
        'Aware Super',
        'LGIAsuper',
        'Vision SS',
        'Not for Profit Fund Median'
    ]

    comparison_list1 = [
        'LGIAsuper Accum - Aggressive',
        'Local Government Super Accum - High Growth',
        'Vision SS - Growth',
        'Aware Super (previously First State Super) - Growth',
        'Aware Super - Growth',
        'LGIAsuper Accum - Diversified Growth',
        'Local Government Super Accum - Balanced Growth',
        'Vision SS - Balanced Growth',
        'Aware Super (previously First State Super) - Balanced Growth',
        'Aware Super - Balanced Growth',
        'LGIAsuper Accum - Balanced',
        'Local Government Super Accum - Balanced',
        'Vision SS - Balanced',
        'Aware Super (previously First State Super) - Conservative Growth',
        'Aware Super - Conservative Growth',
        'LGIAsuper Accum - Stable',
        'Local Government Super Accum - Conservative',
        'Vision SS - Conservative',
        'Aware Super (previously First State Super) Tailored Super Plan - Cash Fund',
        'Aware Super Tailored Super Plan - Cash Fund',
        'LGIAsuper Accum - Cash',
        'Local Government Super Accum - Managed Cash',
        'Vision SS - Cash',
        'Not for Profit Fund Median',
    ]

    column_dict = {
        'Fund': 'Fund',
        'SR Index': 'SR Index',
        'Size $Mill': '$Mill',
        'Size Rank': 'Size Rank',
        'Sharpe Ratio 1 Year %': '1 Year %',
        'Sharpe Ratio 1 Year Rank': '1 Year Rank',
        'Sharpe Ratio 3 Year %': '3 Year %',
        'Sharpe Ratio 3 Year Rank': '3 Year Rank',
        'Sharpe Ratio 5 Year %': '5 Year %',
        'Sharpe Ratio 5 Year Rank': '5 Year Rank',
        'Sharpe Ratio 7 Year %': '7 Year %',
        'Sharpe Ratio 7 Year Rank' : '7 Year Rank',
        'Sharpe Ratio 10 Year %': '10 Year %',
        'Sharpe Ratio 10 Year Rank': '10 Year Rank'
    }

    column_rank_list = [
        'Size Rank',
        'Sharpe Ratio 1 Year Rank',
        'Sharpe Ratio 3 Year Rank',
        'Sharpe Ratio 5 Year Rank',
        'Sharpe Ratio 7 Year Rank',
        'Sharpe Ratio 10 Year Rank',
    ]

    short_name_dict = {
        'Aware Super': 'Aware',
        'Aware Super Tailored Super Plan': 'Aware',
        'LGIAsuper': 'LGIAsuper',
        'Local Government Super': 'LGS',
        'Vision SS': 'Vision',
        'Not for Profit Fund Median': 'NFP Median'
    }

    df_0 = read_superratings(file_path)

    print("Reporting date: ", max(df_0['Date']).date())

    df_0['Fund'] = [(str(x).split(' - '))[0] for x in df_0['Option Name']]
    df_0['Fund'] = [(str(x).split(' ('))[0] for x in df_0['Fund']]
    df_0['Fund'] = [(str(x).split(' Accum'))[0] for x in df_0['Fund']]

    # for column_rank in column_rank_list:
    #
    #     df_0[column_rank] = ['(' + str(int(x)) + ')' if pd.notna(x) else np.nan for x in df_0[column_rank]]

    df_1 = df_0[df_0['SR Index'].isin(sr_index_list)].reset_index(drop=True)

    # df_2 = df_1[df_1['Fund'].isin(comparison_list)].reset_index(drop=True)

    df_1_a = df_1[df_1['Option Name'].isin(lgs_fund_list)]
    df_1_b = df_1[~df_1['Option Name'].isin(lgs_fund_list)]

    df_1_a = df_1_a.reset_index(drop=False)
    df_1_b = df_1_b.reset_index(drop=False)

    df_1_a_25 = df_1_a[df_1_a['SR Index'].isin(sr_index_25)]
    df_1_a_50 = df_1_a[df_1_a['SR Index'].isin(sr_index_50)]

    for column_rank in column_rank_list:
        df_1_a_25[column_rank] = [
            colour_green_dark(x) if x != '-' and int(x) <= 6 else
            colour_green_light(x) if x != '-' and int(x) <= 9 else
            colour_red_light(x) if x != '-' else
            np.nan
            for x in df_1_a_25[column_rank]
        ]

        df_1_a_50[column_rank] = [
            colour_green_dark(x) if x != '-' and int(x) <= 13 else
            colour_green_light(x) if x != '-' and int(x) <= 19 else
            colour_red_light(x) if x != '-' else
            np.nan
            for x in df_1_a_50[column_rank]
        ]

        df_1_b[column_rank] = ['(' + str(int(x)) + ')' if pd.notna(x) else np.nan for x in df_1_b[column_rank]]

    df_1_a_colour = pd.concat([df_1_a_25, df_1_a_50]).sort_values(['index'])

    df_1 = pd.concat([df_1_a_colour, df_1_b]).sort_values(['index']).drop(columns=['index'], axis=1)


    df_2 = df_1[df_1['Option Name'].isin(comparison_list1)].reset_index(drop=True)

    df_3 = df_2[column_dict]

    df_4 = df_3.rename(columns=column_dict)

    df_4['Fund'] = [short_name_dict[x] for x in df_4['Fund']]

    sr_index_to_df = dict(list(df_4.groupby(['SR Index'])))

    for sr_index, df_temp0 in sr_index_to_df.items():

        #df_temp1 = df_temp0.drop(columns=['SR Index'], axis=1)

        df_temp1 = df_temp0[['Fund']]
        df_temp2 = df_temp0[['$Mill', 'Size Rank']]
        df_temp3 = df_temp0[[
            '1 Year %',
            '1 Year Rank',
            '3 Year %',
            '3 Year Rank',
            '5 Year %',
            '5 Year Rank',
            '7 Year %',
            '7 Year Rank',
            '10 Year %',
            '10 Year Rank'
        ]]

        columns_temp_multilevel1 = pd.MultiIndex.from_product([[''], ['Fund']])
        columns_temp_multilevel2 = pd.MultiIndex.from_product([['Market Value'], ['$Mills', 'Rank']])
        columns_temp_multilevel3 = pd.MultiIndex.from_product([['1 Year', '3 Year', '5 Year', '7 Year', '10 Year'], ['SR', 'Rank']])

        df_temp1.columns = columns_temp_multilevel1
        df_temp2.columns = columns_temp_multilevel2
        df_temp3.columns = columns_temp_multilevel3

        df_temp4 = pd.concat([df_temp1, df_temp3], axis=1)

        with open('C:/Users/mnguyen/LGSS/Investments Team - SandPits - SandPits/data/output/lgs/reports/superratings/risk/' + sr_index + '.tex', 'w') as tf:

            tf.write(df_temp4.to_latex(index=False, na_rep='', multicolumn_format='c', escape=False, float_format="{:0.2f}".format))