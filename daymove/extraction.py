import win32com.client
import dateutil.parser
import pandas as pd
import numpy as np


def load(filename, password, sheet_number=1, start_row=3, start_column='A', end_row=1000, end_column='M'):
    # Opens Excel
    app = win32com.client.Dispatch("Excel.Application")
    print("Excel library version:", app.Version)

    # Opens the Excel file
    wb = app.Workbooks.Open(filename, False, True, None, password)

    # Takes the first worksheet in the workbook
    ws = wb.Sheets(sheet_number)
    print('Accessing: ', wb.Name, ws.Name, '\n')

    # Selects the table from the excel worksheet as content
    content = ws.Range(ws.Cells(start_row, start_column), ws.Cells(end_row, end_column)).Value
    date = ws.Cells(2, 'C').Value
    date = dateutil.parser.parse(str(date)).date()

    # Converts the excel content to dataframe and adds date
    df = pd.DataFrame(list(content))
    df.columns = list(df.iloc[0])
    df = df[1:]
    df = df[(df['FUND'].notnull() | df['Index'].notnull())].reset_index(drop=True)
    df.insert(loc=1, column='Date', value=date)

    # Closes the workbook and then quits Excel
    wb.Close(False)
    app.Quit()

    return df


def clean(df):
    # Replaces None with NaN
    df.fillna(value=pd.np.nan, inplace=True)

    for column_name in df.columns:
        df = df.rename(columns={column_name: column_name.strip()})

    # Strips leading and trailing whitespace from from column values
    strings = ['FUND']
    for column_name in strings:
        df[column_name] = [str(df[column_name][i]).strip() for i in range(0, len(df))]

    # Converts columns into numerics and fills NaN with
    numerics = [
        'Current Value',
        'Prior Value',
        'Capital',
        'Fees',
        'Mgmt Fees Rebates',
        'Back Dated Trades',
        'Net Movement',
        'Return',
        'Mger Weighting per sectorWt',
        'Weighted'
    ]

    for column_name in numerics:
        df[column_name] = pd.to_numeric(df[column_name])

    for column_name in numerics:
        column_values = []
        for i in range(0, len(df)):
            if pd.notnull(df['Current Value'][i]):
                if pd.notnull(df[column_name][i]):
                    column_values.append(df[column_name][i])
                else:
                    column_values.append(0)
            else:
                column_values.append(np.nan)
        df[column_name] = column_values

    return df


def match_manager_to_sector(df):
    sectors = [
        'CASH',
        'INDEXED LINKED BONDS',
        'BONDS',
        'AUSTRALIAN EQUITIES',
        'PROPERTY',
        'GLOBAL PROPERTY',
        'INTERNATIONAL EQUITY',
        'ABSOLUTE RETURN',
        'GREEN SECTOR',
        'ACTIVE COMMODITIES SECTOR',
        'LGSS AE OPTION OVERLAY SECTOR',
        'LGSS LEGACY PE SECTOR',
        "LGSS LEGACY DEF'D PESECTOR",
        'LGS PRIVATE EQUITY SECTOR',
        'LGS OPPORTUNISTIC ALTERNATIVES SECTOR',
        'LGS DEFENSIVE ALTERNATIVES SECTOR',
        'TOTAL LGS (LGSMAN)'
    ]

    manager_sector = []

    for i in range(0, len(df)):
        if df['FUND'][i] in sectors:
            sector = df['FUND'][i]

        manager_sector.append(sector)

    df.insert(loc=0, column='Sector', value=manager_sector)

    return df


def match_manager_to_benchmarks(df):
    ignore = [
        'TOTAL - LGSS AE Option Overlay Sector',
        'TOTAL - LGSS Legacy PE Sector',
        "TOTAL - LGSS DEF'D PE SECTOR",
        'TOTAL - LGS Private Equity Sector',
        'TOTAL - LGS Opportunistic Alternatives Sector',
        'TOTAL LGS (LGSMAN)'
    ]

    benchmarks = {}
    for i in range(0, len(df)):
        if type(df['Index'][i]) == str:
            benchmark_name = df['Index'][i]
            # print(benchmark_name)

        elif df['FUND'][i] in ignore:
            benchmark_value = np.nan
            benchmarks['ignore'] = benchmark_value

        elif pd.notna(df['Index'][i]):
            benchmark_value = df['Index'][i]
            benchmarks[benchmark_name] = benchmark_value

    benchmark_name_column = []
    benchmark_value_column = []
    benchmark_name = np.nan
    benchmark_value = np.nan
    for i in range(0, len(df)):
        if df['Index'][i] in benchmarks:
            # print(df['Index'][i], benchmarks[df['Index'][i]])
            benchmark_name = df['Index'][i]
            benchmark_value = benchmarks[df['Index'][i]]

        if df['FUND'][i] in ignore:
            benchmark_value = benchmarks['ignore']

        if df['FUND'][i] == 'LGSS AE OPTION OVERLAY SECTOR' or df['FUND'][i] == 'TOTAL LGS (LGSMAN)':
            benchmark_name = np.nan
            benchmark_value = np.nan

        benchmark_name_column.append(benchmark_name)
        benchmark_value_column.append(benchmark_value)

    df['Benchmark Name'] = benchmark_name_column
    df['Benchmark'] = benchmark_value_column

    return df


def remove_nan_and_header_funds(df):
    nan_and_header = [
        np.nan,
        'nan',
        'CASH',
        'INDEXED LINKED BONDS',
        'BONDS',
        'AUSTRALIAN EQUITIES',
        'PROPERTY',
        'GLOBAL PROPERTY',
        'INTERNATIONAL EQUITY',
        'ABSOLUTE RETURN',
        'GREEN SECTOR',
        'ACTIVE COMMODITIES SECTOR',
        'LGSS AE OPTION OVERLAY SECTOR',
        'LGSS LEGACY PE SECTOR',
        "LGSS LEGACY DEF'D PESECTOR",
        'LGS PRIVATE EQUITY SECTOR',
        'LGS OPPORTUNISTIC ALTERNATIVES SECTOR',
        'LGS DEFENSIVE ALTERNATIVES SECTOR',
    ]
    df = df[~df['FUND'].isin(nan_and_header)].reset_index(drop=True)

    return df
