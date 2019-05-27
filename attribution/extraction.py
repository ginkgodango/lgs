import pandas as pd
import win32com.client


def load_returns(filepath):
    df = pd.read_csv(
        filepath,
        index_col='Date',
        parse_dates=['Date'],
        infer_datetime_format=True,
        float_precision='round_trip'
    )
    return df


def load_market_values(filepath):
    df = pd.read_csv(
        filepath,
        index_col='Date',
        parse_dates=['Date'],
        infer_datetime_format=True,
        float_precision='round_trip'
    )

    return df


def load_asset_allocation(filepath, sheet_number, start_cell, end_cell):
    start_column = start_cell.split(':')[0]
    start_row = start_cell.split(':')[1]
    end_column = end_cell.split(':')[0]
    end_row = end_cell.split(':')[1]

    # Opens Excel
    app = win32com.client.Dispatch("Excel.Application")
    # print("Excel library version:", app.Version)

    # Opens the Excel file
    wb = app.Workbooks.Open(filepath, False, True, None)

    # Takes the first worksheet in the workbook
    ws = wb.Sheets(sheet_number)
    print('Accessing: ', wb.Name, ws.Name, '\n')

    # Selects the table from the excel worksheet as content
    content = ws.Range(ws.Cells(start_row, start_column), ws.Cells(end_row, end_column)).Value

    # Converts the excel content to dataframe and adds date
    df = pd.DataFrame(list(content))
    df.columns = list(df.iloc[0])
    df = df[1:]

    df.insert(loc=0, column='Strategy', value=df.columns[0])

    df = df.rename(
        columns={
            'High Growth': 'Asset Class',
            'Balanced Growth': 'Asset Class',
            'Balanced': 'Asset Class',
            'Conservative': 'Asset Class',
            'Growth': 'Asset Class',
            'Employer Reserve': 'Asset Class'
        }
    )

    # Closes the workbook and then quits Excel
    wb.Close(False)
    app.Quit()

    return df
