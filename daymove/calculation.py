def calculate_returns(df):
    # Calculates Net Cash Flow, Midday Return, and End of Day Return
    df['Net Cash Flow'] = df['Capital'] + df['Fees'] + df['Mgmt Fees Rebates'] + df['Back Dated Trades']

    # Special case of modified dietz return
    df['Midday Return'] = (
            (df['Current Value'] - df['Prior Value'] - df['Net Cash Flow']) /
            (df['Prior Value'] + (0.5 * df['Net Cash Flow']))
    )

    df['End of Day Return'] = (
            (df['Current Value'] - df['Prior Value'] - df['Net Cash Flow']) /
            (df['Prior Value'])
    )

    return df


def calculate_active_returns(df):
    df['Active Return'] = df['End of Day Return'] - df['Benchmark']

    return df


def calculate_mtd_returns(df):
    df['MTD'] = df.groupby(['FUND'])['End of Day Return'].apply(lambda r: ((1+r).cumprod()-1))
    df['MTD Benchmark'] = df.groupby(['FUND'])['Benchmark'].apply(lambda r: ((1+r).cumprod()-1))
    df['MTD Active'] = df['MTD'] - df['MTD Benchmark']

    return df

