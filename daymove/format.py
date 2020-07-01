import numpy as np


def select_rows_today(df):
    date_today = df['Date'].max()
    df = df[df['Date'] == date_today].reset_index(drop=True)
    return df


def select_columns(df):
    df = df[['Sector', 'FUND', 'Benchmark Name', 'Current Value', 'MTD', 'MTD Benchmark', 'MTD Active']]
    df = df.rename(columns={
        'FUND': 'Fund',
        'MTD Benchmark': 'Benchmark',
        'Current Value': 'Market Value',
        'MTD Active': 'Active'
    })
    return df


def format_output(df):
    df['Market Value'] = round(df['Market Value'] / 1000000, 2)
    df['MTD'] = round(df['MTD'] * 100, 2)
    df['Benchmark'] = round(df['Benchmark'] * 100, 2)
    df['Active'] = round(df['Active'] * 100, 2)
    return df


def filter_misc_funds(df):
    misc_filter = [
        'LGCS -  Cash SSC LGCS - UUT',
        'LGCS:  Cash/Other',
        'LGCS:  UUT Sub Total',
        'LGFI - Rebal',
        'LGFI:  Cash/Other',
        'LGFI:  UUT Sub Total',
        'LGAA - Rebalance - UUT',
        'LGAA:  Cash/Other',
        'LGAA:  UUT Sub Total',
        'LGBT:  PENDAL LIQUIDITY MANAGEMENT TRUST  522560U',
        'LGBT:  PENDAL SMALLER COMPANIES TRUST  528808U',
        'LGBT:  Cash/Other',
        'LGBT:  UUT Sub Total',
        'LGUN - Unlisted Property - UUT',
        'LGUN:  Cash/Other',
        'LGUN:  UUT Sub Total',
        'LGPR - Global Property Rebalance - UUT',
        'LGII - REBALANCE - UUT',
        'LGII:  Cash/Other',
        'LGII:  UUT Sub Total',
        'LGMO - LGS IE MESIROW OVERLAY',
        'LGRC - Absolute return - UUT',
        'LGRC:  Cash/Other',
        'LGRC:  UUT Sub Total',
        'LGAT – LGS AUS SUSTAINABLE SHARES REBAL',
        'LGR1:  JPM AUD LVNAV INST MFC37EU',
        'LGR1:  Cash/Other',
        'LGR1:  UUT Sub Total',
        'LGUP:  JPM AUD LVNAV INST MFC37EU',
        'LGUP: JPM AUD LVNAV INST MFC37EU',
        'LGUP:  JJPM USD LIQ LVNAV FD MFC82EU',
        'LGUP: JJPM USD LIQ LVNAV FD MFC82EU',
        'LGUP:  Cash/Other',
        'LGUP:  UUT Sub Total',
        'LGTM - LGS IE TM MACQUARIE 2019'
        'LGRC: SOUTHPEAK REAL CL A MF135EU',
        'LGRC: SOUTHPEAK REAL DIVER MF372EU',
        'LGRC: SOUTHPEAK REAL DIV48 MF388EU',
        'LGRC: SOUTHPEAK REAL DIV48 MF407EU',
        'LGRC: SOUTHPEAK REAL 48V A MF447EU',
        'LGRC: SPK RDF 4 8 A 301117 MF448EU',
        'LGRC: SPK REAL DIV 4 8 CLA MF476EU',
        'LGRC: SOUTHPEAK REAL DIVER MF520EU',
        'LGRC: Attunga Power and Enviro Fund Class E Nov 19 1.1 AA N MFH23EU',
        'LGPA - LIQUID ALTERNATIVES - UUT',
        'LGPA:  Cash / Other',
        'LGPA:  UUT Sub Total',
        'LGPA:  Cash/Other',
        'LGLA:  Cash / Other',
        'LGLA:  UUT Sub Total',
        'LGLA - SHORT TERM FIXED INTEREST - UUT',
        'LGLA:  Cash/Other',
        'LGQP:  Cash / Other',
        'LGQP:  UUT Sub Total',
        'LGQP:  Cash/Other',
        'A'
    ]

    df = df[~df['Fund'].isin(misc_filter)].reset_index(drop=True)
    return df


def rename_funds(df):
    fund_to_name_dict = {
        'LGCS:  QIC CASH ENHANCED FU  570061U': 'QIC Cash',
        'LGQC - LGS CASH RE QIC CREDIT': 'QIC Credit',
        'LGML -  Cash Mutual': 'Mutual Cash',
        'LGCS:  JPM AUD LVNAV INST MFC37EU': 'JPM Cash',
        'TOTAL - Cash': 'Managed Cash',
        'LGFI:  ARDEA WS AU INFLA BD 55675EU': 'Ardea',
        'LGBP - LGS Bonds Pimco': 'PIMCO ESG',
        'LGFW - LGS FI BRANDYWINE': 'Brandywine',
        'LGRA - LGS AFI AMP': 'AMP',
        'LGBM - LGS BONDS MACQUARIE': 'Macquarie',
        'TOTAL - Bonds': 'Bonds',
        'LGAA:  CFS WHOLESALE SMALL 565977U': 'CFS',
        'LGBT - BT': 'Pendal',
        'LGAQ - LGS AUST EQUITIES DNR CAPITAL': 'DNR',
        'LG10 - LGS AUS EQ RE SSGA': 'SSGA',
        'LGBR - LGS AUSTRALIAN EQU BLACKROCK': 'Blackrock',
        'LG16 - LGS AUST EQUITIES RE UBIQUE': 'Ubique',
        'LGR1 - AUS SRI': 'Domestic SRI',
        'LGEC - LGS  AUSTRALIAN EQUITY RE ECP': 'ECP',
        'TOTAL - Australian Equities': 'Australian Equities',
        'LGUN:  LOCAL GOVT PROP FUND  520875U': 'LGS Property',
        'LGUN:  INVESTA PROPERTY GRP  50312EU': 'Investa',
        'LGUN:  GPT WS SC FD NO 1&2  48559EU': 'GPT',
        'LGPM - LGS Property Trust': 'REITS',
        'LGCO - LGS PROPERTY CURRENCY OVERLAY': 'Property Currency Overlay',
        'TOTAL - Property': 'Property',
        'LGGR - GREIT Resolution Capital': 'Resolution',
        'TOTAL - Global Property': 'Global Property',
        'LGSV - LGS IE LSV GLOBAL VALUE': 'LSV',
        'LGMF - LGS INTL EQUITY MFS': 'MFS',
        'LGMX - LGS INTL EQUITY RE IMPAX': 'Impax',
        'LGUP - LGS INTL EQUITY RE UBS SRI': 'International SRI',
        'LGSR - LGS Hermes IE': 'Hermes',
        'LGIP - LGS INTL EQ RE LONGVIEW PRTNRS': 'Longview',
        'LGMO - LGS IE MESIROW OVERLAY': 'Mesirow',
        'LGII:  Wellington Management Portfolios Emerging Markets Equity 572320U': 'Wellington',
        'LGII:  Delaware Investments Emerging Markets Fund 47552EU': 'Macquarie EM',
        'LGII:  MACQUARIE EMERGING MARKETS FUND CLASS I USD MFJ96EU': 'Macquarie EM',
        'LGII:  AQR GLBL ENHAN EQ FD 53062EU': 'AQR',
        'LGOV -LGS NAB CURRENCY OVERLAY': 'NAB Overlay',
        'LGTM - LGS IE TM MACQUARIE 2019': 'Macquarie Transition',
        'LGTW - LGS IE WCM': 'WCM',
        'TOTAL - International Equity': 'International Equities',
        'LGRC:  GAM ABS RETR BOND AU 33114DU': 'GAM',
        'LGRC: Attunga Power and Enviro Fund Main 1.1 AA N 32510EU': 'Attunga',
        'LGRC: CQS CRE MULTI AST FD 58645EU': 'CQS',
        'LGRC:Winton Global Alpha Fund 580055U': 'Winton',
        'LGRC: AQR WS DELTA MP-S1 33119BU': 'AQR Delta',
        'LGRC: SOUTHPEAK REAL CL A MF135EU': 'SouthPeak',
        'LGRC: SOUTHPEAK REAL DIVER MF372EU': 'SouthPeak MF372EU',
        'LGRC: SOUTHPEAK REAL DIV48 MF388EU': 'SouthPeak MF388EU',
        'LGRC: SOUTHPEAK REAL DIV48 MF407EU': 'SouthPeak MF407EU',
        'LGRC: SOUTHPEAK REAL 48V A MF447EU': 'SouthPeak MF447EU',
        'LGRC: SPK RDF 4 8 A 301117 MF448EU': 'SouthPeak MF448EU',
        'LGRC: SPK REAL DIV 4 8 CLA MF476EU': 'SouthPeak MF476EU',
        'LGRC: SOUTHPEAK REAL DIVER MF520EU': 'SouthPeak MF520EU',
        'LGRC: AQR WS DELTA MP2 S2 MF585EU': 'AQR MF585EU',
        'LGRC: AQR DELTA MP2 SR 4 MFE75EU': 'AQR MFE75EU',
        'LGRC: GMO SYS GL MAC TR B MFB06EU': 'GMO',
        'LGMI - Absolute Return re MIML': 'Macquarie',
        'LGKP - LGS Absolute Return re Kapstream': 'Kapstream',
        'TOTAL - Absolute Return': 'Absolute Return',
        'LGPA:Winton Global Alpha Fund 580055U': 'Winton',
        'LGPA: AQR WS DELTA MP-S1 33119BU': 'AQR 1',
        'LGPA: AQR DELTA MP2 SR 4 MFE75EU': 'AQR 2',
        'LGPA: AQR WS DELTA MP2 S2 MF585EU': 'AQR 3',
        'LGPA: GMO SYS GL MAC TR B MFB06EU': 'GMO',
        'TOTAL - LIQUID ALTERNATIVES': 'Liquid Alternatives',
        'LGMI - Absolute Return re MIML': 'Macquarie Credit',
        'LGMI - Short Team FI MIML': 'Macquarie Credit',
        'LGMI - Short Term FI MIML': 'Macquarie Credit',
        'LGMI - Short Term FI Macquarie': 'Macquarie Credit',
        'LGKP - LGS Absolute Return re Kapstream': 'Kapstream',
        'LGKP - LGS Short Term FI Kapstream': 'Kapstream',
        'LGKP - LGS Short Team FI Kapstream': 'Kapstream',
        'LGAU - LGS AR TCW': 'TCW',
        'LGAU - LGS Short Term FI TCW': 'TCW',
        'LGLA:  ARDEA REAL OUTCOME MFF50EU': 'Ardea',
        'LGLA: CQS CRE MULTI AST FD 58645EU': 'CQS',
        'TOTAL - SHORT TERM FIXED INTEREST': 'Short Term Fixed Interest',
        'LGIT – LGS GREEN INTRINSIC INV MGT': 'Intrinsic',
        'TOTAL - Green Sector': 'Green Sector',
        'LGCA - LGS Active Commodities re : H3': 'H3',
        'TOTAL - Active Commodities Sector': 'Commodities',
        'LGCL – LGS OPTION OVERLAY STRATEGY MANAGER': 'AE Options Overlay',
        'TOTAL - LGSS AE Option Overlay Sector': 'Options Overlay',
        'LGER – ER PRIVATE EQUITY': 'Legacy PE',
        'TOTAL - LGSS Legacy PE Sector': 'Legacy Private Equities',
        'LGPS – DEFENSIVE PRIVATE EQUITY': 'Legacy Def PE',
        "TOTAL - LGSS DEF'D PE SECTOR": "Legacy Defensive PE",
        'LGPP - PRIVATE EQUITY': 'PE',
        'TOTAL - LGS Private Equity Sector': 'Private Equities',
        'LGQP - OA REBAL': 'Opportunistic Alts',
        'TOTAL - LGS Opportunistic Alternatives Sector': 'Opportunistic Alternatives',
        'LGDD - DA REBAL': 'Defensive Alts',
        'TOTAL - LGS Defensive Alternatives Sector': 'Defensive Alternatives',
        'TOTAL LGS (LGSMAN)': 'TOTAL LGS',
        'LGRC:  ARDEA REAL OUTCOME MFF50EU': 'Ardea Real',
        'LGRC: Attunga Power and Enviro Fund Class E Nov 19 1.1 AA N MFH23EU': 'Attunga',
        'LGAU - LGS AR TCW': 'TCW',
        'LGQP: Attunga Power and Enviro Fund Main 1.1 AA N 32510EU': 'Attunga OA1',
        'LGQP: Attunga Power and Enviro Fund Class E Nov 19 1.1 AA N MFH23EU': 'Attunga OA2',
        'LGUP: JPM USD LIQ LVNAV FD MFC82EU': 'JPM USD Cash'
    }

    df['Fund'] = [
        fund_to_name_dict[df['Fund'][i]]
        for i in range(0, len(df))
    ]
    return df


def aggregate_southpeak(df):
    southpeak = [
        'SouthPeak',
        'SouthPeak MF372EU',
        'SouthPeak MF388EU',
        'SouthPeak MF407EU',
        'SouthPeak MF447EU',
        'SouthPeak MF448EU',
        'SouthPeak MF476EU',
        'SouthPeak MF520EU'
    ]

    southpeak_market_value = 0
    for i in range(0, len(df)):
        if df['Fund'][i] in southpeak:
            southpeak_market_value += df['Market Value'][i]

    df = df[~df['Fund'].isin(southpeak[1:])].reset_index(drop=True)

    df.loc[df['Fund'] == 'SouthPeak', 'Market Value'] = southpeak_market_value
    return df


def aggregate_aqr(df):
    aqr = [
        'AQR Delta',
        'AQR MF585EU',
        'AQR MFE75EU'
    ]

    aqr_market_value = 0
    for i in range(0, len(df)):
        if df['Fund'][i] in aqr:
            aqr_market_value += df['Market Value'][i]

    df = df[~df['Fund'].isin(aqr[1:])].reset_index(drop=True)

    df.loc[df['Fund'] == 'AQR Delta', 'Market Value'] = aqr_market_value
    return df


def rename_benchmarks(df):
    benchmark_to_name_dict = {
        'ASX Accum Small Cap Ords Index': 'S&P/ASX Accum Small Cap',
        'S&P/ASX Small Ords Accum Index': 'S&P/ASX Accum Small Cap',
        'Aust Govt 10 Year bond yield + 4% ': 'Aus Gov 10 Yr Bond + 4.0%',
        'Bloomberg AusBond Bank Bill Index': 'AusBond Bank Bill',
        'Bloomberg AusBond Bank Bill Index + 1.0%p.a.': 'AusBond Bank Bill + 1.0%',
        'Bloomberg AusBond Infl Govt 0+ Yr Index': 'AusBond Infl Govt 0+ Yr',
        'Bloomberg Ausbond Composite Index': 'AusBond Composite',
        'Bloomberg Commodity Index Australian Dollar Hedged Total Return': 'Bloomberg Commodity',
        'CASH + 1.5% P.A': 'Cash + 1.5%',
        'EPRA/NARETT  (AUD)': 'EPRA/NARETT',
        'MSCI ACWI EX AUS': 'MSCI ACWI ex Aus',
        'S&P/ASX 100 Accum Index ': 'S&P/ASX 100 Accum',
        'S&P 200 PROPERTY': 'S&P 200 Property',
        'S&P 300 ACC INDEX': 'S&P/ASX 300 Accum',
        'S&P/ASX 200 Accumulation Index': 'S&P/ASX 200 Accum',
        'S&P/ASX Accum 100 Index': 'S&P/ASX 100 Accum',
        'UBS BBINDEX 3 MONTH': 'UBS Bank Bill 3 Month',
        'Mercer/IPD Australian Property Pooled Fund Index': 'Mercer/IPD Australian Property',
        'MSCI World Value Ex Australia Net Index': 'MSCI World Value Ex Aus Net',
        'Barclays Capital Global Agg Index (Hedged)': 'Barclays Capital Global Agg',
        'Bloomberg AusBond Composite 0+ Yr Index': 'Ausbond Composite 0+ Yr',
        'S&P/ASX 200 Accum Index': 'S&P/ASX 200 Accum',
        'S&P/ASX 300 Accum Index': 'S&P/ASX 300 Accum',
        'MSCI ACWI ex Australia': 'MSCI ACWI ex Aus',
        'MSCI ACWI ex Australia(Net) 40% Hedged': 'MSCI ACWI ex Aus 40% Hedged',
        'Bloomberg AusBond Bank Bill Index + 2.0%p.a.': 'AusBond Bank Bill + 2.0%',
        'Bloomberg AusBond Bank Bill Index + 1.5%p.a.':  'AusBond Bank Bill + 1.5%',
        'Bloomberg AusBond Bank Bill Index + 0.2%p.a.': 'AusBond Bank Bill + 0.2%',
        'Zero': '',
        'ZERO': 'MSCI ACWI ex Aus',
        np.nan: ''
    }

    df['Benchmark Name'] = [
        benchmark_to_name_dict[df['Benchmark Name'][i]]
        for i in range(0, len(df))
    ]

    return df


def collect_inactive_funds(df):
    df = df[df['Market Value'] == 0].reset_index(drop=True)
    df = df[['Sector', 'Fund', 'Market Value']]

    return df


def filter_inactive_funds(df):
    df = df[df['Market Value'] != 0].reset_index(drop=True)

    return df


def fillna(df):
    df = df.fillna('')
    return df


def collect_sectors(df):
    sectors = [
        'Managed Cash',
        'Bonds',
        'Australian Equities',
        'Property',
        'Global Property',
        'International Equities',
        'Absolute Return',
        'Liquid Alternatives',
        'Short Term Fixed Interest',
        'Commodities',
        'Options Overlay',
        'Legacy Private Equities',
        "Legacy Defensive PE",
        'Private Equities',
        'Opportunistic Alternatives',
        'Defensive Alternatives',
        'TOTAL LGS'
    ]

    df = df[df['Fund'].isin(sectors)].reset_index(drop=True)
    return df


"""
#
# df_result['Fund'] = [df_result['Fund'][i].replace('TOTAL - ', '')
#        for i in range(0,len(df_result))
#        ]

# df_result_sectors['Fund'] = [df_result_sectors['Fund'][i].replace('TOTAL - ', '')
#        for i in range(0,len(df_result_sectors))
#        ]
"""


def reorder_sectors(df):
    df = df.set_index('Sector')

    sector_order = [
        'AUSTRALIAN EQUITIES',
        'INTERNATIONAL EQUITY',
        'GLOBAL PROPERTY',
        'PROPERTY',
        'BONDS',
        'LIQUID ALTERNATIVES',
        'SHORT TERM FIXED INTEREST',
        'CASH',
        'ACTIVE COMMODITIES SECTOR',
        'LGSS AE OPTION OVERLAY SECTOR',
        'LGSS LEGACY PE SECTOR',
        "LGSS LEGACY DEF'D PESECTOR",
        'LGS PRIVATE EQUITY SECTOR',
        'LGS OPPORTUNISTIC ALTERNATIVES SECTOR',
        'LGS DEFENSIVE ALTERNATIVES SECTOR',
        'TOTAL LGS (LGSMAN)'
    ]

    df = df.loc[sector_order].reset_index(drop=True)

    return df


def rename_to_latex_headers(df):
    df.columns = ['Fund', 'Benchmark Name', '{Market Value}', '{MTD}', '{Benchmark}', '{Active}']
    return df


def create_latex_table(df):
    latex_string = (
            df.to_latex(index=False, escape=True)[:len('\\begin{tabular}')]
            + '{\\textwidth}'
            + df.to_latex(index=False)[len('\\begin{tabular}'):]
    )

    latex_string = (
        latex_string
        .replace('tabular', 'tabularx')
        .replace('llrrll', 'p{4cm}p{5cm}R{2.2}R{2.2}R{2.2}R{2.2}')
        .replace('llrlll', 'p{4cm}p{5cm}R{2.2}R{2.2}R{2.2}R{2.2}')
        .replace('llllll', 'p{4cm}p{5cm}R{2.2}R{2.2}R{2.2}R{2.2}')
        .replace('\\{', '{')
        .replace('\\}', '}')
    )
    return latex_string

