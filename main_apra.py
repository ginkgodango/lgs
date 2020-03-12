import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_apra = pd.read_excel(
    pd.ExcelFile('U:/CIO/#Data/input/apra/Annual Fund-level Superannuation Statistics Back Series June 2019.xlsx'),
    sheet_name='Table 3',
    skiprows=[0, 1, 2, 3, 5, 6, 7]
)

df_apra = df_apra.replace('*', np.nan)

# df_apra['Net assets at beginning of period'] = pd.to_numeric(df_apra['Net assets at beginning of period'])

df_apra = df_apra[(df_apra['Net assets at beginning of period'] >= 5000)].reset_index(drop=True)

descriptive = df_apra.describe()

quantile = df_apra.quantile([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

df_apra_yearly = df_apra.groupby('Period').quantile([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
