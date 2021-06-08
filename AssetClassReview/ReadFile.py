import pandas as pd
import numpy as np


class ReadFile:

    def __init__(selfs):
        self.file = " "

    def jpm_wide_to_long(self, df, set_date_name, set_index_name, set_values_name):
        """

        :param df:
        :param set_date_name:
        :param set_index_name:
        :param set_values_name:
        :return:
        """
        return (
            pd.melt(
            (
                df
                .replace('-', np.NaN)
                .rename(columns={'Unnamed: 0': set_date_name})
                .set_index('Date')
                .transpose()
                .reset_index(drop=False)
                .rename(columns={'index': set_index_name})
            ),
            id_vars = [set_index_name],
            value_name=set_values_name)
            .sort_values([set_index_name, set_date_name])
            .reset_index(drop=True)
        )
