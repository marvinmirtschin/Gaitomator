import numpy as np
import pandas as pd

from src.core.base_classes import SafeTransformer


class CycleCleaner(SafeTransformer):
    """
    Cleans cycles from data frame which deviate too much from the mean. Note that the cycles need to be interpolated.

    Parameters
    ----------
    number_of_accepted_outlier_values : int, default=10
        Number of values with needs to be out of range of the standard deviation from the mean before being declared an outlier.
    number_of_accepted_stds : float, default=2
        Number of standard deviations a value needs to be away from the mean before being declared an outlier.

    Returns
    -------
        Cleaned cycles
    """

    def __init__(self, number_of_accepted_outlier_values=10, number_of_accepted_stds=2, **kwargs):
        self.number_of_accepted_outlier_values = number_of_accepted_outlier_values
        self.number_of_accepted_stds = number_of_accepted_stds
        SafeTransformer.__init__(self, **kwargs)

    def _transform(self, data):
        mean = data.apply(np.mean)
        std = data.apply(np.std)
        cleaned_data_frame = data.apply(make_outlier_rows_to_nan, means=mean, stds=std,
                                        number_of_accepted_outlier_values=self.number_of_accepted_outlier_values,
                                        number_of_accepted_stds=self.number_of_accepted_stds, axis=1)
        cleaned_data_frame = cleaned_data_frame.dropna(axis=0, how='all')
        return cleaned_data_frame


def make_outlier_rows_to_nan(data_series, means, stds, number_of_accepted_outlier_values=10, number_of_accepted_stds=2):
    centered_series = pd.Series(data_series.values - means.values).abs()
    counter = 0
    assert len(centered_series), len(stds)
    for value, std in zip(centered_series.values, stds):
        if value > number_of_accepted_outlier_values * std:
            counter += 1
            if counter > number_of_accepted_stds:
                return pd.Series(np.nan, index=data_series.index)
    return data_series
