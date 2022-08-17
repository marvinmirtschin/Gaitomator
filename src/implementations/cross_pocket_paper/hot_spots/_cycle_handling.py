import numpy as np
import pandas as pd

from src.core.base_classes import SafeTransformer


class AverageCycleCreator(SafeTransformer):

    def __init__(self, averaging_method='mean', include_std_cycle=True, **kwargs):
        self.averaging_method = averaging_method
        self.include_std_cycle = include_std_cycle
        SafeTransformer.__init__(self, **kwargs)

    def _transform(self, data):
        if self.averaging_method == 'mean':
            average_cycle = data.apply(np.mean)
        elif self.averaging_method == 'median':
            average_cycle = data.apply(np.median)
        else:
            raise Exception("Unknown averaging method '{}'. Either use one of the following or implement missing once: ['mean', 'median']")

        average_cycle.index = [f"avg_{column}" for column in average_cycle.index]

        if self.include_std_cycle:
            standard_deviation_cycle = data.apply(np.std)
            standard_deviation_cycle.index = [f"std_{column}" for column in standard_deviation_cycle.index]
            # return pd.DataFrame([average_cycle, standard_deviation_cycle])
            return pd.concat([average_cycle, standard_deviation_cycle], axis=0).to_frame().transpose()
        else:
            return average_cycle.to_frame().transpose()


def calculate_distance(series_1, series_2):
    """
    Calculate distance between to segments of equal length.

    Parameters
    ----------
    series_1 : pd.Series
        First segment used for distance calculation.
    series_2
        Second segment used for distance calculation.
    Returns
    -------
        Distance between the two series.
    """
    return pd.Series(series_1.values - series_2.values).abs().sum()
