import numpy as np
import pandas as pd
import pywt
from scipy import signal

from src.core.base_classes import SafeTransformer

DEFAULT_FILTER_WINDOW = 5
DEFAULT_DEGREE_OF_POLYNOMIAL = 2


class MultiLevelWaveletFilter(SafeTransformer):
    """
    High-Pass filter using multilevel wavelets. The default values are taken from 'On the Instability of Sensor
    Orientation in Gait Verification on Mobile Phone' by Hoang et al.

    Parameters
        ----------
        wavelet_name : Wavelet object or name string, default='db6'
            Wavelet to use
        mode : str, default='periodic'
            # Note: not mentioned in paper, default of library is symmetric
            Signal extension mode, see `pywt.Modes.modes`.
        level : int, default=2
            Decomposition level (must be >= 0). If level is None (default) then it will be calculated using the ``pywt.dwt_max_level`` function.
    Returns
        ----------
         Filtered data frame.
    """

    def __init__(self, wavelet_name='db6', mode='periodic', level=2, **kwargs):
        self.wavelet_name = wavelet_name
        self.mode = mode
        self.level = level
        super().__init__(**kwargs)

    def _transform(self, data):
        filtered_values = list()
        for column in data:
            filtered_values.append(self.apply_wavelet_filter(data[column]))
        return pd.concat(filtered_values, axis=1)

    def apply_wavelet_filter(self, data_series: pd.Series):
        # coeffs are an array with the coarse coefficients from level self.level at [0] and
        # detailed coefficients in reversed level-order from then on.
        coeffs = pywt.wavedec(data=data_series.values, wavelet=self.wavelet_name, mode=self.mode, level=self.level)

        for level in range(1, self.level + 1):
            # set the detailed coefficients to 0 for each level
            coeffs[-level] = np.zeros_like(coeffs[-level])

        filtered_values = pywt.waverec(coeffs, self.wavelet_name)

        if len(filtered_values) != len(data_series):
            # TODO: this needs to be further investigated whether it is expected behaviour or a bug
            filtered_values = filtered_values[:len(data_series)]

        return pd.Series(filtered_values, index=data_series.index, name=data_series.name)


def apply_weighted_moving_average_according_to_bours_2018(pandas_series, **kwargs):
    """
    According to paper Bours, Denzer (2018): Cross-Pocket Gait Recognition, using 5 window moving average with weight of [1,2,3,2,1]:
    Ri = ri−2 + 2∗ri−1 + 3∗ri + 2∗ri + 1 + ri + 2

    :param pandas_series: array_like, one-dimensional ndarray with axis labels (including time series); e.g.
    data of one column from pandas data frame
    :return: weighted moved average sequence
    """
    wma_sequence = []
    for row_index, value in enumerate(pandas_series):
        if not any([row_index < 2, row_index > len(pandas_series) - 3]):
            wma_sequence.append(
                (pandas_series[row_index - 2] + 2 * pandas_series[row_index - 1] + 3 * pandas_series[row_index] + 2 *
                 pandas_series[row_index + 1] + pandas_series[
                     row_index + 2]) / 9)  # divide by 9, not 5 because of (1+2+3+2+1) # paper: Holien 2008

    return pd.Series(wma_sequence, index=pandas_series.index[2:len(pandas_series) - 2])


def apply_savitzky_golay_filter(pandas_series, **kwargs):
    """
    Applies the Savitzky-Golay filter.

    :param pandas_series: array_like, one-dimensional ndarray with axis labels (including time series); e.g.
    data of one column from pandas data frame
    :return: filtered data sequence
    """

    filter_window_length = kwargs.get("filter_window_length", DEFAULT_FILTER_WINDOW)
    degree_of_polynomial = kwargs.get("degree_of_polynomial", DEFAULT_DEGREE_OF_POLYNOMIAL)

    if degree_of_polynomial > filter_window_length:
        raise Exception("The degree of polynomial must be smaller than filter's window length")
    return signal.savgol_filter(x=pandas_series, window_length=filter_window_length, polyorder=degree_of_polynomial,
                                mode='nearest')


FILTER_MAPPING = {
    "weighted_moving_average": apply_weighted_moving_average_according_to_bours_2018,
    "savitzky_golay"         : apply_savitzky_golay_filter
}
