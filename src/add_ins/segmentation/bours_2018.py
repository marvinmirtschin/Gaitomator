# The paper "Cross Pocket Recognition" by Bours et al. is heavily copied from "Spoof attacks on gait authentication system" (especially the
# segmentation). Here is an excerpt from the paper:
#
# Regarding the cycle detection:
#
# •  Cycle detection: The natural cadence of the human walking is in the range steps/min [27] (i.e., about cycles per minute) and the sampling
# frequency of an accelerometer sensor is about 100 observations/s. With the aid of such clues, we perform cycle detection. Let R = (r_1, ... ,
# r_k) be a resulting acceleration signal, and [r_m_1, ... , r_m_L] be the local minima in this signal, which needs to be found. Every cycle
# contains about 100 observations. A first minimum observation is found from the first 250 observations in the signal (i.e., r_m_1 = min(r_1, r_2,
# ..., r_250)). This minimum observation represents the beginning of the first cycle. Then, the end of the cycle is found as follows r_m_2 = min(
# r_m_1 + M - d, ..., r_m_1 + M + d), where M = 100 and d = 20. The end of one cycle is considered the start of the next cycle. The procedure is
# repeated until the last cycle is found.
# • Time normalization: Usually, the number of observations in each cycle will not be constant. Therefore, in order to calculate an average cycle
# of the person, every cycle is normalized in time (by interpolation). Each normalized gait cycle contains exactly 100 observations.
# • Averaged cycle: After cycles are found and normalized, an average cycle A = (a_1, ..., a_n) is calculated as follows:
#         a_i = median(w_ji), where i = 1, ..., n; n = 100 and w_ji is the observation value at observation number in the normalized cycle,
#         is the number of detected cycles in the signal. In other words, each observation in the averaged cycle is the median of the
#         corresponding observations in the normalized cycle
# The motivation for selecting the median rule for averaging is to reduce the influence of very unusual steps (i.e., cycles). => this is changed in
# the newer paper

import pandas as pd

from src.add_ins.segmentation import split_data_for_indices
from src.core.base_classes import SafeTransformer


class CycleDetectionTransformer(SafeTransformer):

    def __init__(self, cut_at_dimension, frequency=100., neighborhood_factor=0.2, **kwargs):
        self.cut_at_dimension = cut_at_dimension
        self.frequency = frequency
        self.neighborhood_factor = neighborhood_factor
        super().__init__(**kwargs)

    def _transform(self, data):
        return get_cycles(data, cut_at_dimension=self.cut_at_dimension, frequency=self.frequency, neighborhood_factor=self.neighborhood_factor)


def get_cycles(data, cut_at_dimension, frequency=100., neighborhood_factor=0.2):
    """
    Parameters
    ----------
    data : pd.DataFrame
        Data to be cut into segments.
    cut_at_dimension : str,
        Name of the column which should be used for peak detection to find cycle start indices.
    frequency : float, default=100
        Frequency of the input data. Based on this the search ranges are calculated which are taken from research.
    neighborhood_factor : float, default=0.2
        Percentage from the interval (0, 0.5) for how many values should be included in the search for the next minimum.

    Returns
    -------
        Data split into cycles.
    """
    indices = _get_cycle_start_indices(data, cut_at_dimension=cut_at_dimension, frequency=frequency, neighborhood_factor=neighborhood_factor)
    return split_data_for_indices(data, indices)


def _get_cycle_start_indices(data, cut_at_dimension, frequency=100., neighborhood_factor=0.2):
    assert 0 < neighborhood_factor < 0.5
    series = data[cut_at_dimension]
    series.reset_index(drop=True, inplace=True)
    search_radius = round(2.5 * frequency)  # the first minimum should be found within the first 2.5 s of a recording
    neighborhood_range = round(frequency * neighborhood_factor)  # range to search the next cycle start in
    values_per_second = round(frequency)
    indices = list()

    index = series.iloc[:search_radius].idxmin()
    indices.append(index)

    while index < len(series):
        start = max(index + values_per_second - neighborhood_range, 0)
        end = min(index + values_per_second + neighborhood_range, len(series))

        if start >= end:
            break

        index = series.iloc[start:end].idxmin()
        indices.append(index)

    return indices
