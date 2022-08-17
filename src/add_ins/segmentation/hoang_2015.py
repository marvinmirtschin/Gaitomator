import copy
import math

import numpy as np
import pandas as pd
import pytest
import scipy.signal

from src.core.base_classes import SafeTransformer
from src.core.constants import DATA_FRAMES_KEY
from src.core.error_handling.exceptions import SegmentationException
from src.add_ins.segmentation import split_data_for_indices
from src.core.utility.cycle_data import clean_cycle_data
from src.core.utility.testing_helper import get_example_cycles


class CycleDetectionTransformer(SafeTransformer):

    def __init__(self, cut_at_dimension='accelerometer_z', window_size=10, **kwargs):
        # Note that the default value for window_size was chosen by us.
        self.cut_at_dimension = cut_at_dimension
        self.window_size = window_size
        super().__init__(**kwargs)

    def _transform(self, data):
        indices = _get_cycle_start_indices(data, cut_at_dimension=self.cut_at_dimension, window_size=self.window_size)
        cycle_data = split_data_for_indices(data, indices)
        cycles = clean_cycle_data(cycle_data, use_mean=False, use_std=False)
        return pd.DataFrame(cycles, columns=[DATA_FRAMES_KEY])


class SegmentationTransformer(SafeTransformer):

    def __init__(self, cycles_per_segment=4, number_of_overlapping_cycles=2, **kwargs):
        self.cycles_per_segment = cycles_per_segment
        self.number_of_overlapping_cycles = number_of_overlapping_cycles
        super().__init__(**kwargs)

    def _transform(self, data):
        return merge_cycles_into_segments(data, self.cycles_per_segment, self.number_of_overlapping_cycles)


def _get_cycle_start_indices(data, cut_at_dimension, window_size):
    # TODO: user_factor is not settable; should also be renamed
    series = data[cut_at_dimension]

    indices = _find_negative_peaks_indices(series)
    indices = _remove_peaks_by_amplitude(series.copy(), copy.copy(indices), user_factor=0.05)  # This is 0.5 in OG code, was 0.1 for us
    coefficients = _calculate_auto_correlation_coefficients(series)
    coefficients = _apply_moving_average_filter(coefficients, window_size=window_size)

    length = _find_approximate_cycle_length(coefficients)
    return _remove_peaks_by_length2(series.values, indices, length)


def _find_negative_peaks_indices(series):
    indices = scipy.signal.find_peaks(-series)
    return list(indices[0])


def _remove_peaks_by_amplitude(series: pd.Series, indices, user_factor):
    values = pd.Series([series.values[index] for index in indices])
    mean = values.mean()
    std = values.std()
    threshold = mean - user_factor * std

    i = len(values) - 1
    for value in reversed(values):
        if value > threshold:
            del indices[i]
        i -= 1
    return indices


def _calculate_auto_correlation_coefficients(series):
    """
    Based on the formula presented in the paper 'On The Instability of Sensor Orientation in Gait Verification on Mobile Phone'.
    Parameters
    ----------
    series

    Returns
    -------

    """
    values = series.values
    coefficients = list()
    number_of_items = len(series)
    for t in range(number_of_items):
        factor1 = number_of_items / (number_of_items - t)
        numerator = sum([values[index] * values[index + t] for index in range(number_of_items - t)])
        dividend = series.apply(np.square).sum()
        factor2 = numerator / dividend
        coefficient = factor1 * factor2
        coefficients.append(coefficient)
    return pd.Series(coefficients)


def _apply_moving_average_filter(series, window_size):
    return series.rolling(window=window_size, min_periods=1).mean()


def _find_approximate_cycle_length(auto_corr_coefficients):
    """
    Taken from https://github.com/thanghoang/GaitAuth/blob/master/detectGaitCycle.m
    """
    cycle_length = 0
    flag = 0

    for i in range(2, len(auto_corr_coefficients) - 1):
        if auto_corr_coefficients[i] > auto_corr_coefficients[i - 1] and auto_corr_coefficients[i] > auto_corr_coefficients[i + 1]:
            flag += 1
            if flag == 2:
                cycle_length = i - 1
                break
    return cycle_length


def _remove_peaks_by_length(series, indices, approximate_length, alpha=0.25, beta=0.75, gamma=1 / 6):
    """
    Taken from https://github.com/thanghoang/GaitAuth/blob/master/detectGaitCycle.m

    Parameters
    ----------
    series
    indices : array-like
        Amplitude cleaned indices of minima in data
    approximate_length : int
        Approximate length of a cycle.
    alpha : float, default=0.25
        User-specific value taken from original implementation.
    beta : float, default=0.75
        User-specific value taken from original implementation.
    gamma : float, default=1/6
        User-specific value taken from original implementation.

    Returns
    -------
        Cycle start point indices cleaned by length and amplitude.
    """
    i = 2
    while i < len(indices):
        if indices[i] - indices[i - 1] < approximate_length and series[indices[i]] < series[indices[i - 1]]:
            del indices[i - 1]
        else:
            break
    i = 1
    # Note: -1 is a adaptation from the original code to ensure that no index error occurs
    while i < len(indices) - 1:
        if indices[i] - indices[i - 1] < alpha * approximate_length:
            if series[indices[i]] <= series[indices[i - 1]]:
                del indices[i - 1]
                continue
            else:
                del indices[i]
                continue
        elif indices[i] - indices[i - 1] < beta * approximate_length:
            if indices[i + 1] - indices[i] < gamma * approximate_length:
                if series[indices[i + 1]] <= series[indices[i]]:
                    del indices[i]
                    continue
                else:
                    del indices[i + 1]
                    continue
            else:
                del indices[i]
                continue
        else:
            i += 1

    if indices[-1] - indices[-2] < beta * approximate_length:
        del indices[i - 1]

    return indices


def _remove_peaks_by_length2(series, indices, approximate_length, alpha=0.25, beta=0.75, gamma=1 / 6):
    """
    Taken from https://github.com/thanghoang/GaitAuth/blob/master/detectGaitCycle.m

    Parameters
    ----------
    series
    indices : array-like
        Amplitude cleaned indices of minima in data
    approximate_length : int
        Approximate length of a cycle.
    alpha : float, default=0.25
        User-specific value taken from original implementation.
    beta : float, default=0.75
        User-specific value taken from original implementation.
    gamma : float, default=1/6
        User-specific value taken from original implementation.

    Returns
    -------
        Cycle start point indices cleaned by length and amplitude.
    """
    i = 1
    while i < len(indices):
        if indices[i] - indices[i - 1] < approximate_length and series[indices[i]] < series[indices[i - 1]]:
            indices[i - 1] = -1
            i += 1
        else:
            break

    i = len(indices) - 1
    while i > 0:
        if indices[i] == -1:
            del indices[i]
        i -= 1

    i = 1
    # Note: -1 is a adaptation from the original code to ensure that no index error occurs
    while i < len(indices) - 1:
        if indices[i] - indices[i - 1] < alpha * approximate_length:
            if series[indices[i]] <= series[indices[i - 1]]:
                indices[i - 1] = -1
            else:
                indices[i] = -1
        elif indices[i] - indices[i - 1] < beta * approximate_length:
            if indices[i + 1] - indices[i] < gamma * approximate_length:
                if series[indices[i + 1]] <= series[indices[i]]:
                    indices[i] = -1
                else:
                    indices[i + 1] = -1
            else:
                indices[i] = -1

        i += 1

    if indices[-1] - indices[-2] < beta * approximate_length:
        indices[i - 1] = -1

    i = len(indices) - 1
    while i > 0:
        if indices[i] == -1:
            del indices[i]
        i -= 1

    return indices


def merge_cycles_into_segments(cycle_data, cycles_per_segment, number_of_overlapping_cycles):
    """

    Parameters
    ----------
    cycle_data: pd.DataFrame
        DataFrame containing separate cycles as rows in the column src.core.constants#DATA_FRAMES_KEY.
    cycles_per_segment: int
        Number of cycles to put into a single segment.
    number_of_overlapping_cycles: int

    Returns
    -------
        DataFrame containing segments as rows in the column src.core.constants#DATA_FRAMES_KEY.
    """
    segments = merge_cycles_into_segments_list(cycle_data[DATA_FRAMES_KEY].values, cycles_per_segment=cycles_per_segment,
                                               number_of_overlapping_cycles=number_of_overlapping_cycles)
    return pd.DataFrame({DATA_FRAMES_KEY: segments})


def merge_cycles_into_segments_list(cycle_data_list, cycles_per_segment, number_of_overlapping_cycles):
    if 0 < number_of_overlapping_cycles < cycles_per_segment:
        number_of_segments = math.ceil(max((len(cycle_data_list) - (cycles_per_segment - 1)), 0) / max(number_of_overlapping_cycles, 1))
    elif number_of_overlapping_cycles == 0 and cycles_per_segment > 0:
        number_of_segments = math.ceil((len(cycle_data_list) / cycles_per_segment))
    else:
        raise SegmentationException(
            "Number of cycles per segment must be greater than 0 and greater than number_of_overlapping_cycles but were {} and {}".format(
                cycles_per_segment, number_of_overlapping_cycles))

    segments = list()
    for i in range(number_of_segments):
        start = i * (cycles_per_segment - number_of_overlapping_cycles)
        end = min(start + cycles_per_segment, len(cycle_data_list))
        segment = cycle_data_list[start:end]
        segments.append(pd.concat(segment))
    return segments


def test_segmentation():
    cycle1, cycle2, cycle3, cycle4 = get_example_cycles()
    data = pd.DataFrame(
        [cycle1, cycle2, cycle3, cycle4, cycle1, cycle2, cycle3, cycle4, cycle1, cycle2, cycle3, cycle4, cycle1, cycle2, cycle3, cycle4],
        columns=[DATA_FRAMES_KEY])
    assert len(merge_cycles_into_segments(data, 4, 2)), 8
    assert len(merge_cycles_into_segments(data, 8, 7)), 8
    assert len(merge_cycles_into_segments(data, 1, 0)), 16
    with pytest.raises(SegmentationException):
        merge_cycles_into_segments(data, 0, 1)
    with pytest.raises(SegmentationException):
        merge_cycles_into_segments(data, 5, 5)
