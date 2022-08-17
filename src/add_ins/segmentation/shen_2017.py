import math

from src.add_ins.segmentation import split_data_for_indices
from src.core.base_classes import SafeTransformer


class Shen2017SegmentationTransformer(SafeTransformer):

    def __init__(self, values_per_second, seconds_per_window=1, sample_std=True, column_to_use="linearAcceleration_x", use_performance_boost=False,
                 **kwargs):
        self.values_per_second = values_per_second
        self.seconds_per_window = seconds_per_window
        self.sample_std = sample_std
        self.column_to_use = column_to_use
        self.use_performance_boost = use_performance_boost
        super().__init__(**kwargs)

    def _transform(self, data):
        import numpy as np
        # Question: on which dimension of which sensor was this performed ?

        index_column = data[self.column_to_use]
        number_of_iterations = math.ceil(len(index_column) / 10)

        # listening rate is given -> 20 values per second
        window_length = self.values_per_second * self.seconds_per_window
        overlap = 0.5

        start = 0
        indices = set()
        for _ in range(number_of_iterations):
            end = min(start + window_length, len(index_column))
            sub_set = index_column.iloc[start:end, ]
            start_indices = np.array(get_cycle_start_indices(sub_set))
            indices.update(start + start_indices)
            start += int(window_length * overlap)

        indices = list(indices)
        indices.sort()
        if self.use_performance_boost:
            indices = perform_cycle_concatenation(indices, window_length)
        return split_data_for_indices(data, indices)


def perform_cycle_concatenation(indices, values_per_second):
    """
    Searches for single steps (half a cycle) found by the cycle detection and merges consecutive steps together.

    Parameters
    ----------
    indices: array-like
        Cycle start indices found by cycle detection, possibly containing single steps (half a cycle).
    values_per_second: float
        Number of values per second in data set.

    Returns
    -------
    new_indices: array-like
        Cycle start indices with merged consecutive steps.
    """
    differences = [indices[i + 1] - indices[i] for i in range(len(indices) - 1)]

    new_indices = list()
    predecessor_half = False
    counter = -1
    for i, difference in enumerate(differences):
        if difference <= 0.8 * values_per_second:
            if predecessor_half:
                counter += 1
                predecessor_half = False
            else:
                counter += 1
                new_indices.append(indices[counter])
                predecessor_half = True
        else:
            counter += 1
            new_indices.append(indices[counter])
            predecessor_half = False

    if len(indices) > 1:
        new_indices.append(indices[-1])
    return new_indices


def get_cycle_start_indices(data, threshold_a=1., threshold_b=0.6, number_of_comparisons=10):
    data_array = data.to_numpy()
    condition_one_peaks = get_peaks(data_array, threshold=threshold_a, number_of_comparisons=number_of_comparisons)
    condition_two_peaks = get_difference_peaks(data_array, threshold=threshold_b, number_of_comparisons=number_of_comparisons)
    condition_three_peaks = get_points_with_changing_gradient(data_array, number_of_comparisons=number_of_comparisons)
    indices = list(set(condition_one_peaks).intersection(set(condition_two_peaks)).intersection(set(condition_three_peaks)))
    return indices


# Condition 1 of the paper
def get_peaks(data, threshold=0.6, number_of_comparisons=10):
    """
    Searches for values in given data which exceed a given threshold and exceeds all neighbouring values for a given range,

    Parameters
    ----------
    data: array-like
        Data to be searched
    threshold: float, default=0.6
        Minimum value for values to be considered a peak. Default value is taken from the paper.
    number_of_comparisons: int, default=10
        Number of elements to compare the current one with. Must be even. Default value is taken from the paper.

    Returns
    -------
    peak_indices: array-like
        Indices of values that meet the conditions

    Raises
    -------
    AssertionException
        If number_of_comparisons is not even
    """
    assert (number_of_comparisons % 2) == 0
    peak_indices = list()
    for index in range(len(data)):
        current_data = data[index]
        if current_data <= threshold:
            continue
        start = max(0, int(index - (number_of_comparisons / 2)))
        end = min(len(data), int(index + (number_of_comparisons / 2) + 1))
        is_greatest_value = True
        for compare_data in data[start:end]:
            if compare_data > current_data:
                is_greatest_value = False
                break
        if is_greatest_value:
            peak_indices.append(index)
    return peak_indices


def test_peak_search():
    data = [1, 2, 3, 4, 5, 3, 2, 1, 3, 2, 4, 2, 4, 1, 6, 2, 2, 7, 2, 7]
    peak_indices = get_peaks(data, threshold=3, number_of_comparisons=4)
    assert peak_indices == [4, 10, 14, 17, 19]


# Condition 2 of the paper
def get_difference_peaks(data, threshold=1., number_of_comparisons=10, allow_border_peaks=False):
    """
    Parameters
    ----------
    data: array-like
        Data to be searched
    threshold: float, default=1.
        Default value is taken from the paper.
    number_of_comparisons: int, default=10
        Number of elements to compare the current one with. Must be even. Default value is taken from the paper.
    allow_border_peaks: bool, default=False
        If true, considers bordering values to be able to be the peak. NOTE: This is not part of the paper.

    Returns
    -------
    peak_indices: array-like
        Indices of values that meet the conditions

    Raises
    -------
    AssertionException
        If number_of_comparisons is not even
    """
    peak_indices = list()
    for index in range(len(data)):
        current_data = data[index]

        max_down = threshold
        start = max(0, int(index - (number_of_comparisons / 2)))
        has_data_before = False
        for compare_data in data[start:index]:
            has_data_before = True
            max_down = max(max_down, current_data - compare_data)

        max_up = threshold
        end = min(len(data), int(index + (number_of_comparisons / 2) + 1))
        has_data_after = False
        for compare_data in data[index + 1:end]:
            has_data_after = True
            max_up = max(max_up, current_data - compare_data)

        if (allow_border_peaks and not has_data_before or max_down > threshold) \
                and (allow_border_peaks and not has_data_after or max_up > threshold):
            peak_indices.append(index)
    return peak_indices


def test_diff_peak_search():
    data = [1, 2, 3, 4, 5, 3, 2, 1, 3, 2, 4, 2, 4, 1, 6, 2, 2, 7, 2, 7]
    peak_indices = get_difference_peaks(data, threshold=3, number_of_comparisons=4)
    assert peak_indices == [14, 17]


# Condition 3 of the paper
def get_points_with_changing_gradient(data, number_of_comparisons=10):
    """
    Checks if the gradient of preceding values is increasing and the gradient of succeeding values decreasing.

    Parameters
    ----------
    data: array-like
        Data to be searched
    number_of_comparisons: int, default=10
        Number of elements to compare the current one with. Must be even. Default value is taken from the paper.

    Returns
    -------
    peak_indices: array-like
        Indices of values that meet the conditions

    See Also
    -------
    get_points_with_changing_gradient_improved : Computationally improved version fo this method.
    """
    peak_indices = list()
    for index in range(len(data)):

        sum_down = 0
        start = max(0, int(index - (number_of_comparisons / 2)))
        for i in range(start, index):
            sum_down += data[i + 1] - data[i]

        sum_up = 0
        end = min(len(data), int(index + (number_of_comparisons / 2)) + 1)
        for i in range(index + 1, end):
            sum_up += data[i] - data[i - 1]

        if sum_down > 0 > sum_up:
            peak_indices.append(index)
    return peak_indices


def get_points_with_changing_gradient_improved(data, number_of_comparisons=10):
    """
    This is just the difference between the first and last of the considered neighboring values.

    Parameters
    ----------
    data: array-like
        Data to be searched
    number_of_comparisons: int, default=10
        Number of elements in range over the current index value. Must be even. Default value is taken from the paper. Name is kept according to
        original implementation.

    Returns
    -------
    peak_indices: array-like
        Indices of values that meet the conditions

    See Also
    -------
    get_points_with_changing_gradient : Implementation following the formulas given in the original paper.
    """
    peak_indices = list()
    for index in range(len(data)):

        start = max(0, int(index - (number_of_comparisons / 2)))
        end = min(len(data), int(index + (number_of_comparisons / 2) + 1))
        gradient_before = data[index] - data[start]
        gradient_after = data[end - 1] - data[index]

        if gradient_before > 0 > gradient_after:
            peak_indices.append(index)
    return peak_indices


def test_get_points_with_changing_monotony():
    data = [1, 2, 3, 4, 5, 3, 2, 1, 3, 2, 4, 2, 4, 1, 6, 2, 2, 7, 2, 7]
    peak_indices_original = get_points_with_changing_gradient(data, number_of_comparisons=4)
    peak_indices_improved = get_points_with_changing_gradient_improved(data, number_of_comparisons=4)
    assert peak_indices_improved == peak_indices_original == [3, 4, 14]
