import numpy as np
import pandas as pd

from src.core.base_classes import SafeTransformer
from src.core.constants import DATA_FRAMES_KEY
from src.core.error_handling.exceptions import CycleCleaningException, EmptyDataFrameException
from src.core.utility.interpolation_helper import interpolate_cycle_list


class CycleCleaner(SafeTransformer):

    def __init__(self, column_index=None, cycle_deviation_threshold=2, use_length=True, use_mean=True, use_std=True,
                 **kwargs):
        self.column_index = column_index
        self.cycle_deviation_threshold = cycle_deviation_threshold
        self.use_length = use_length
        self.use_mean = use_mean
        self.use_std = use_std
        super().__init__(**kwargs)

    def transform(self, data):
        result = super().transform(data)
        return result

    def _transform(self, cycle_data_data_frames):
        if len(cycle_data_data_frames) == 0:
            raise EmptyDataFrameException()

        cleaned_cycles = clean_cycle_data(cycle_data_data_frames, column_index=self.column_index,
                                          cycle_deviation_threshold=self.cycle_deviation_threshold, use_length=self.use_length,
                                          use_mean=self.use_mean, use_std=self.use_std)

        if len(cleaned_cycles) == 0:
            raise CycleCleaningException("Cycle detection: to much variance in cycles. All were removed by cycle cleaning.")

        return cleaned_cycles.to_frame()


class CycleInterpolator(SafeTransformer):

    def __init__(self, interpolation_method="linear", cycle_length=100, **kwargs):
        self.interpolation_method = interpolation_method
        self.cycle_length = cycle_length
        super().__init__(**kwargs)

    def _transform(self, data):
        if len(data) == 0:
            raise EmptyDataFrameException("Received no data frames for interpolation")
        return interpolate_cycle_list(data[DATA_FRAMES_KEY], self.cycle_length, self.interpolation_method)


def get_selected_column_names(selected_columns, column_names):
    """
    Transform given column selection of different typings to a list of resulting column names. Note that duplicate column names in return are
    possible if multiple selection parameter are transformed to the same column.

    Parameters
    ----------
    selected_columns : None, int, str or array-like
        Selection parameter with will be mapped to the given column names if possible.
    column_names : array-like
        Column names to be matched for given indices.

    Returns
    -------
    selected_column_names: list
        Returns list of given column name(s) or column name(s) for given indices. If selected_columns is None or an empty list, all given column
        names will be returned.

    Raises
    -------
    IndexError
        If given indices are out of bound for the given column names.
    AssertionError
        If a column name given in selected_columns is not available in column_names.
    """
    if selected_columns is None:
        selected_columns = list(column_names)
    elif not isinstance(selected_columns, list):
        selected_columns = [selected_columns]

    selected_column_names = list()
    for selected_column in selected_columns:
        if isinstance(selected_column, int):
            selected_column = column_names[selected_column]
        assert selected_column in column_names
        selected_column_names.append(selected_column)
    if len(selected_column_names) == 0:
        selected_column_names = list(column_names)
    return selected_column_names


def test_get_selected_column_names():
    column_names = ["a", "b", "c", "d", "e", "f"]
    assert get_selected_column_names(None, column_names) == column_names
    assert get_selected_column_names([], column_names) == column_names
    # noinspection PyTypeChecker
    assert get_selected_column_names(1, column_names) == ["b"]
    assert get_selected_column_names([2, 3, 4], column_names) == ["c", "d", "e"]
    assert get_selected_column_names("a", column_names) == ["a"]
    assert get_selected_column_names(["a"], column_names) == ["a"]
    assert get_selected_column_names(["a", "f"], column_names) == ["a", "f"]
    assert get_selected_column_names([1, "f"], column_names) == ["b", "f"]
    assert get_selected_column_names([0, "a"], column_names) == ["a", "a"]
    np.testing.assert_raises(AssertionError, get_selected_column_names, ["g"], column_names)
    np.testing.assert_raises(IndexError, get_selected_column_names, [8], column_names)


def clean_cycles_by_metric(method, data, column_indices, threshold):
    """
    Used to remove cycles which deviate to much from the others in regards to the given metric method. Be aware that operations are performed in
    place.

    Parameters
    ----------
    method
        A method to calculate a (single) metric per cycle (e.g. len, mean, std)
    data : pd.DataFrame
        Contains data frames which represent cycles.
    column_indices : int or string
        Index of the column used for cleaning or column name of the column to use.
    threshold
        Threshold defining how many standard deviations the outliers must be away to be removed.

    Returns
    -------
        Cleaned cycle list.
    """
    data_frames = data.copy(deep=True)[DATA_FRAMES_KEY]
    if len(data_frames) == 0:
        return data_frames

    columns = get_selected_column_names(column_indices, data_frames[0].columns)

    metrics_list = list()
    metric_boundary = list()
    for column_name in columns:
        metrics_list.append([method(data_frame.loc[:, column_name]) for data_frame in data_frames])
        mean_metric = np.mean(metrics_list[-1])
        std_metric = np.std(metrics_list[-1])

        metric_boundary.append((mean_metric + threshold * std_metric, mean_metric - threshold * std_metric))

    rows_to_remove = set()
    for i, metrics in enumerate(metrics_list):
        for j, metric in enumerate(metrics):
            if metric > metric_boundary[i][0] or metric < metric_boundary[i][1]:
                rows_to_remove.add(j)
    rows_to_remove = list(rows_to_remove)
    rows_to_remove.sort()

    data_frames.drop(data_frames.index[rows_to_remove])
    # for i in reversed(rows_to_remove):
    #     del data_frames[i]

    return data_frames


def clean_cycle_data(cycle_data_list, column_index=None, cycle_deviation_threshold=2, use_length=True, use_mean=True, use_std=True):
    """
    Post-process cycle data and remove cycles that deviate to much from the others mean, length, and standard deviation.

    Parameters
    ----------
    cycle_data_list : list(pd.DataFrame)
        A list of data frames (cycle data).
    column_index : int or str
        If provided, this column will be used for cleaning otherwise an average of all columns will be used.
    cycle_deviation_threshold : int, default=2
        Indicates how many deviations a value can be away from the mean.
    use_length : bool, default=True
        Use length of cycles as a indicator to clean the given cycles.
    use_mean : bool, default=True
        Use mean of cycles as a indicator to clean the given cycles.
    use_std : bool, default=True
        Use std of cycles as a indicator to clean the given cycles.

    Returns
    -------
    cycle_data_list : list(pd.DataFrames)
        A list of data frames. Each data frame hold post-processed and cleaned data_frames.
    """
    if use_length:
        cycle_data_list = clean_cycles_by_metric(len, cycle_data_list, column_index, cycle_deviation_threshold)
    if use_mean:
        cycle_data_list = clean_cycles_by_metric(np.mean, cycle_data_list, column_index, cycle_deviation_threshold)
    if use_std:
        cycle_data_list = clean_cycles_by_metric(np.std, cycle_data_list, column_index, cycle_deviation_threshold)
    return cycle_data_list


def select_n_best_cycles(cycle_list, cycle_rank_list, n=10):
    if len(cycle_list) > n:
        # get 10 best cycles (with less deviation)
        deviation_indices = np.where(cycle_rank_list <= np.sort(cycle_rank_list)[min(len(cycle_list), n - 1)])[0]
        best_cycles_list = [cycle for i, cycle in enumerate(cycle_list) if i in deviation_indices]
        return best_cycles_list
    else:
        return cycle_list
