import math
import random

import numpy as np
import pandas as pd

from src.core import constants
from src.core.base_classes import FeatureCalculator, NestedDataFrameTransformer
from src.core.constants import DATA_FRAMES_KEY, LABEL_RECORD_KEY, LABEL_USER_KEY, TEST_DATA
from src.core.error_handling.exceptions import EmptyDataFrameException
from src.core.utility.testing_helper import get_segment_data_frame


class MeanAbsoluteDifferenceCalculator(FeatureCalculator):
    # Mean Absolute Difference (Average absolute deviation would be more correct)

    def __init__(self, **kwargs):
        super().__init__(feature_name="meanAbsDiff", **kwargs)

    def _transform(self, data: pd.DataFrame()):
        return data.apply(calculate_average_absolute_deviation, raw=True)


def calculate_average_absolute_deviation(data):
    return np.sum((abs(value - np.mean(data)) for value in data))


def test_mean_absolute_difference_calculator():
    df = pd.DataFrame(data=[[1, 2], [3, 4], [5, 6]])
    calculator = MeanAbsoluteDifferenceCalculator()
    transformed_df = calculator.transform(df)
    assert np.allclose(transformed_df.iloc[0].values, [4, 4], 0)


class RootMeanSquareCalculator(FeatureCalculator):

    def __init__(self, **kwargs):
        super().__init__(feature_name="rms", **kwargs)

    def _transform(self, data):
        return data.apply(calculate_root_mean_square, raw=True)


def calculate_root_mean_square(data):
    return np.sqrt(np.mean(np.square(data)))


def test_root_mean_square_calculator():
    df = pd.DataFrame(data=[[1, 2], [3, 4], [5, 6]])
    calculator = RootMeanSquareCalculator()
    transformed_df = calculator.transform(df)
    assert np.allclose(transformed_df.iloc[0].values, [3.415, 4.32], 0.001)


class StandardDeviationCalculator(FeatureCalculator):

    def __init__(self, use_sample_std=True, **kwargs):
        self.use_sample_std = use_sample_std
        super().__init__(feature_name="std", **kwargs)

    def _transform(self, data):
        return data.apply(calculate_standard_deviation, args=[self.use_sample_std], raw=True)


def calculate_standard_deviation(data, use_sample_std=True):
    cleaned_data = np.square(data - np.mean(data))
    if use_sample_std:
        variance = np.sum(cleaned_data)
        variance /= len(data) - 1
    else:
        variance = np.mean(cleaned_data)
    return np.sqrt(variance)


def test_standard_deviation_calculator():
    df = pd.DataFrame(data=[[1, 2], [3, 4], [5, 6]])
    calculator = StandardDeviationCalculator()
    transformed_df = calculator.transform(df)
    assert np.allclose(transformed_df.iloc[0].values, [2, 2], 0)


def test_standard_deviation_calculator_no_sample():
    df = pd.DataFrame(data=[[1, 2], [3, 4], [5, 6]])
    calculator = StandardDeviationCalculator(use_sample_std=False)
    transformed_df = calculator.transform(df)
    assert np.allclose(transformed_df.iloc[0].values, [1.632, 1.632], 0.001)


def test_standard_deviation_nested_calculator():
    calculator = NestedDataFrameTransformer(StandardDeviationCalculator())
    transformed_df = calculator.transform(TEST_DATA)[DATA_FRAMES_KEY]
    assert np.allclose(transformed_df.iloc[0].values, [4.74341649, 4.74341649, 4.74341649], 0)


def test_standard_deviation_nested_calculator_no_sample():
    calculator = NestedDataFrameTransformer(StandardDeviationCalculator(use_sample_std=False))
    transformed_df = calculator.transform(TEST_DATA)[DATA_FRAMES_KEY]
    assert np.allclose(transformed_df.iloc[0].values, [4.24264069, 4.24264069, 4.24264069], 0.001)


class WaveformLengthCalculator(FeatureCalculator):

    def __init__(self, **kwargs):
        super().__init__(feature_name="waveformLength", **kwargs)

    def _transform(self, data):
        return data.apply(calculate_waveform_length, raw=True)


def calculate_waveform_length(data):
    return np.sum(np.abs(data[:-1] - data[1:]))


def test_waveform_length_calculator():
    df = pd.DataFrame(data=[[1, 2, 6], [3, 4, 4], [5, 6, 2]])
    calculator = WaveformLengthCalculator()
    transformed_df = calculator.transform(df)
    assert np.allclose(transformed_df.iloc[0].values, [4, 4, 4], 0)
    df = pd.DataFrame(data=[[1, -1, 6], [2, 2, -5], [3, -3, 4], [4, 4, -3], [5, -5, 2], [6, 6, -1]])
    transformed_df = calculator.transform(df)
    assert np.allclose(transformed_df.iloc[0].values, [5, 35, 35], 0)


class BinHistogramDistributionCalculator(FeatureCalculator):

    def __init__(self, number_of_bins=10, **kwargs):
        self.number_of_bins = number_of_bins
        super().__init__(feature_name="binHistogram", **kwargs)

    def _transform(self, data):
        # no raw values as number of rows may be greater then before
        return data.apply(_calculate_histogram_distribution, args=[self.number_of_bins])


def _calculate_histogram_distribution(series, number_of_bins=10):
    binned_values = np.histogram(series, bins=number_of_bins)[0]
    return binned_values / len(series)


def test_bin_histogram_distribution_calculator():
    expected_values = [0.33333333, 0, 0, 0, 0, 0.33333333, 0, 0, 0, 0.33333333]
    df = pd.DataFrame(data=[[1, 2], [3, 4], [5, 6]], columns=["a", "b"])
    calculator = BinHistogramDistributionCalculator()
    transformed_df = calculator.transform(df)
    assert np.allclose(transformed_df['binHistogram_a'].values, expected_values, 0.000001)
    assert np.allclose(transformed_df['binHistogram_b'].values, expected_values, 0.000001)


class AverageMinimumAccelerationCalculator(FeatureCalculator):

    def __init__(self, **kwargs):
        super().__init__(feature_name="avgMin", **kwargs)

    def _transform(self, data):
        cycles = get_cycles_from_segment(data)
        return cycles.apply(np.min).apply(np.mean, raw=True)


def test_average_minimum_acceleration_calculator():
    df = get_segment_data_frame()
    result = NestedDataFrameTransformer(AverageMinimumAccelerationCalculator()).transform(df)[DATA_FRAMES_KEY]
    assert np.allclose(result[0], [-0.175, -0.175, -0.175], 0.00001)
    assert np.allclose(result[1], [-0.33333, -0.33333, -0.33333], 0.0001)


class AverageMaximumAccelerationCalculator(FeatureCalculator):

    def __init__(self, **kwargs):
        super().__init__(feature_name="avgMax", **kwargs)

    def _transform(self, data):
        cycles = get_cycles_from_segment(data)
        return cycles.apply(np.max).apply(np.mean, raw=True)


def test_average_maximum_acceleration_calculator():
    calculator = NestedDataFrameTransformer(AverageMaximumAccelerationCalculator())
    data_frame = get_segment_data_frame()
    result = calculator.transform(data_frame)[DATA_FRAMES_KEY]
    assert np.allclose(result[0], [1.35, 1.35, 1.35], 0.00001)
    assert np.allclose(result[1], [1.133333, 1.133333, 1.133333], 0.0001)


class AverageGaitCycleLengthCalculator(FeatureCalculator):

    def __init__(self, **kwargs):
        super().__init__(feature_name="avgGaitLen", **kwargs)

    def _prepare_result(self, result):
        return result

    def _transform(self, data):
        cycles = get_cycles_from_segment(data)
        length = cycles.apply(len)
        return pd.DataFrame(data=[np.mean(length)], columns=[self.feature_name])


def test_average_length_acceleration_calculator():
    df = get_segment_data_frame()
    result = NestedDataFrameTransformer(AverageGaitCycleLengthCalculator()).transform(df)[DATA_FRAMES_KEY]
    assert np.allclose(result[0], [4])
    assert np.allclose(result[1], [4.333333], 0.0001)


class MeanCalculator(FeatureCalculator):

    def __init__(self, **kwargs):
        super().__init__(feature_name="mean", **kwargs)

    def _transform(self, data):
        return data.apply(np.mean)


class MinMaxDifferenceCalculator(FeatureCalculator):

    def __init__(self, **kwargs):
        super().__init__(feature_name="mmDiff", **kwargs)

    def _transform(self, data):
        return data.apply(_calculate_max_min_diff)


def _calculate_max_min_diff(series: pd.Series):
    maximum = series.max()
    minimum = series.min()
    return maximum - minimum


class MinimumCalculator(FeatureCalculator):

    def __init__(self, **kwargs):
        super().__init__(feature_name="min", **kwargs)

    def _transform(self, data):
        return data.apply(min)


class MaximumCalculator(FeatureCalculator):

    def __init__(self, **kwargs):
        super().__init__(feature_name="max", **kwargs)

    def _transform(self, data):
        return data.apply(max)


class CorrelationCalculator(FeatureCalculator):
    """
    Calculates the correlation between all distinct combinations of dimensions for each of the sensors given.

    Parameters
    ----------
    method : str or callable, default='pearson'
        One of ['pearson', 'spearman', 'kendall']  or any function, method, or object with .__call__() that accepts two one-dimensional arrays and
        returns a floating-point number.
    """

    def __init__(self, method='pearson', **kwargs):
        self.method = method
        super().__init__(feature_name="correlation", **kwargs)

    def _prepare_result(self, result):
        return result

    def _transform(self, data: pd.DataFrame):
        correlations = list()

        # split columns based on sensors
        sensors = list(set(['_'.join(column.split('_')[:-1]) for column in data.columns]))
        sensors.sort()  # create deterministic order for tests
        for sensor in sensors:
            df = data.filter(regex=sensor)

            # correlate all dimensions of each sensor with one another
            columns = df.columns
            for i in range(len(columns) - 1):
                for j in range(i + 1, len(columns)):
                    series_1 = df.iloc[:, i]
                    series_2 = df.iloc[:, j]
                    # TODO: Improvement: if one of the series is constant its std will be 0, therefore the correlation will be nan. We could add a
                    #  different handling. Ideas:
                    #  - if both are constant the correlation should be 1 or -1
                    #  - could adjust a single value by an infinitesimal to make the std not 0
                    correlations.append(pd.Series(data=series_1.corr(series_2, method=self.method),
                                                  name=self.feature_name + "_" + "{sensor}_{dim1}_{dim2}"
                                                  .format(sensor=sensor, dim1=series_1.name.split('_')[-1],
                                                          dim2=series_2.name.split('_')[-1])))
        return pd.DataFrame(correlations).T


def test_correlation():
    data = get_testing_data_frame()
    correlation = NestedDataFrameTransformer(CorrelationCalculator()).transform(data)
    assert len(correlation) == 2
    assert np.array_equal(correlation.iloc[0][DATA_FRAMES_KEY].columns,
                          ["correlation_sensor1_x_y", "correlation_sensor1_x_z", "correlation_sensor1_y_z"])
    assert np.array_equal(correlation.iloc[1][DATA_FRAMES_KEY].columns, ["correlation_sensor1_x_y", "correlation_sensor2_z_y"])


class InterquartileRangeCalculator(FeatureCalculator):

    def __init__(self, interpolation_method="linear", **kwargs):
        # Question: interpolation parameter is not given
        self.interpolation_method = interpolation_method
        super().__init__(feature_name="iqr", **kwargs)

    def _transform(self, data: pd.DataFrame):
        # noinspection PyTypeChecker
        return data.apply(calculate_interquartile_range, args=(self.interpolation_method,), raw=True)


def calculate_interquartile_range(data: np.ndarray, interpolation_method="linear"):
    return np.percentile(data, 75, interpolation=interpolation_method) - np.percentile(data, 25, interpolation=interpolation_method)


def test_iqr():
    data = get_testing_data_frame()
    iqr = NestedDataFrameTransformer(InterquartileRangeCalculator()).transform(data)
    assert np.array_equal(iqr.iloc[0]["data_frames"].columns, ["iqr_sensor1_x", "iqr_sensor1_y", "iqr_sensor1_z"])


class DynamicTimeWarpingCalculator(FeatureCalculator):

    def __init__(self, train_row=None, **kwargs):
        super().__init__(feature_name="dtw", **kwargs)
        self.train_row = train_row
        self.template = None

    def _fit(self, data, y=None, **fit_params):
        """
        Select a cycle as template. The cycle will later be used to calculate the distance between itself and the given cycles using dynamic time
        warping.

        Parameters
        ----------
        data: pd.DataFrame
            Data containing the cycle to be trained. For specifying the data to take use self.train_row.
        y: array-like, default=None
            Labels used for fitting the data.

        Returns
        -------
        self: DynamicTimeWarpingCalculator
            Returns the calculator itself
        """
        # Question: did they choose a specific one or random
        if data.empty:
            raise EmptyDataFrameException()

        if self.train_row is None:
            selected_row = math.floor(random.random() * len(data))
        elif 0 < self.train_row > len(data):
            raise Exception("Unable to use given row. Should use row {} out of {} row{}"
                            .format(self.train_row, len(data), "s." if len(data) != 1 else "."))
        else:
            selected_row = self.train_row

        row = data.iloc[selected_row]
        if LABEL_USER_KEY in data.columns:
            self.user_id = row.pop(LABEL_USER_KEY)
            data = data[data[LABEL_USER_KEY] == self.user_id]
        if LABEL_RECORD_KEY in data.columns:
            self.record_id = row.pop(LABEL_RECORD_KEY)
            data = data[data[LABEL_RECORD_KEY] == self.record_id]
        cycles = get_cycles_from_segment(data)  # Question: Do I have segments here? can't find anything on that in the paper?
        self.selected_cycle = math.floor(random.random() * len(cycles))
        self.template = cycles.iloc[self.selected_cycle]
        if LABEL_USER_KEY in self.template.columns:
            del self.template[LABEL_USER_KEY]
        if LABEL_RECORD_KEY in self.template.columns:
            del self.template[LABEL_RECORD_KEY]
        return self

    def _transform(self, data):
        assert self.template is not None
        return data.apply(self.calculate_distance)

    def calculate_distance(self, data_series):
        from dtaidistance import dtw

        name = data_series.name
        reference_series = self.get_reference(name)

        return dtw.distance(reference_series.to_numpy(), data_series.to_numpy())

    def get_reference(self, name):
        assert self.template is not None
        return self.template[name]


def test_dynamic_time_warping_calculator():
    from src.core.constants import TEST_DATA
    import random
    random.seed(constants.RANDOM_SEED)
    data = TEST_DATA.copy(deep=True)
    calculator = NestedDataFrameTransformer(DynamicTimeWarpingCalculator())
    calculator.fit(data)
    result = calculator.transform(data)
    assert isinstance(result, pd.DataFrame)
    assert_columns(data, result)
    for df in result[DATA_FRAMES_KEY]:
        # noinspection PyUnresolvedReferences
        assert_columns(df, calculator.transformer.template, calculator.transformer.feature_name)
    assert len(data) == len(result)
    assert data.equals(TEST_DATA)


def assert_columns(data1, data2, feature_string=None):
    assert len(data1.columns) == len(data2.columns)
    for column in data2.columns:
        if feature_string:
            assert "{}_{}".format(feature_string, column) in data1.columns
        else:
            assert column in data1.columns


# test helper methods
def get_testing_data_frame():
    df1 = pd.DataFrame(data=[[1, 2, 0], [4, 1, 0], [7, 0, 0], [10, -1, 0], [13, -2, 0.1]], columns=["sensor1_x", "sensor1_y", "sensor1_z"])
    df2 = pd.DataFrame(data=[[1, 2, 3, 0], [4, 5, 6, 1], [7, 8, 9, 0], [10, 11, 12, 1], [13, 14, 15, 0]],
                       columns=["sensor1_x", "sensor1_y", "sensor2_z", "sensor2_y"])
    return pd.DataFrame(data=[[df1, "1"], [df2, "2"]], columns=["data_frames", "label_user"])


def get_cycles_from_segment(data, return_as_list=False):
    cycle_start_indices = np.where(data.index == 0)[0]
    cycles = list()
    for index in range(len(cycle_start_indices)):
        if index < len(cycle_start_indices) - 1:
            cycle = data.iloc[cycle_start_indices.item(index):cycle_start_indices[index + 1]]
        else:
            cycle = data.iloc[cycle_start_indices.item(index):]
        cycles.append(cycle)

    if return_as_list:
        return cycles

    return pd.Series(cycles)
