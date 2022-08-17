import numpy as np
import pandas as pd

from src.core.base_classes import FeatureCalculator, NestedDataFrameTransformer
from src.core.constants import DATA_FRAMES_KEY, TEST_DATA
from src.core.error_handling.exceptions import IncorrectInputTypeException


class MagnitudeCalculator(FeatureCalculator):
    KEY = "magnitude"

    def __init__(self, sensor_names, append_to_data_frame=False, **kwargs):
        self.sensor_names = sensor_names if isinstance(sensor_names, list) else [sensor_names] if isinstance(sensor_names, str) else None
        if sensor_names is None:
            raise IncorrectInputTypeException("Parameter 'sensor_names' must be 'str' or 'list-like'")
        self.append_to_data_frame = append_to_data_frame
        super().__init__(self.KEY, **kwargs)

    def fit(self, data_frame, y=None, **kwargs):
        if not isinstance(data_frame, pd.DataFrame):
            raise IncorrectInputTypeException(data_frame, pd.DataFrame)
        return super().fit(data_frame, y, **kwargs)

    def _transform(self, data_frame):
        if not isinstance(data_frame, pd.DataFrame):
            raise IncorrectInputTypeException(data_frame, pd.DataFrame)

        data_frame = data_frame.copy(deep=True)
        for single_sensor in self.sensor_names:
            data_frame[single_sensor] = calculate_euclidean_magnitude_for_sensor_name(data_frame, single_sensor)
        if self.append_to_data_frame:
            return data_frame
        else:
            # return just magnitude columns
            return data_frame[self.sensor_names]


def calculate_euclidean_magnitude_for_sensor_name(data_frame, sensor_name):
    """
    Apply euclidean calculation to a pandas data frame for one specific sensor name.

    Parameters
    ----------
    data_frame: pd.DataFrame
        column-wise data frame, having timestamps as rows and sensor dimensions as columns
    sensor_name: string
        Name of sensor, e.g. "accelerometer"

    Returns
    -------
    Array of magnitude values
    """
    sensor_data_frame = data_frame.filter(regex=("{}_*".format(sensor_name)), axis=1)

    return calculate_euclidean_magnitude_for_data_frame(data_frame=sensor_data_frame)


def calculate_euclidean_magnitude_for_data_frame(data_frame):
    """
    Apply euclidean calculation to a pandas data frame. Using the apply function to data frames rows,
    where sensor's dimensions are stored column-wise.

    Parameters
    ----------
    data_frame: pd.DataFrame
        column-wise data frame, having timestamps as rows and sensor dimensions as columns

    Returns
    -------
    Array of magnitude values
    """
    return data_frame.apply(lambda data_frame_row: calculate_euclidean_magnitude(data_frame_row), axis=1)


def calculate_euclidean_magnitude(vector):
    """
    A vector x in an n-dimensional Euclidean space can be defined as an ordered list of n real
    numbers (the Cartesian coordinates of P): x = [x1, x2, ..., xn].
    Its magnitude (also known as vector length) is most commonly defined
    as its Euclidean norm (or Euclidean length): sqrt(x**2 + y**2 + z **2)

    Parameters
    ----------
    vector: ndarray
        Contain dimensional data, e.g. sensor's actual_data_frame, Y, Z axis

    Returns
    -------
    euclidean magnitude: int
    """
    squared_vector = np.power(vector, 2)
    return np.sqrt(sum(squared_vector))


def test_magnitude_calculator():
    input_data = TEST_DATA
    sensor = "accelerometer"
    for df in input_data[DATA_FRAMES_KEY]:
        df.columns = [sensor + "_" + column for column in df.columns]

    calculator = NestedDataFrameTransformer(MagnitudeCalculator(sensor_names=sensor))
    transformed_data = calculator.transform(input_data)

    assert transformed_data.shape == (3, 2)
    data_frames = transformed_data[DATA_FRAMES_KEY]
    assert np.allclose(data_frames[0].values, data_frames[1].values)
    assert np.allclose(data_frames[0].values.T, [3.74166, 8.77496, 13.92839, 19.10497, 24.28992], 0.0001)
    assert np.allclose(data_frames[2].values.T, [.374166, .877496, 1.392839, .1910497, .2428992], 0.0001)
    assert data_frames[0].columns == data_frames[1].columns == data_frames[2].columns == [MagnitudeCalculator.KEY + "_" + sensor]
