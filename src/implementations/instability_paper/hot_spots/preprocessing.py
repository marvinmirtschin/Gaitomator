import pandas as pd

from src.core.base_classes import SafeTransformer
from src.core.error_handling.exceptions import MissingSensorDataException
from src.add_ins.processing.features import calculate_euclidean_magnitude_for_data_frame
from src.add_ins.processing.features.coordinate_transformation import transform_acceleration_vector_row


class SensorInputTransformer(SafeTransformer):
    """
    Recorded sensor data input in the paper 'On the Instability of Sensor Orientation in Gait Verification on Mobile Phone' by Hoang et al is
    transformed to counter the dissimilarity in gait signals of different walks caused by the user walking in different directions. For that,
    the combined x and y axis data as well as the magnitude of the input data is utilized in combination with the z-axis input.
    """

    def __init__(self, sensor_axis_names=None, **kwargs):
        if not sensor_axis_names:
            sensor_axis_names = ["accelerometer_x", "accelerometer_y", "accelerometer_z"]
        if len(sensor_axis_names) != 3:
            raise Exception("This method accepts exactly 3 axis to transform the data but {} were given.".format(len(sensor_axis_names)))
        self.sensor_axis_names = sensor_axis_names
        super().__init__(**kwargs)

    def _transform(self, data):
        mag_xy = calculate_euclidean_magnitude_for_data_frame(data[self.sensor_axis_names[:-1]])
        mag = calculate_euclidean_magnitude_for_data_frame(data)
        z_axis = data[self.sensor_axis_names[-1]]
        df = pd.concat([z_axis, mag_xy, mag], axis=1)
        df.columns = ["accelerometer_z", "magnitude_xy", "magnitude_xyz"]
        return df


class CoordinateTransformer(SafeTransformer):

    def _transform(self, data: pd.DataFrame):
        try:
            transformed_df = data.apply(lambda data_frame_row: transform_acceleration_vector_row(*data_frame_row), axis=1)
        except TypeError as e:
            raise MissingSensorDataException(e)

        return pd.DataFrame(transformed_df.values.tolist(), index=data.index, columns=["accelerometer_x", "accelerometer_y", "accelerometer_z"])
