import pandas as pd

from src.core import constants
from src.core.base_classes import SafeTransformer
from src.core.error_handling.exceptions import IncorrectInputTypeException
from src.core.utility.interpolation_helper import correct_sampling_frequency_by_interpolation


class SensorSynchronizer(SafeTransformer):
    """
    Aligns values across different sensors (time-wise) and uses interpolation for a stable frequency.
    """

    def __init__(self, method="linear", reference_sensor="accelerometer", listening_rate=20, frequency=None, old_listening_rate=20,
                 old_frequency=None, **kwargs):
        """
        Parameters
        ----------
        method : str, default="linear"
            Interpolation method to use.
        reference_sensor : str, default="accelerometer"
            Sensor to use as reference for the synchronization.
        listening_rate : float, default=20
            Number of milliseconds between two time events; it is used for sample frequency correction, pretending that sensor events are called
            consistent in a interval of e.g. 20 milliseconds.
        frequency : int, default=None
            Frequency in Hz or values per second to use for the interpolation. Will we transformed into listening_rate.
        """
        self.method = method
        self.reference_sensor = reference_sensor
        if frequency is not None:
            self.frequency = frequency
            self.listening_rate = 1000 / frequency
        else:
            self.listening_rate = listening_rate
            self.frequency = 1000 / listening_rate

        if old_frequency is not None:
            self.old_frequency = old_frequency
            self.old_listening_rate = 1000 / old_frequency
        else:
            self.old_listening_rate = old_listening_rate
            self.old_frequency = 1000 / old_listening_rate

        super().__init__(**kwargs)

    def fit(self, data_frame, y=None, **kwargs):
        if not isinstance(data_frame, pd.DataFrame):
            raise IncorrectInputTypeException(data_frame, pd.DataFrame)
        return super().fit(data_frame, y)

    def _transform(self, data_frame):
        if not isinstance(data_frame, pd.DataFrame):
            raise IncorrectInputTypeException(data_frame, pd.DataFrame)

        time_unit = 'ms'

        if not isinstance(data_frame.index, pd.TimedeltaIndex):
            if self.old_listening_rate is None:
                raise Exception("Invalid Input")
            start = f"{data_frame.index[0]} {'ms'}"
            end = f"{data_frame.index[-1] * self.old_listening_rate} {time_unit}"
            time_delta = pd.Timedelta(value=self.old_listening_rate, unit=time_unit)
            time_delta_index = pd.timedelta_range(start=start, end=end, freq=time_delta)
            data_frame.index = time_delta_index

        # interpolate reference sensor
        _regex = "{sensor_name}_{dimension}".format(sensor_name=self.reference_sensor, dimension=constants.DIMENSIONS_KEY_LIST)
        reference_sensor_data = data_frame.filter(regex=_regex, axis=1)
        interpolated_reference_sensor = correct_sampling_frequency_by_interpolation(data_frame=reference_sensor_data.dropna(how='all'),
                                                                                    interpolation_method=self.method,
                                                                                    listening_rate=self.listening_rate,
                                                                                    time_unit=time_unit, old_listening_rate=self.old_listening_rate)
        interpolation_data_list = [interpolated_reference_sensor]

        # interpolate other sensors
        for sensor in data_frame.filter(regex=".*_(x|0)", axis=1).columns:
            sensor = sensor.split("_")[0]
            if sensor == self.reference_sensor:
                continue

            _regex = "{sensor_name}_{dimension}".format(sensor_name=sensor, dimension=constants.DIMENSIONS_KEY_LIST)
            sensor_data = data_frame.filter(regex=_regex, axis=1)
            interpolated_sensor_data = correct_sampling_frequency_by_interpolation(data_frame=sensor_data.dropna(how='all'),
                                                                                   interpolation_method=self.method,
                                                                                   listening_rate=self.listening_rate,
                                                                                   time_unit='ms',
                                                                                   start=interpolated_reference_sensor.index[0],
                                                                                   end=interpolated_reference_sensor.index[-1],
                                                                                   old_listening_rate=self.old_listening_rate)
            interpolation_data_list.append(interpolated_sensor_data)
        return pd.concat(interpolation_data_list, axis=1)
