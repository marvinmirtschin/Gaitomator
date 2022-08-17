import copy
import os
import random

import pandas as pd

import src.core.constants as constants
import src.core.utility.file_handling as file_handling
from src.core.error_handling.exceptions import InvalidConfigurationException
from src.core.utility import expand_dict_to_data_frame
from src.core.utility.data_frame_transformer import expand_data_frame

LINEAR_ACCELERATION = "Linear Acceleration Sensor"
ROTATION_MATRIX = "Rotation Matrix"
ACCELEROMETER = "BMA 150 3-axis Accelerometer"

SENSORS = [LINEAR_ACCELERATION, ROTATION_MATRIX, ACCELEROMETER]


# Note: One file was removed from the data set: user ID16 had only 1 recording for id 16210 -> only rotation matrix, therefore it was unusable
def get_data(max_number_of_users=None, max_number_of_recordings_per_user=None, user_ids=None, genders=None, record_ids=None,
             sensors=None, recording_numbers=None, **kwargs):
    data_frame_map = create_paper_data_frames(max_number_of_users, max_number_of_recordings_per_user, user_ids, genders, record_ids, sensors,
                                              recording_numbers)
    return expand_dict_to_data_frame(data_frame_map)


def create_paper_data_frames(max_number_of_users=None, max_number_of_recordings_per_user=None, user_ids=None, genders=None, record_ids=None,
                             sensors=None, recording_numbers=None):
    file_names = get_files_from_paper(user_ids=user_ids, genders=genders, record_ids=record_ids, sensors=sensors, recording_numbers=recording_numbers)
    return _create_data_frames(file_names, max_number_of_users=max_number_of_users,
                               max_number_of_recordings_per_user=max_number_of_recordings_per_user)


def get_files_from_paper(user_ids=None, genders=None, record_ids=None, sensors=None, recording_numbers=None):
    return _get_filtered_files(get_all_file_names_for_paper(), user_ids=user_ids, genders=genders, record_ids=record_ids, sensors=sensors,
                               recording_numbers=recording_numbers)


def get_all_file_names_for_paper():
    return file_handling.get_file_names_in_directory_for_pattern(get_data_directory(), "*.txt")


def get_data_directory():
    directory = os.path.join(file_handling.get_data_directory(), "instability_paper")
    return directory


def _create_data_frames(file_names, max_number_of_users=None, max_number_of_recordings_per_user=None):
    data_frames = dict()

    user_counter = 0
    recording_counter = dict()

    # Remember to seed for predictable results
    random.shuffle(file_names)

    for file_name in file_names:
        user_id, _, record_id, sensor, _ = file_name.split("/")[-1][:-4].split("_")
        if user_id not in data_frames:
            if max_number_of_users and user_counter >= max_number_of_users:
                continue
            data_frames[user_id] = dict()
            user_counter += 1
            recording_counter[user_id] = 0

        df = pd.read_csv(file_name, header=None)

        if "Rotation Matrix" in sensor:
            df.columns = ["timestamp", "rotation_0", "rotation_1", "rotation_2", "rotation_3", "rotation_4", "rotation_5", "rotation_6", "rotation_7",
                          "rotation_8"]
        elif "Accelerometer" in sensor:
            df.columns = ["timestamp", "accelerometer_x", "accelerometer_y", "accelerometer_z"]
        else:
            df.columns = ["timestamp", "linearAcceleration_x", "linearAcceleration_y", "linearAcceleration_z"]

        df = __set_time_delta_as_index(df, timestamp_unit='ms')

        if record_id not in data_frames[user_id]:
            if max_number_of_recordings_per_user and recording_counter[user_id] >= max_number_of_recordings_per_user:
                continue
            data_frames[user_id][record_id] = df
            recording_counter[user_id] += 1
        else:
            data_frames[user_id][record_id] = df.merge(data_frames[user_id][record_id], how='outer', left_index=True, right_index=True)

    for user_id in data_frames:
        for record_id in data_frames[user_id]:
            data_frames[user_id][record_id] = _reorder_columns(data_frames[user_id][record_id])

    return data_frames


def _reorder_columns(data_frame):
    reordered_columns = list(data_frame.columns)
    reordered_columns.sort()
    reordered_data_frame = data_frame[reordered_columns]
    return reordered_data_frame


def _get_filtered_files(file_names, user_ids=None, genders=None, record_ids=None, sensors=None, recording_numbers=None):
    filtered_files = copy.copy(file_names)

    for file_name in file_names:
        short_file_name = file_name.split("/")[-1]
        short_file_name = short_file_name[:-4]

        user_id, gender, record_id, sensor, recording_number = short_file_name.split("_")
        sensor = sensor.split(']')[-1]

        if sensors is not None:
            sensors = get_mapped_sensor_names(sensors)

        if user_ids is not None and user_id not in user_ids:
            filtered_files.remove(file_name)
        elif genders is not None and gender not in genders:
            filtered_files.remove(file_name)
        elif record_ids is not None and record_id not in record_ids:
            filtered_files.remove(file_name)
        elif sensors is not None and sensor not in sensors:
            filtered_files.remove(file_name)
        elif recording_numbers is not None and recording_number not in recording_numbers:
            filtered_files.remove(file_name)

    return filtered_files


def get_mapped_sensor_names(sensors):
    """
    Map the commonly used sensors names to the once used in this data set.
    Parameters
    ----------
    sensors: list-like
        Commonly used names for sensors.

    Returns
    -------
        Requested sensors as used in this data set.
    """
    mapped_sensors = list()
    for sensor in sensors:
        if "lin" in sensor.lower() or "acceleration" in sensor.lower():
            mapped_sensors.append(LINEAR_ACCELERATION)
        elif "rot" in sensor.lower():
            mapped_sensors.append(ROTATION_MATRIX)
        elif "acc" in sensor.lower():
            mapped_sensors.append(ACCELEROMETER)
        else:
            raise InvalidConfigurationException(f"Unable to match {sensor} to one of the following sensors: {SENSORS}.")
    return mapped_sensors


# @DeprecationWarning  # usage of TimedeltaIndex is not preferred
def __set_time_delta_as_index(data_frame, timestamp_unit="ms"):
    """
    Set timestamp column, converted to timeDeltaIndex, as index.
    Parameters
    ----------
    data_frame : pd.DataFrame
    timestamp_unit: string, default ‘ms’
        With unit=’ms’ and origin=’unix’ (the default),
        this would calculate the number of milliseconds to the unix epoch start.

    Returns
    -------
    actual_data_frame : pd.DataFrame
        actual_data_frame, having a using TimeDelta as index
    """
    timestamp_to_timedelta = pd.to_timedelta(data_frame.loc[:, constants.TIMESTAMP_KEY], unit=timestamp_unit)
    time_delta_index = pd.TimedeltaIndex(timestamp_to_timedelta)
    time_delta_index.name = None
    data_frame = pd.DataFrame(data_frame.values, index=time_delta_index, columns=data_frame.columns)
    data_frame.drop(constants.TIMESTAMP_KEY, axis=1, inplace=True)
    return data_frame


def test_clean_file_names():
    """
    Rename files not matching with the parsing values template.
    """
    file_names = get_all_file_names_for_paper()
    for file_name in file_names:
        parsing_values = file_name.split("/")[-1][:-4].split("_")

        if len(parsing_values) == 6:
            if "Nexus_One" in file_name:
                new_file_name = file_name.replace("Nexus_One", "NexusOne")
                os.rename(file_name, new_file_name)


def test_get_data():
    df = get_data()
    expanded_df = expand_data_frame(df)
    assert df.shape == (721, 3)
    assert expanded_df.shape == (1660991, 17)
    assert len(df[constants.LABEL_USER_KEY].unique()) == 38


def test_file_paths():
    data_directory = get_data_directory()
    assert 2163 == len(file_handling.get_file_names_in_directory_for_pattern(data_directory, "*.txt"))
