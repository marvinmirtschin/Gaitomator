#  we captured 3-axial linear acceleration and 3-axial angular velocity at a constant rate of 50Hz
# - The units used for the accelerations (total and body) are 'g's (gravity of earth -> 9.80665 m/seg2).
# - The gyroscope units are rad/seg.

# The sensor signals (accelerometer and gyroscope) were pre-processed by applying noise filters and then sampled in fixed-width sliding windows of
# 2.56 sec and 50% overlap (128 readings/window) -> this is not relevant
# The sensor acceleration signal, which has gravitational and body motion components, was separated using a Butterworth low-pass filter into body
# acceleration and gravity. The gravitational force is assumed to have only low frequency components, therefore a filter with 0.3 Hz cutoff
# frequency was used

import os
from itertools import chain

import pandas as pd

from src.core.constants import LABEL_ACTIVITY_KEY, LABEL_USER_KEY
from src.core.error_handling.exceptions import InvalidConfigurationException
from src.core.utility import file_handling

UCI_HAR_DATA_FOLDER = "UCI HAR Dataset"
TEST = "test"
TRAIN = "train"
INERTIAL_SIGNALS = "Inertial Signals"

ACTIVITIES = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING"
}

WALKING_ACTIVITIES = [1, 2, 3]

LINEAR_ACCELERATION = "body_acc"
GYROSCOPE = "body_gyro"
ACCELEROMETER = "total_acc"

TYPES = [TEST, TRAIN]
SENSORS = [LINEAR_ACCELERATION, GYROSCOPE, ACCELEROMETER]
DIMENSIONS = ["x", "y", "z"]


def get_resource_directory():
    directory = file_handling.get_data_directory()
    return os.path.join(directory, UCI_HAR_DATA_FOLDER)


def get_data(*, users=None, activities=None, types=None, sensors=None, dimensions=None, **kwargs):
    """
    Load relevant activities from y_train.txt, y_test.txt into LABEL_ACTIVITY_KEY, then load matching users from subject_train.txt,
    subject_test.txt into LABEL_USER_KEY and finally load relevant data (Accelerometer - total_acc_?_test/train.txt, Linear Acceleration -
    body_acc_?_test/train, Gyroscope - body_gyro_?_test/train).

    Parameters
    ----------
    users: list-like, default=None
        Filter for users.
    activities: list-like, default=None
        Filter for activities.
    types: list-like, default=None
        Filter for types.
    sensors: list-like, default=None
        Filter for sensors.
    dimensions: list-like, default=None
        Filter for dimensions.

    Returns
    -------
    result_df: pd.DataFrame
        DataFrame containing the data matching the given filter criteria.
    """
    try:
        activities = set_and_assert_parameter(activities, WALKING_ACTIVITIES)
    except AssertionError:
        activities = map_activities(activities)
    types = set_and_assert_parameter(types, TYPES)
    try:
        sensors = set_and_assert_parameter(sensors, SENSORS)
    except AssertionError:
        sensors = map_sensors(sensors)
    dimensions = set_and_assert_parameter(dimensions, DIMENSIONS)

    dfs, skip_rows = get_activity_df(types, activities)
    dfs, skip_rows = get_user_data_frames(types, users, dfs, skip_rows)

    # Each data row will be extended into 64 rows due to structure; Therefore we duplicate existing rows for matching
    dfs = multiply_rows(types, dfs)

    dfs = read_data(types, sensors, dimensions, dfs, skip_rows)

    # merge types
    result_df = pd.concat(dfs.values())
    # Order by user and activity; this preserves value ordering
    result_df = result_df.sort_values(by=[LABEL_USER_KEY, LABEL_ACTIVITY_KEY])
    result_df.reset_index(inplace=True, drop=True)

    return result_df


def get_activity_df(types, activities):
    skip_rows = dict()
    dfs = dict()
    # get activity file and filter for relevant activities
    for _type in types:
        activity_file = os.path.join(get_resource_directory(), _type, f"y_{_type}.txt")
        df = pd.read_csv(activity_file, names=[LABEL_ACTIVITY_KEY])
        rows = df.index
        df = df[[activity in activities for activity in df[LABEL_ACTIVITY_KEY].values]]

        # Remember which rows were removed by filtering to avoid unnecessary data loading
        used_rows = df.index
        skipped_rows = [index + 1 for index in rows if index not in used_rows]
        skip_rows[_type] = skipped_rows

        del rows
        del used_rows
        del skipped_rows

        df.reset_index(drop=True, inplace=True)
        dfs[_type] = df
        del df
        del activity_file

    return dfs, skip_rows


def get_user_data_frames(types, users, dfs, skip_rows):
    for _type in types:
        user_file = os.path.join(get_resource_directory(), _type, f"subject_{_type}.txt")
        df = pd.read_csv(user_file, names=[LABEL_USER_KEY], skiprows=skip_rows[_type])

        if users is not None:
            # remove rows not belonging to requested users
            rows = df.index
            df = df[[user in users for user in df[LABEL_USER_KEY].values]]

            # Remember which rows were removed by filtering to avoid unnecessary data loading
            used_rows = df.index
            skipped_rows = [index + 1 for index in rows if index not in used_rows]
            skip_rows[_type] += skipped_rows

            del rows
            del used_rows
            del skipped_rows

        dfs[_type] = dfs[_type].join(df, how="inner")
        del df
        del user_file
    return dfs, skip_rows


def read_data(types, sensors, dimensions, dfs, skip_rows):
    for _type in types:
        for sensor in sensors:
            for dimension in dimensions:
                data_file = os.path.join(get_resource_directory(), _type, INERTIAL_SIGNALS, f"{sensor}_{dimension}_{_type}.txt")
                # Values are given in 128-dim vector with a 50 % overlap -> use range to remove the overlapping
                df = pd.read_csv(data_file, delim_whitespace=True, header=None, usecols=range(0, 64), skiprows=skip_rows[_type])
                # Reshape data into single readings instead of windows
                sensors_values = df.values.reshape(-1, 1).ravel()
                dfs[_type][f"{get_sensor_column_name(sensor)}_{dimension}"] = pd.Series(sensors_values)

                del data_file
                del df
                del sensors_values
    return dfs


def multiply_rows(types, dfs, multiplier: int = 64):
    # Each data row will be extended into #multiplier rows due to structure; Therefore we duplicate existing rows for matching
    for _type in types:
        new_df = dict()
        old_df = dfs[_type]
        for column in old_df:
            values = old_df[column]
            new_df[column] = chain(*[multiplier * [value] for value in values])

        dfs[_type] = pd.DataFrame(data=new_df)
    return dfs


def map_sensors(sensors):
    mapped_sensors = list()
    for sensor in sensors:
        if "lin" in sensor.lower() or "acceleration" in sensor.lower():
            mapped_sensors.append(LINEAR_ACCELERATION)
        elif "gyr" in sensor.lower():
            mapped_sensors.append(GYROSCOPE)
        elif "acc" in sensor.lower():
            mapped_sensors.append(ACCELEROMETER)
        else:
            raise InvalidConfigurationException(f"Unable to match {sensor} to one of the following sensors: {SENSORS}.")
    return mapped_sensors


def get_sensor_column_name(sensor):
    # TODO: use constants
    if sensor.lower() == LINEAR_ACCELERATION:
        return "linear_acceleration"
    elif sensor.lower() == GYROSCOPE:
        return "gyroscope"
    elif sensor.lower() == ACCELEROMETER:
        return "accelerometer"


def map_activities(activities):
    mapped_activities = list()
    for activity in activities:
        if "upstair" in str(activity).lower():
            mapped_activities.append(2)
        elif "downstair" in str(activity).lower():
            mapped_activities.append(3)
        elif "walking" in str(activity).lower():
            mapped_activities.append(1)
        else:
            raise InvalidConfigurationException(
                f"Unable to match {activity} to one of the following activities: {[(_, ACTIVITIES[_]) for _ in WALKING_ACTIVITIES]}.")
    return mapped_activities


def set_and_assert_parameter(actual_parameter, default_value):
    if actual_parameter is None:
        actual_parameter = default_value
    else:
        assert all(str(value) in str(default_value) for value in actual_parameter)
    return actual_parameter


# TESTS
def test_get_data():
    data_frame = get_data()
    assert data_frame.shape == (299008, 11)
    label_data_frame = data_frame[[LABEL_USER_KEY, LABEL_ACTIVITY_KEY]]
    label_data_frame.drop_duplicates(inplace=True)
    assert label_data_frame.shape == (90, 2)


def test_map_activities():
    import numpy as np
    np.testing.assert_equal(map_activities(["upstair", "downstair", "walking"]), [2, 3, 1])
    np.testing.assert_raises(InvalidConfigurationException, map_activities, ["laying", "sitting"])


def test_map_sensors():
    import numpy as np
    np.testing.assert_equal(map_sensors(["linAcc", "accelerometer", "gyr", "gyroscope", "acc", "linear acceleration"]),
                            [LINEAR_ACCELERATION, ACCELEROMETER, GYROSCOPE, GYROSCOPE, ACCELEROMETER, LINEAR_ACCELERATION])
    np.testing.assert_raises(InvalidConfigurationException, map_sensors, ["gy", "liA"])
