import os
from statistics import mean

import pandas as pd

from src.core.constants import DATA_FRAMES_KEY
from src.core.data_access.wisdm.constants import *
from src.core.error_handling.exceptions import InvalidConfigurationException
from src.core.utility import file_handling, time_unit
from src.core.utility.data_frame_transformer import shrink_data_frame
from src.core.utility.time_unit import SECONDS

# TODO: Timestamp column in raw seems to be broken
#   contains constant values and singular increasing values as well as correct samples

ACTIVITIES = [
    "Walking",
    "Jogging",
    "Sitting",
    "Standing",
    "Upstairs",
    "Downstairs"
]

RELEVANT_ACTIVITIES = [
    "Walking",
    "Upstairs",
    "Downstairs"
]


def get_resource_directory(*, version=2):
    directory = file_handling.get_data_directory()
    if version == 2:
        return os.path.join(directory, WISDM, WISDM_V2)
    elif version == 1:
        return os.path.join(directory, WISDM, WISDM_V1)
    else:
        raise FileNotFoundError("No file found in WISDM for version {}".format(version))


def get_data(_type=TYPE_RAW, *, user_ids=None, sensors=None, activities=None, features=None, version=2, **kwargs):
    if sensors is not None and not sensors == "accelerometer" and not ("accelerometer" in sensors and len(sensors) == 1):
        raise InvalidConfigurationException(f"Data set only contains data for sensor accelerometer. You requested: {sensors}")
    if _type == TYPE_RAW:
        return _get_raw_data(user_ids=user_ids, activities=activities, version=version)
    elif _type == TYPE_TRANSFORMED:
        return _get_transformed_data(user_ids=user_ids, activities=activities, features=features, version=version)
    else:
        raise InvalidConfigurationException(f"Unknown type {_type} used to get data from WISDM data set. Please use one of the following: {TYPES}")


def _get_raw_data(*, user_ids=None, activities=None, version=2):
    resource_directory = get_resource_directory(version=version)
    if version == 2:
        file_name = os.path.join(resource_directory, get_raw_file(WISDM_V2))
    elif version == 1:
        file_name = os.path.join(resource_directory, get_raw_file(WISDM_V1))
    else:
        raise FileNotFoundError("No file found in WISDM for version {}".format(version))
    data_frame = pd.read_csv(file_name, names=RAW_COLUMNS)
    if user_ids:
        data_frame = data_frame[data_frame[LABEL_USER_KEY].isin(user_ids)]
    if activities:
        data_frame = data_frame[data_frame[LABEL_ACTIVITY_KEY].isin(activities)]

    columns = [LABEL_USER_KEY, LABEL_ACTIVITY_KEY] + list((set(data_frame.columns).difference([LABEL_USER_KEY, LABEL_ACTIVITY_KEY])))
    data_frame = data_frame[columns]
    return data_frame.reset_index(drop=True)


def _get_transformed_data(*, user_ids=None, activities=None, features=None, version=2):
    resource_directory = get_resource_directory(version=version)
    if version == 2:
        file_name = os.path.join(resource_directory, get_transformed_file(WISDM_V2))
        data_frame = pd.read_csv(file_name, names=TRANSFORMED_COLUMNS, skiprows=49)
    elif version == 1:
        file_name = os.path.join(resource_directory, get_transformed_file(WISDM_V1))
        data_frame = pd.read_csv(file_name, names=[UNIQUE_ID] + TRANSFORMED_COLUMNS, skiprows=50, usecols=TRANSFORMED_COLUMNS)
    else:
        raise FileNotFoundError("No file found in WISDM for version {}".format(version))
    if user_ids:
        data_frame = data_frame[data_frame[LABEL_USER_KEY].isin(user_ids)]
    if activities:
        data_frame = data_frame[data_frame[LABEL_ACTIVITY_KEY].isin(activities)]
    if features:
        if LABEL_USER_KEY not in features:
            features.append(LABEL_USER_KEY)
        if LABEL_ACTIVITY_KEY not in features:
            features.append(LABEL_ACTIVITY_KEY)
        data_frame = data_frame[features]

    columns = [LABEL_USER_KEY, LABEL_ACTIVITY_KEY] + list((set(data_frame.columns).difference([LABEL_USER_KEY, LABEL_ACTIVITY_KEY])))
    data_frame = data_frame[columns]
    return data_frame.reset_index(drop=True)


def get_raw_file(file_name):
    return file_name + "_raw.txt"


def get_transformed_file(file_name):
    return file_name + "_transformed.arff"


# TESTS
def test_data_reading_raw_v1():
    expected_users = {33, 17}
    expected_activities = {"Walking", "Upstairs"}
    data_frame = _get_raw_data(user_ids=expected_users, activities=expected_activities, version=1)
    assert_raw(data_frame, expected_users, expected_activities)


def test_data_reading_raw_v2():
    expected_users = {599, 1793}
    expected_activities = {"Walking", "Stairs"}
    data_frame = _get_raw_data(user_ids=expected_users, activities=expected_activities, version=2)
    assert_raw(data_frame, expected_users, expected_activities)


def test_data_reading_transformed_v1():
    expected_users = {33, 17}
    expected_activities = {"Walking", "Upstairs"}
    expected_features = ["X0", "Y0", "Z0"]
    data_frame = _get_transformed_data(user_ids=expected_users, activities=expected_activities, features=expected_features, version=1)
    assert_transformed(data_frame, expected_users, expected_activities, expected_features)


def test_data_reading_transformed_v2():
    expected_users = {1104, 1793}
    expected_activities = {"Walking", "Stairs"}
    expected_features = ["X0", "Y0", "Z0"]
    data_frame = _get_transformed_data(user_ids=expected_users, activities=expected_activities, features=expected_features, version=2)
    assert_transformed(data_frame, expected_users, expected_activities, expected_features)


def assert_raw(data_frame, expected_users, expected_activities):
    actual_users = set(data_frame[LABEL_USER_KEY])
    assert actual_users == expected_users
    actual_activities = set(data_frame[LABEL_ACTIVITY_KEY])
    assert actual_activities == expected_activities


def assert_transformed(data_frame, expected_users, expected_activities, expected_features):
    actual_users = set(data_frame[LABEL_USER_KEY])
    assert actual_users == expected_users
    actual_activities = set(data_frame[LABEL_ACTIVITY_KEY])
    assert actual_activities == expected_activities
    columns = set(data_frame.columns)
    assert columns == set([LABEL_ACTIVITY_KEY, LABEL_USER_KEY] + expected_features)


def test_broken_timestamp_column():
    data = get_data()
    df = data[[LABEL_USER_KEY, LABEL_ACTIVITY_KEY, LABEL_TIMESTAMP_KEY]]
    # del data
    label_df = df[[LABEL_USER_KEY, LABEL_ACTIVITY_KEY]]
    label_df.drop_duplicates(inplace=True)
    label_df = label_df[label_df[LABEL_ACTIVITY_KEY] == "Walking"]
    problems = list()
    for index, row in label_df.iterrows():
        user_id, activity = row
        current_df = data[df[LABEL_USER_KEY] == user_id]
        current_df = current_df[current_df[LABEL_ACTIVITY_KEY] == activity]
        timestamps = current_df[LABEL_TIMESTAMP_KEY].values
        x = timestamps[1:] - timestamps[:-1]
        y = [40 < value < 60 for value in x]
        # TODO: look also for empty spaces in index
        if not all(y):
            problems.append(row)


def test_print_v1_report():
    data: pd.DataFrame = get_data(_type=TYPE_RAW, version=1)
    data.drop(["acceleration_x", "acceleration_y", "acceleration_z"], axis=1, inplace=True)
    _print_report(data, time_unit.NANOSECONDS)


def test_print_v2_report():
    data: pd.DataFrame = get_data(_type=TYPE_RAW, version=2)
    data.drop(["acceleration_x", "acceleration_y", "acceleration_z"], axis=1, inplace=True)
    _print_report(data, time_unit.MILLISECONDS)


def _print_report(data, time_u: time_unit.MTime):
    # TODO: observe if v2 also needs special handling
    number_of_users = len(data[LABEL_USER_KEY].unique())
    number_values = len(data)
    number_of_walking_values = len(data[data[LABEL_ACTIVITY_KEY] == "Walking"])
    shrunken_dfs = shrink_data_frame(data[data[LABEL_ACTIVITY_KEY] == "Walking"])
    durations = list()
    frequencies = list()
    print()
    for j, row in shrunken_dfs.iterrows():
        print("---------------")
        df = row[DATA_FRAMES_KEY]
        df = df.sort_values(by=['timestamp'])

        time_diffs = df["timestamp"].iloc[1:].values - df["timestamp"].iloc[:-1].values
        x = abs(time_diffs)

        # FIXME: this needs to be fixed for general usage
        indices = list()
        for i, v in enumerate(x):
            # According to the docs the sample frequency should be around 20 Hz (50 ms between) and the values are given in nanoseconds
            # -> therefore we choose the duration of 1 second to be "unusual" and use it as a mark for a new recording
            if v > time_u.from_unit(1, SECONDS):
                indices.append(i + 1)

        for i in range(len(indices) + 1):
            start = 0 if i == 0 else indices[i - 1]
            end = indices[i] if i < len(indices) else len(df)
            cut_df = df.iloc[start:end]

            if len(cut_df) <= 1:
                continue

            duration = time_u.to_seconds(cut_df["timestamp"].iloc[-1] - cut_df["timestamp"].iloc[0])
            frequency = len(cut_df) / duration

            if frequency > 50:
                # Note: Sometimes it counts straight up
                continue  # Investigation

            durations.append(duration)
            frequencies.append(frequency)

            print(f"Got {len(cut_df)} data points over {duration} s ({frequency} Hz)")
    print("==============")
    print(f"From {number_values} {number_of_walking_values} are walking")
    print(
        f"Got data for {number_of_users} people with a duration of about {time_unit.SECONDS.to_minutes(sum(durations))} min and an average "
        f"frequency of "
        f"{mean(frequencies)}")
