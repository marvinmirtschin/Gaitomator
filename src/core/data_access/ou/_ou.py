# http://www.am.sanken.osaka-u.ac.jp/BiometricDB/InertialGait.html
#
# After simple preprocessing to remove invalid data and extract interested data, we found that the validity of captured data was not equal for all
# subjects, hence our dataset could be optimized with respect to several variation factors. The most important factor is the number of subjects,
# therefore we made the first dataset that included the maximum number of subjects. Meanwhile, the second dataset was maximized with the variations
# of sensor location, ground condition, and sensor type, therefore the number of subjects was sacrificed for these variations.

# TYPE_AUTO
# In the first subset, level walk data of 744 subjects (389 males and 355 females) with ages ranging from 2 to 78 years was captured by the center
# IMUZ. In this subset, two different level-walk sequences for each subject were extracted automatically by using motion trajectory constraint and
# signal autocorrelation.
# Question: Do we want to use data of people below the age of 10 ? Not for now

# TYPE_MANUAL
# In the second subset, variations of ground slope and sensor location were focused. For each subject and each sensor, we extracted two sequences
# for level walk, a sequence for up-slope walk, and a sequence for down-slope walk. In total, we had a sub-subset from 3 IMUZ sensors for 495
# subjects and a sub-subset from the smartphone for 408 subjects. In this subset, the data extraction for each sequence was performed manually by
# synchronizing with a simultaneously captured video.
#
# 2 data sets:
#   - automatically cut walking
#   - manually cut walking, waling up-slope, walking down-slope
#
# Naming convention:
#   auto:
#       - T0_<user_id>_Center_seq<count>.csv, for count in [0, 1]
#   manual:
#       - T0_<user_id>_<activity><count?>.csv, for count in [1, 2] if present
#       - activities: [Walk, SlopeDown, SlopeUp]
#
# Auto data was recorded with just one device (IMU center) while manual was recorded using 4 devices (IMU left, center, right and android device)
import os
from itertools import chain

import pandas as pd

from src.core.constants import DATA_FRAMES_KEY, LABEL_ACTIVITY_KEY, LABEL_RECORD_KEY, LABEL_USER_KEY
from src.core.error_handling.exceptions import InvalidConfigurationException
from src.core.utility import file_handling
from src.core.utility.data_frame_transformer import expand_data_frame

# CONSTANTS

GIVEN_FREQUENCY = 100  # Hz

OU_INERTIAL_GAIT_DATA_FOLDER = "OU-InertialGaitData"
AUTO_DATA_FOLDER = "AutomaticExtractionData_IMUZCenter"
MANUAL_DATA_FOLDER = "ManualExtractionData"

ANDROID_DIRECTORY = "Android"
IMUZ = "IMUZ"
IMUZ_CENTER_DIRECTORY = IMUZ + "Center"
IMUZ_LEFT_DIRECTORY = IMUZ + "Left"
IMUZ_RIGHT_DIRECTORY = IMUZ + "Right"

ACCELEROMETER = "accelerometer"
GYROSCOPE = "gyroscope"
SENSORS = (ACCELEROMETER, GYROSCOPE)

TYPE_AUTO = "auto"
TYPE_MANUAL = "manual"
TYPES = (TYPE_AUTO, TYPE_MANUAL)

LABEL_TYPE_KEY = "label_type"

CENTER = "center"
LEFT = "left"
RIGHT = "right"
ANDROID = "android"
POSITIONS = (CENTER, LEFT, RIGHT, ANDROID)

AGE_0_TO_9 = "Under10"
AGE_10_TO_19 = "Age_group_10-19"
AGE_20_TO_29 = "Age_group_20-29"
AGE_30_TO_39 = "Age_group_30-39"
AGE_40_TO_49 = "Age_group_40-49"
AGE_OVER_50 = "Over50"

# TODO: move and standardize in constants
ACTIVITY_WALKING = "Walk"
ACTIVITY_SLOPE_DOWN = "SlopeDown"
ACTIVITY_SLOPE_UP = "SlopeUp"
ACTIVITIES = (ACTIVITY_WALKING, ACTIVITY_SLOPE_DOWN, ACTIVITY_SLOPE_UP)

AGE_GROUPS = (AGE_0_TO_9, AGE_10_TO_19, AGE_20_TO_29, AGE_30_TO_39, AGE_40_TO_49, AGE_OVER_50)
DEFAULT_AGE_GROUPS = (AGE_10_TO_19, AGE_20_TO_29, AGE_30_TO_39, AGE_40_TO_49, AGE_OVER_50)

FEMALE = "female"
MALE = "male"
GENDERS = (FEMALE, MALE)


# FUNCTIONS


def get_resource_directory():
    directory = file_handling.get_data_directory()
    return os.path.join(directory, OU_INERTIAL_GAIT_DATA_FOLDER)


def get_data(*, types=None, activities=None, position=CENTER, sensors=None, ages=DEFAULT_AGE_GROUPS, skip_data=False, **kwargs):
    # TODO: refactor position to enable multiple locations; leave default as is !
    """
    # Question: should position be a list-like object instead of a single position?
    #   Answer: Data is recorded in parallel therefore they should not be mixed

    Parameters
    ----------
    types : list-like, optional,
        Iterable containing which type of data should be retrieved. See #TYPES.
        # Question: does it make sense to use type as list-like object?
        #   Answer: yes but only for fixed position (center) and only for walking
        # Follow-up Question: does it need to be a fixed position? Each position can be considered a different walk
    activities : list-like, optional,
        Iterable containing the activities which should be retrieved. See #ACTIVITIES.
    position : str, default=CENTER
        Position of the recording device.
    sensors : list-like, optional,
        Iterable containing the sensor names which should be retrieved.
    ages : list-like, default=DEFAULT_AGE_GROUPS
        Iterable containing age range constants (see #AGE_GROUPS) which should be used for the returned data set.
        Note that by default the age group of 0 to 10 is omitted.
    skip_data : bool, default=False
        If true, no data will be loaded. Just for statistical purposes.

    Returns
    -------

    """
    if types is None:
        if (position is None or position is CENTER) and activities is None or (ACTIVITY_WALKING in activities and len(activities) == 1):
            types = TYPES
        else:
            types = (TYPE_MANUAL,)

    if TYPE_AUTO in types:
        assert position is None or CENTER in str(position).lower()
        assert activities is None or (ACTIVITY_WALKING in activities and len(activities) == 1)

    data_frame_list = list()
    record_id_list = list()
    activity_list = list()
    user_id_list = list()
    types_list = list()
    for _type in types:
        file_names = get_file_names(_type, position)

        if activities is not None:
            mapped_activities = map_activities(activities)
            file_names = get_filtered_file_names(file_names, *mapped_activities)

        user_ids_for_age = get_user_ids_for_age(*ages)
        sensor_columns = get_sensor_columns(sensors)

        for file_name in file_names:
            user_id, activity, counter = parse_file_name(file_name)
            if ages != AGE_GROUPS and user_id not in user_ids_for_age:
                # NOTE: missing age for 104, 9443, 58346, 66030, 66134, 159558, 266968, 300939, 301838, 312531, 317026, 319760, 321530, 355444,
                #  364443, 367556, 372127, 416960, 457659, 466450
                continue

            if (str(user_id) == "9547" or str(user_id) == "310005") and activity == ACTIVITY_WALKING and counter == 0 and ACCELEROMETER in sensors:
                # no acceleration data
                continue

            record_id_list.append(counter)
            activity_list.append(activity)
            user_id_list.append(user_id)

            if not skip_data:
                # enable inspection of filtering and stats without reading real data all the time
                df = pd.read_csv(file_name, skiprows=2, usecols=sensor_columns, names=list(get_sensor_names(sensor_columns)))
                data_frame_list.append(df)

        types_list += [_type] * (len(user_id_list) - len(types_list))

    frame = {DATA_FRAMES_KEY: data_frame_list, LABEL_ACTIVITY_KEY: activity_list, LABEL_USER_KEY: user_id_list, LABEL_RECORD_KEY: record_id_list}

    if skip_data:
        del frame[DATA_FRAMES_KEY]
    if len(types) == 2:
        frame[LABEL_TYPE_KEY] = types_list

    result_df = pd.DataFrame(frame)
    result_df.sort_values(by=[LABEL_USER_KEY, LABEL_RECORD_KEY], inplace=True)
    result_df.reset_index(inplace=True, drop=True)
    return result_df


def get_file_names(_type, position):
    if _type == TYPE_MANUAL:
        data_directory = os.path.join(get_resource_directory(), MANUAL_DATA_FOLDER,
                                      get_directory_for_position(position))
    elif _type == TYPE_AUTO:
        data_directory = os.path.join(get_resource_directory(), AUTO_DATA_FOLDER)
    else:
        raise InvalidConfigurationException(f"Unknown type {_type} used. Please use one of the following: {TYPES}.")

    return file_handling.get_file_names_in_directory_for_pattern(data_directory, pattern="T0*.csv")


def map_activities(activities):
    mapped_activities = list()
    for activity in activities:
        if activity.lower() == "walking":
            mapped_activities.append(ACTIVITY_WALKING)
        else:
            mapped_activities.append(activity)
    return mapped_activities


def parse_file_name(file_name):
    # Naming convention:
    #   auto:
    #       - T0_<user_id>_Center_seq<count>.csv, for count in [0, 1]
    #       - only done for walking samples
    #   manual:
    #       - T0_<user_id>_<activity><count?>.csv, for count in [1, 2] if present (only for Walk)
    #       - activities: [Walk, SlopeDown, SlopeUp]

    short_file_name = file_name.split('/')[-1]
    short_file_name = short_file_name[:-4]  # remove file ending
    splits = short_file_name.split('_')
    if len(splits) == 4:
        _, user_id, _, activity = splits
    else:
        _, user_id, activity = splits

    user_id = int(user_id[2:])

    try:
        counter = int(activity[-1])
        activity = activity[:-1]
        if activity == ACTIVITY_WALKING:
            counter -= 1
    except ValueError:
        counter = 0

    if activity == "seq":
        activity = ACTIVITY_WALKING

    return user_id, activity, counter


def get_sensor_names(indices):
    indices.sort()
    for index in indices:
        if index == 0:
            yield ACCELEROMETER + "_x"
        elif index == 1:
            yield ACCELEROMETER + "_y"
        elif index == 2:
            yield ACCELEROMETER + "_z"
        elif index == 3:
            yield GYROSCOPE + "_x"
        elif index == 4:
            yield GYROSCOPE + "_y"
        else:
            yield GYROSCOPE + "_z"


def get_filtered_file_names(files, *sub_strings):
    for file in files:
        for sub_string in sub_strings:
            if sub_string.lower() in file.lower() or (sub_string == ACTIVITY_WALKING and "seq" in file):
                yield file


def get_directory_for_position(position):
    position = str(position).lower()
    if CENTER in position:
        return IMUZ_CENTER_DIRECTORY
    elif LEFT in position:
        return IMUZ_LEFT_DIRECTORY
    elif RIGHT in position:
        return IMUZ_RIGHT_DIRECTORY
    elif ANDROID in position:
        return ANDROID_DIRECTORY
    else:
        raise InvalidConfigurationException(f"Unknown position {position} used. Please use one of the following: {POSITIONS}.")


def get_protocols_directory():
    return os.path.join(get_resource_directory(), "Protocols")


def get_age_file():
    return os.path.join(get_protocols_directory(), "5.7", "Age_group_ID_list.csv")


def get_gender_file():
    return os.path.join(get_protocols_directory(), "5.6", "GenderIDList.txt")


def get_user_ids_for_age(*ages):
    df = pd.read_csv(get_age_file(), skiprows=1, names=AGE_GROUPS)
    df = df[list(ages)]

    values = chain(*df.values.tolist())
    return pd.Series(values).dropna().convert_dtypes().values


def get_age_groups(*ages):
    # Note: Missing age for 104, 9443, 58346, 66030, 66134, 159558, 266968, 300939, 301838, 312531, 317026, 319760, 321530, 355444, 364443, 367556,
    #  372127, 416960, 457659, 466450
    df = pd.read_csv(get_age_file(), skiprows=1, names=AGE_GROUPS)
    return get_values_per_user_id(df, *ages)


def get_genders(*genders):
    # Note: Missing gender for 4, 9443, 58346, 66030, 66134, 159558, 266968, 271747, 275447, 276970, 300315, 300939, 301838, 308235, 309342,
    #  310317, 312531, 314641, 317026, 319760, 321530, 325126, 326025, 350117, 350221, 354649, 354857, 355444, 355756, 356863, 357138, 362125,
    #  362437, 364235, 364443, 364963, 367452, 367556, 369354, 369978, 371124, 371644, 372127, 373130, 373754, 400108, 401111, 402946, 405851,
    #  406854, 406958, 407545, 410422, 410526, 410734, 410942, 411425, 412012, 412116, 412948, 413015, 413327, 413743, 413951, 414226, 414330,
    #  414434, 416232, 416960, 417443, 418342, 419345, 419969, 420008, 420320, 420424, 420944, 451225, 451329, 451537, 453231, 453959, 454754,
    #  455445, 455549, 456448, 456656, 456760, 457659, 457763, 457867, 457971, 458454, 458558, 458766, 459353, 459457, 460224, 460848, 462854,
    #  462958, 463025, 463545, 463753, 463961, 464340, 465135, 465759, 465863, 466034, 466242, 466346, 466450, 466658, 467245, 467349, 468040,
    #  468872, 469251, 469355, 469459, 469875, 470122, 471437
    df = pd.read_csv(get_gender_file(), skiprows=1, names=GENDERS, sep="\t")
    return get_values_per_user_id(df, *genders)


def get_values_per_user_id(df, *filter_parameters):
    if len(filter_parameters) > 0:
        df = df[list(filter_parameters)]

    index = list()
    values = list()
    for column in df.columns:
        user_ids = list(df[column].dropna().values)
        index += user_ids
        values += [column] * len(user_ids)

    return pd.Series(data=values, index=pd.Index(data=index, dtype=int))


def get_sensor_columns(sensors):  # TODO: case sensitive
    if sensors is not None:
        sensor_columns = list()
        if ACCELEROMETER in sensors:
            sensor_columns += [0, 1, 2]
        if GYROSCOPE in sensors:
            sensor_columns += [3, 4, 5]
    else:
        sensor_columns = [0, 1, 2, 3, 4, 5]
    return sensor_columns


# TESTS

def test_get_data():
    data_frame = get_data(types=[TYPE_AUTO, TYPE_MANUAL], ages=AGE_GROUPS)
    assert data_frame.shape == (3466, 5)
    assert expand_data_frame(data_frame).shape == (1531923, 10)
    label_data_frame = data_frame[[LABEL_USER_KEY, LABEL_ACTIVITY_KEY]]
    label_data_frame.drop_duplicates(inplace=True)
    assert label_data_frame.shape == (1748, 2)


def test_get_file_names():
    file_names_auto = get_file_names(TYPE_AUTO, CENTER)
    walking_files_auto = list(get_filtered_file_names(file_names_auto, ACTIVITY_WALKING))
    assert len(file_names_auto) == 1490
    assert len(walking_files_auto) == 1490

    file_names_manual = get_file_names(TYPE_MANUAL, CENTER)
    walking_files_manual = list(get_filtered_file_names(file_names_manual, ACTIVITY_WALKING))
    assert len(file_names_manual) == 1976
    assert len(walking_files_manual) == 992


def test_age_filtering():
    assert len(get_user_ids_for_age(AGE_0_TO_9)) == 157
    assert len(get_user_ids_for_age(AGE_10_TO_19)) == 191
    assert len(get_user_ids_for_age(AGE_20_TO_29)) == 93
    assert len(get_user_ids_for_age(AGE_30_TO_39)) == 100
    assert len(get_user_ids_for_age(AGE_40_TO_49)) == 139
    assert len(get_user_ids_for_age(AGE_OVER_50)) == 64
    assert len(get_user_ids_for_age(AGE_10_TO_19)) + len(get_user_ids_for_age(AGE_OVER_50)) == len(get_user_ids_for_age(AGE_10_TO_19, AGE_OVER_50))


def test_activity_filtering():
    data_directory = os.path.join(get_resource_directory(), MANUAL_DATA_FOLDER, get_directory_for_position(CENTER))
    file_names = file_handling.get_file_names_in_directory_for_pattern(data_directory, pattern="T0*.csv")
    assert len(list(get_filtered_file_names(file_names, ACTIVITY_WALKING))) == 992
    assert len(list(get_filtered_file_names(file_names, ACTIVITY_SLOPE_DOWN))) == 495
    assert len(list(get_filtered_file_names(file_names, ACTIVITY_SLOPE_UP))) == 489
    assert len(list(get_filtered_file_names(file_names, ACTIVITY_WALKING, ACTIVITY_SLOPE_DOWN, ACTIVITY_SLOPE_UP))) == len(file_names)


def test_sensor_filtering():
    data_frame = get_data(types=[TYPE_AUTO, TYPE_MANUAL], sensors=(ACCELEROMETER,), ages=AGE_GROUPS)
    data_frame = data_frame.iloc[0]
    assert data_frame.shape == (5,)
    expand = expand_data_frame(data_frame)
    assert expand.shape == (536, 7)


def test_get_age_group():
    assert len(get_age_groups()) == len(get_age_groups(*AGE_GROUPS)) == 744
    assert len(get_age_groups(*DEFAULT_AGE_GROUPS)) == 587
    assert len(get_age_groups(AGE_0_TO_9)) == 157


def test_get_genders():
    assert len(get_genders()) == len(get_genders(*GENDERS)) == 640
    assert len(get_genders(FEMALE)) == 320
