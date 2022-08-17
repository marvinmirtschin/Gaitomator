from src.core.constants import DATA_FRAMES_KEY, LABEL_ACTIVITY_KEY, LABEL_USER_KEY
from src.core.data_access.ou._ou import (ACTIVITY_WALKING, AGE_GROUPS, GIVEN_FREQUENCY, TYPE_AUTO, TYPE_MANUAL, get_age_groups, get_data, get_genders)

LABEL_AGE_GROUP_KEY = "label_age_group"
LABEL_GENDER_KEY = "label_gender"
LABEL_VALUE_COUNT_KEY = "label_value_count"
UNKNOWN = "UNKNOWN"


# TODO: Find useful statistics for presentation / paper

def test_something_else():
    data_frame = get_data(types=(TYPE_AUTO, TYPE_MANUAL), ages=AGE_GROUPS, skip_data=False)
    add_gender(data_frame)
    add_age_group(data_frame)
    add_number_of_values(data_frame)
    users_with_unknown_gender = get_users_with_unknown_gender(data_frame)
    users_with_unknown_age = get_users_with_unknown_age(data_frame)
    average_walking_time = get_average_walking_time(data_frame)
    average_recording_time = get_average_recording_time(data_frame)
    print()


def add_age_group(data_frame):
    age_groups = get_age_groups()
    data_frame[LABEL_AGE_GROUP_KEY] = [age_groups[user_id] if user_id in age_groups.index else UNKNOWN for user_id in data_frame[LABEL_USER_KEY]]


def add_gender(data_frame):
    genders = get_genders()
    data_frame[LABEL_GENDER_KEY] = [genders[user_id] if user_id in genders.index else UNKNOWN for user_id in data_frame[LABEL_USER_KEY]]


def add_number_of_values(data_frame):
    data_frame[LABEL_VALUE_COUNT_KEY] = [len(df) for df in data_frame[DATA_FRAMES_KEY]]


def get_number_of_walking_samples(data_frame):
    return data_frame.groupby(LABEL_ACTIVITY_KEY)[LABEL_USER_KEY].count().loc[ACTIVITY_WALKING]


def get_number_of_walking_samples_per_user(data_frame):
    return data_frame[data_frame[LABEL_ACTIVITY_KEY] == ACTIVITY_WALKING].groupby(LABEL_USER_KEY)[LABEL_ACTIVITY_KEY].count()


def get_average_walking_time(data_frame):
    return data_frame[data_frame[LABEL_ACTIVITY_KEY] == ACTIVITY_WALKING][LABEL_VALUE_COUNT_KEY].mean() / GIVEN_FREQUENCY


def get_average_recording_time(data_frame):
    return data_frame[LABEL_VALUE_COUNT_KEY].mean() / GIVEN_FREQUENCY


def get_users_with_unknown_gender(data_frame):
    return data_frame[data_frame[LABEL_GENDER_KEY] == UNKNOWN][LABEL_USER_KEY].unique()


def get_users_with_unknown_age(data_frame):
    return data_frame[data_frame[LABEL_AGE_GROUP_KEY] == UNKNOWN][LABEL_USER_KEY].unique()


def print_list_as_seperated_string(m_list):
    print(", ".join(m_list.astype(str))[2:])
