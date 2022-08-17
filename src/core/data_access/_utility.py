import random

from src.core.constants import LABEL_RECORD_KEY, LABEL_USER_KEY


def reduce_data_frame(data_frame, max_number_of_users: int = None, max_number_of_recordings_per_user: int = None):
    """
    Default method to reduce number of data before calculation. This is primarily done to reduce computation. Remember to seed before usage if you
    want predictable results.

    Parameters
    ----------
    data_frame : pd.DataFrame
        DataFrame to be reduce.
    max_number_of_users : int, default=None
        If given, number of users will be reduced to this (if more are available) by random selection.
    max_number_of_recordings_per_user
        If given and available, number of records per user will be reduced to this (if more are available) by random selection.
    Returns
    -------

    """
    users = data_frame[LABEL_USER_KEY].unique()

    if max_number_of_users is not None and len(users) > max_number_of_users:
        random.shuffle(users)
        users = users[:max_number_of_users]

    selected_data_frame = data_frame[data_frame[LABEL_USER_KEY].isin(users)]

    records = list()
    if max_number_of_recordings_per_user is not None and LABEL_RECORD_KEY in selected_data_frame.columns:
        for user in users:
            available_records = list(selected_data_frame[selected_data_frame[LABEL_USER_KEY] == user][LABEL_RECORD_KEY].unique())
            if len(available_records) > max_number_of_recordings_per_user:
                random.shuffle(available_records)
                available_records = available_records[:max_number_of_recordings_per_user]
            records += available_records

        selected_data_frame = selected_data_frame[selected_data_frame[LABEL_RECORD_KEY].isin(records)]

    return selected_data_frame
