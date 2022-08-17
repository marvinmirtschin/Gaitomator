import os

import numpy as np
import pandas as pd

import src.core.utility.file_handling as file_handling
from src.core.constants import DATA_FRAMES_KEY, LABEL_RECORD_KEY, LABEL_USER_KEY


def run_deterministic_test(actual_data_frame, paper_name, should_safe=False):
    directory = file_handling.get_data_directory()
    directory = os.path.join(directory, "self_created")
    os.makedirs(directory, exist_ok=True)
    file_name = os.path.join(directory, "deterministic-{}_file.csv".format(paper_name))
    if should_safe or not os.path.exists(file_name):
        # this will override existing files
        actual_data_frame.to_csv(file_name)
        # ensure this is not seen as successful check
        print("Create new file {}".format(file_name))
        assert False
    expected_data_frame = pd.read_csv(file_name, index_col=0, dtype={LABEL_USER_KEY: str, LABEL_RECORD_KEY: str})
    found_unexpected_behavior = False
    missing_columns = list()
    errors = list()
    for column in expected_data_frame.columns:
        try:
            if str(column) not in actual_data_frame.columns:
                missing_columns.append(column)
                found_unexpected_behavior = True
            elif "label" in str(column) or actual_data_frame[column].dtype == str or expected_data_frame[str(column)].dtype == str:
                assert actual_data_frame[column].equals(expected_data_frame[str(column)])
            else:
                assert all(np.isclose(actual_data_frame[column], expected_data_frame[str(column)], equal_nan=True))
        except (AssertionError, TypeError) as e:
            if hasattr(e, 'message'):
                message = e.message
            else:
                message = e.__class__
            errors.append("Error in column {}: {}".format(column, message))
            found_unexpected_behavior = True
    extra_columns = list()
    for column in actual_data_frame.columns:
        if column not in expected_data_frame.columns:
            extra_columns.append(column)
            found_unexpected_behavior = True

    # print report
    if found_unexpected_behavior:
        print()
        if len(missing_columns) > 0:
            print("Missing Columns: {}".format(np.array(missing_columns)))
        if len(extra_columns) > 0:
            print("Extra Columns: {}".format(np.array(extra_columns)))
        if len(errors) > 0:
            print("Faulty Columns:")
        for error in errors:
            print("\t\t{}".format(error))
    assert not found_unexpected_behavior


def get_example_cycles():
    cycle1 = pd.DataFrame(data=[[1, 1, 1], [1.2, 1.2, 1.2], [0.3, 0.3, 0.3], [-0.7, -0.7, -0.7]], columns=['x', 'y', 'z'])
    cycle2 = pd.DataFrame(data=[[2, 2, 2], [1, 1, 1], [0.3, 0.3, 0.3]], columns=['x', 'y', 'z'])
    cycle3 = pd.DataFrame(data=[[0.5, 0.5, 0.5], [1, 1, 1], [0.3, 0.3, 0.3], [-0.2, -0.2, -0.2]], columns=['x', 'y', 'z'])
    cycle4 = pd.DataFrame(data=[[1.2, 1.2, 1.2], [1, 1, 1], [0.3, 0.3, 0.3], [0, 0, 0], [-0.1, -0.1, -0.1]], columns=['x', 'y', 'z'])
    return cycle1, cycle2, cycle3, cycle4


def get_segment_data_frame():
    cycle1, cycle2, cycle3, cycle4 = get_example_cycles()
    segment1 = pd.concat([cycle1, cycle2, cycle3, cycle4])
    segment2 = pd.concat([cycle1, cycle3, cycle4])
    return pd.DataFrame(data=[segment1, segment2], columns=[DATA_FRAMES_KEY])
