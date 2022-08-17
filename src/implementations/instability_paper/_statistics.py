import random

from sklearn.pipeline import Pipeline

from src.core.base_classes import NestedDataFrameTransformer
from src.core.constants import LABEL_RECORD_KEY, LABEL_USER_KEY
from src.core.data_access import DataAccessor
from src.core.data_access.hoang_2015._access import get_files_from_paper
from src.add_ins.segmentation.hoang_2015 import CycleDetectionTransformer, SegmentationTransformer
from src.core.utility.data_frame_transformer import Tabularizer
from src.core.utility.sensor_fusion import SensorSynchronizer
from src.core.utility.signal_filter_methods import MultiLevelWaveletFilter
from src.implementations.instability_paper import CoordinateTransformer, SensorInputTransformer


def get_all_users():
    file_names = get_files_from_paper()
    users = set()
    for file_name in file_names:
        short_file_name = file_name.split("/")[-1]
        short_file_name = short_file_name[:-4]

        user_id, gender, record_id, sensor, recording_number = short_file_name.split("_")
        users.add(user_id)
    return list(users)


def get_all_users_record_pairs():
    file_names = get_files_from_paper()
    user_record_pairs = set()
    for file_name in file_names:
        short_file_name = file_name.split("/")[-1]
        short_file_name = short_file_name[:-4]

        user_id, gender, record_id, sensor, recording_number = short_file_name.split("_")
        user_record_pairs.add((user_id, record_id))
    return list(user_record_pairs)


def test_print_number_of_found_cycles():
    # Note: result is saved in master thesis git
    user_ids = get_all_users()
    users_per_run = 4
    print_cycles_for_user_ids(user_ids, users_per_run=users_per_run)


def print_cycles_for_user_ids(user_ids, users_per_run=4, window_size=10, listening_rate=1000 / 27):
    # frequency of 27 Hz
    start = 0
    while start < len(user_ids):
        end = min(start + users_per_run, len(user_ids))

        data_accessor = DataAccessor(DataAccessor.DATA_HOANG_2015,
                                     filter_parameters={"max_number_of_users": 2, "max_number_of_recordings_per_user": 3},
                                     sensors=["Linear Acceleration", "Rotation Matrix"])
        data_frame = data_accessor.transform()

        cycle_detector = NestedDataFrameTransformer(
            CycleDetectionTransformer(cut_at_dimension='accelerometer_z', window_size=window_size))
        segmentation_transformer = NestedDataFrameTransformer(
            SegmentationTransformer(cycles_per_segment=1, number_of_overlapping_cycles=0, ))

        pipe = Pipeline([
            ('sensor_fusion', NestedDataFrameTransformer(SensorSynchronizer(reference_sensor="linearAcceleration", listening_rate=listening_rate))),
            ('acceleration_transformation', NestedDataFrameTransformer(CoordinateTransformer())),
            ('new_input', NestedDataFrameTransformer(SensorInputTransformer())),
            ('filtering', NestedDataFrameTransformer(MultiLevelWaveletFilter())),
            ('cycle_detection', cycle_detector),
            ('segmentation', segmentation_transformer),
            ('tabularizer', Tabularizer()),
        ])
        transformed_data_fame = pipe.transform(data_frame)
        print_results(transformed_data_fame, user_ids[start:end])
        start += users_per_run


def print_results(transformed_data_fame, user_ids):
    for user_id in user_ids:
        selected_df = transformed_data_fame[transformed_data_fame[LABEL_USER_KEY] == user_id]
        rec_ids = set(selected_df[LABEL_RECORD_KEY])
        for rec_id in rec_ids:
            df = selected_df[selected_df[LABEL_RECORD_KEY] == rec_id]
            print("{} - {} : {}".format(user_id, rec_id, len(df)))


def test_check_parameter_for_segmentation():
    user_ids = get_all_users()
    user_ids = random.sample(user_ids, 4)
    print("\nWindow: {}, Listening: {}".format(20, 20))
    print_cycles_for_user_ids(user_ids, window_size=20, listening_rate=20)

    print("\nWindow: {}, Listening: {}".format(20, 50))
    print_cycles_for_user_ids(user_ids, window_size=20, listening_rate=50)

    print("\nWindow: {}, Listening: {}".format(30, 20))
    print_cycles_for_user_ids(user_ids, window_size=30, listening_rate=20)

    print("\nWindow: {}, Listening: {}".format(30, 50))
    print_cycles_for_user_ids(user_ids, window_size=30, listening_rate=50)
