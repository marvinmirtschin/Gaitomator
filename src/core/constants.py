import pandas as pd

RANDOM_SEED = 14159265  # used for reproducible results

LABEL_RECORD_KEY = "label_record"
LABEL_USER_KEY = "label_user"
LABEL_ACTIVITY_KEY = "label_activity"
DATA_FRAMES_KEY = "data_frames"

DEVICE_TYPE_PHONE = "phone"
DEVICE_TYPE_WATCH = "watch"

TIMESTAMP_KEY = "timestamp"
TIME = "time"

DIMENSION_TIMESTAMP = 0
DIMENSION_X = 1

VIS_TYPE_BAR = 'bar'
VIS_TYPE_LINE = 'list'

DIMENSIONS_KEY_LIST = ["x", "y", "z", 0, 1, 2, 3, 4, 5, 6, 7, 8]

# TODO: maybe move into separate testing constants file
TEST_DATA_FRAME_1 = pd.DataFrame(data=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]], columns=["x", "y", "z"])
TEST_DATA_FRAME_2 = pd.DataFrame(data=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]], columns=["x", "y", "z"])
TEST_DATA_FRAME_3 = pd.DataFrame(data=[[.1, .2, .3], [.4, .5, .6], [.7, .8, .9], [.10, .11, .12], [.13, .14, .15]], columns=["x", "y", "z"])
TEST_DATA = pd.DataFrame(data=[[TEST_DATA_FRAME_1, "1"], [TEST_DATA_FRAME_2, "2"], [TEST_DATA_FRAME_3, "3"]], columns=["data_frames", "label_user"])
