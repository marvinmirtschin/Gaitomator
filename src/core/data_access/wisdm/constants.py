from src.core.constants import LABEL_ACTIVITY_KEY, LABEL_USER_KEY

WISDM = "WISDM"
WISDM_V1 = WISDM + "_ar_v1.1"
WISDM_V2 = WISDM + "_at_v2.0"
UNIQUE_ID = "UNIQUE_ID"

TYPE_RAW = "raw"
TYPE_TRANSFORMED = "transformed"
TYPES = [TYPE_RAW, TYPE_TRANSFORMED]

# TODO: use labels from const to be consistent
# USER = "user"
# ACTIVITY = "activity"
LABEL_TIMESTAMP_KEY = "timestamp"
TRANSFORMED_COLUMNS = [
    LABEL_USER_KEY,
    "X0",
    "X1",
    "X2",
    "X3",
    "X4",
    "X5",
    "X6",
    "X7",
    "X8",
    "X9",
    "Y0",
    "Y1",
    "Y2",
    "Y3",
    "Y4",
    "Y5",
    "Y6",
    "Y7",
    "Y8",
    "Y9",
    "Z0",
    "Z1",
    "Z2",
    "Z3",
    "Z4",
    "Z5",
    "Z6",
    "Z7",
    "Z8",
    "Z9",
    "XAVG",
    "YAVG",
    "ZAVG",
    "XPEAK",
    "YPEAK",
    "ZPEAK",
    "XABSOLDEV",
    "YABSOLDEV",
    "ZABSOLDEV",
    "XSTANDDEV",
    "YSTANDDEV",
    "ZSTANDDEV",
    "RESULTANT",
    LABEL_ACTIVITY_KEY
]
RAW_COLUMNS = [LABEL_USER_KEY, LABEL_ACTIVITY_KEY, LABEL_TIMESTAMP_KEY, "accelerometer_x", "accelerometer_y", "accelerometer_z"]
