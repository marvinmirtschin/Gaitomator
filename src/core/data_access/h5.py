import os

import pandas as pd

import src.core.utility.file_handling
from src.core.constants import DATA_FRAMES_KEY, LABEL_USER_KEY

FILE1 = "20180921-160639_Walk.h5"
FILE2 = "20180921-160822_Walk.h5"
FILE3 = "20180921-161005_Walk.h5"

FILES = (FILE1, FILE2, FILE3)

SENSORS = ("Accelerometer", "Barometer", "Gyroscope", "Magnetometer", "Temperature", "Time")
# TODO: time as index?
RELEVANT_SENSORS = ("Accelerometer", "Gyroscope", "Magnetometer", "Time")
# TODO: take a look into configurations and metrics
CONFIGURATION = "Configuration"
METRICS = "Metrics"

DIMENSIONS = {
    0: "x",
    1: "y",
    2: "z"
}


def read_h5_file(files=FILES, ids=None, sensors=SENSORS):
    path = os.path.join(src.core.utility.file_handling.get_home_path(), "Documents", "Privat", "gait_projects", "sawd_gcvs", "matlab", "data")

    data_frames = list()
    used_ids = list()
    for file in files:
        file_path = os.path.join(path, file)

        store = pd.HDFStore(file_path, 'r')

        # annotations = store.root.Annotations  # -> not used for now
        # processed = store.root.Processed -> not used for now

        if ids is None:
            use_ids = list(store.root.Sensors._v_children.keys())
        else:
            use_ids = ids

        for _id in use_ids:
            try:
                # different way to access the available ids:
                # available_ids = list(store.root.Sensors._v_children.keys())
                values = store.root.Sensors[_id]
            except IndexError:
                continue

            dfs = list()
            for sensor in sensors:
                data_frame = pd.DataFrame(values[sensor])
                data_frame.columns = _get_dimension(sensor, len(data_frame.columns))
                dfs.append(data_frame)
            try:
                data_frames.append(pd.concat(dfs, axis=1))
                used_ids.append(_id)
            except ValueError:
                # No objects to concatenate
                print(f"No data for sensors ({sensors}) available")
                pass
            # TODO: do I have information about the user?
        store.close()
    data = [data_frames, used_ids]
    combined_df = pd.DataFrame(data, index=[DATA_FRAMES_KEY, LABEL_USER_KEY])
    return combined_df.T


def _get_dimension(sensor, count):
    sensor = sensor.lower()
    if count == 1:
        return (sensor,)
    elif count == 3:
        return (sensor + "_" + DIMENSIONS[i] for i in range(count))
    else:
        return (sensor + "_" + i for i in range(count))


def test_read():
    dfs = read_h5_file(files=FILES)
    assert len(dfs) == 18
    print()


def test_print_data_report():
    import src.core.utility.time_unit as time_unit
    from statistics import mean
    dfs = read_h5_file(files=FILES)
    number_users = len(dfs)
    lengths = list()
    durations = list()
    frequencies = list()
    print()
    for df in dfs["data_frames"]:
        length = len(df)
        # assuming microseconds since epoch
        duration = time_unit.MICROSECONDS.to_seconds(df["time"].iloc[-1] - df["time"].iloc[0])
        frequency = length / duration

        lengths.append(length)
        durations.append(duration)
        frequencies.append(frequency)
        print(f"Got {length} data points over {duration} s ({frequency} Hz)")
    print("==============")
    print(
        f"Got data for {number_users} people with a duration of about {time_unit.SECONDS.to_minutes(sum(durations))} min and an average frequency of "
        f"{mean(frequencies)}")
