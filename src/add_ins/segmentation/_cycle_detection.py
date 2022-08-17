import pandas as pd

from src.core.constants import DATA_FRAMES_KEY


def split_data_for_indices(data, indices):
    cycles = list()
    for cycle_index in range(len(indices) - 1):
        cycle = data.iloc[indices[cycle_index]:indices[cycle_index + 1]]
        cycle.reset_index(inplace=True, drop=True)
        cycles.append(cycle)

    return pd.DataFrame({DATA_FRAMES_KEY: cycles})
