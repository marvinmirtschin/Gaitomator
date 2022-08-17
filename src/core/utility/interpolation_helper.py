import numpy as np
import pandas as pd
from scipy import interpolate

from src.core.error_handling.exceptions import InterpolationException


def interpolate_cycle_list(cycles, interpolation_cycle_length=100, interpolation_method="linear"):
    interpolated_cycle_list = []
    for cycle_values in cycles:
        interpolated = cycle_values.apply(interpolate_data_series_by_length, interpolation_cycle_length=interpolation_cycle_length,
                                          interpolation_method=interpolation_method)

        interpolated_cycle_list.append(interpolated)
    return pd.concat(interpolated_cycle_list, axis=1)


def interpolate_data_series_by_length(pandas_series, interpolation_cycle_length, interpolation_method="linear"):
    """
    Computes a function, that covers values in the range of [x_o,x_n]. Any x-value between this range can be used to compute the associated
    y-value. All possible methods are listed in <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html#scipy
    .interpolate.interp1d">Interpolation Function</a>

    Parameters
    ----------
    pandas_series : pd.Series
        One-dimensional ndarray with axis labels (including time series); e.g. one column of data frame.
    interpolation_cycle_length : int
        Absolute amount of values in one cycle.
    interpolation_method : str
        Specifies the kind of interpolation, e.g. linear, nearest, quadratic, cubic, etc. (Check all possible methods in documentation).

    Returns
    -------
        interpolated data series : pd.Series
    """
    try:
        index = np.arange(0, pandas_series.shape[0])
        data_array = pandas_series.values
        interpolation_function = interpolate.interp1d(index, data_array, kind=interpolation_method, fill_value="extrapolate")
        interpolation_index = np.linspace(0, pandas_series.shape[0], interpolation_cycle_length)
        interpolated_data_array = interpolation_function(interpolation_index)
        return pd.Series(interpolated_data_array)
    except ValueError as e:
        raise InterpolationException("Cycle could not be interpolated: " + str(e))


def correct_sampling_frequency_by_interpolation(data_frame, interpolation_method="linear", listening_rate=20, time_unit='ms', start=None, end=None,
                                                old_listening_rate=None):
    """
    Parameters
    ----------
    time_unit : str, default='ms' (milliseconds)

    data_frame : pd.DataFrame with c
        The data used to interpolate along the features' axis; two-dimensional data frame, shape (n_samples, n_features), index is a TimeDeltaIndex
        of time series' timestamp information.
    interpolation_method : str, default="linear"
    listening_rate : float, default=20
        Number of milliseconds between two time events; it is used for sample frequency correction, pretending that sensor events are called
        consistent in a interval of e.g. 20 milliseconds.
    start : pd.TimedeltaIndex, default=None
        TimedeltaIndex where to start with interpolation
    end : pd.TimedeltaIndex, default_None
        TimedeltaIndex where to end with interpolation

    Returns
    -------
    interpolated_data_frame : pd.DataFrame
    """
    # if not isinstance(data_frame.index, pd.TimedeltaIndex):
    #     raise Exception("The index of the data frame is not of type timeDelta. "
    #                     "Treating the given index values as milliseconds would result in wrong results.")
    if not isinstance(data_frame.index, pd.TimedeltaIndex):
        if old_listening_rate is None:
            raise Exception("Invalid Input")
        if start is None:
            start = f"{data_frame.index[0]} {time_unit}"
        if end is None:
            end = f"{data_frame.index[-1] * old_listening_rate} {time_unit}"
        time_delta = pd.Timedelta(value=old_listening_rate, unit=time_unit)
        time_delta_index = pd.timedelta_range(start=start, end=end, freq=time_delta)
        data_frame.index = time_delta_index
    else:
        if start is None:
            start = data_frame.index[0]
        if end is None:
            end = data_frame.index[-1]

    time_delta = pd.Timedelta(value=listening_rate, unit=time_unit)
    new_index = pd.timedelta_range(start=start, end=end, freq=time_delta)
    interpolation_function = interpolate.interp1d(data_frame.index.astype('i8'), data_frame.values.T, kind=interpolation_method,
                                                  fill_value="extrapolate")
    interpolated_data_frame = pd.DataFrame(data=interpolation_function(new_index.astype('i8')).T, index=new_index)
    interpolated_data_frame.columns = data_frame.columns

    return interpolated_data_frame
