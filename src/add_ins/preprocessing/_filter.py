from pykalman import KalmanFilter as mKalmanFilter

from src.core.base_classes import SafeTransformer
from src.core.utility.signal_filter_methods import apply_savitzky_golay_filter, apply_weighted_moving_average_according_to_bours_2018


class WeightedMovingAverageFilter(SafeTransformer):

    def _transform(self, data_frame, y=None):
        return data_frame.apply(apply_weighted_moving_average_according_to_bours_2018)


class SavitzkyGolayFilter(SafeTransformer):

    def __init__(self, filter_parameter, **kwargs):
        self.filter_parameter = filter_parameter
        super().__init__(**kwargs)

    def _transform(self, data):
        return data.apply(apply_savitzky_golay_filter, **self.filter_parameter)


class KalmanFilter(SafeTransformer):

    def __init__(self, filter_parameter, **kwargs):
        self.filter_parameter = filter_parameter
        super().__init__(**kwargs)

    def _transform(self, data):
        return data.apply(apply_kalman_filter, filter_parameter=self.filter_parameter)


def apply_kalman_filter(data, filter_parameter):
    """
    Retrieved smoothened vales for data by applying kalman filter predictions.

    Parameters
    ----------
    data
    filter_parameter

    Returns
    -------

    """
    smoothened_data, uncertainties = mKalmanFilter(**filter_parameter).smooth(data)
    # transform from numpy.ndarray to pd.Series
    # return pd.Series(map(lambda y: y[0], smoothened_data), name=data.name, index=data.index)
    return [data_point[0] for data_point in smoothened_data]
