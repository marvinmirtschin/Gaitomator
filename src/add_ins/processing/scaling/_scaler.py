import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.core import constants
from src.core.base_classes import SafeTransformer
from src.core.error_handling.exceptions import IncorrectInputTypeException, UnknownMethodException

TEST_SEED = constants.RANDOM_SEED


class SignalScaler(SafeTransformer):
    ZERO_NORMALIZATION = "zero_normalization"
    MIN_MAX_SCALER = "minmax"
    Z_SCORE_NORMALIZER = "zscore"
    STANDARD_SCALER = "standard"

    def __init__(self, scaling_method="zscore", raw_output=False, **kwargs):
        self.scaling_method = scaling_method
        self.raw_output = raw_output
        super().__init__(**kwargs)

    def fit(self, data_frame, y=None, **fit_params):
        if not isinstance(data_frame, pd.DataFrame):
            raise IncorrectInputTypeException(data_frame, pd.DataFrame)
        return super().fit(data_frame, y, **fit_params)

    # noinspection PyTypeChecker
    def _transform(self, data_frame):
        if not isinstance(data_frame, pd.DataFrame):
            raise IncorrectInputTypeException(data_frame, pd.DataFrame)

        if self.scaling_method is not None:
            if self.scaling_method in SCALING_MAPPING.keys():
                result = SCALING_MAPPING[self.scaling_method](data_frame)
                if self.raw_output:
                    return result
                else:
                    return pd.DataFrame(result, columns=data_frame.columns)
            else:
                raise UnknownMethodException("Scaling method {} is not implemented. Either implement it and add it to "
                                             "src.feature.normalizer.SCALING_MAPPING or use one of the available ones: {}"
                                             .format(self.scaling_method, SCALING_MAPPING.keys()))
        else:
            warnings.warn("DataFramePruner: Data frame could not be normalized. It was disabled in configuration, however it is included in the "
                          "estimator (pipeline)")
            return data_frame


def apply_zero_mean_normalization(data):
    """
    Normalize data by its mean.

    Parameters
    ----------
    data : array_like, one-dimensional ndarray (including time series)
        E.g. data of one column from pandas data frame

    Returns
    -------
    normalized data sequence
    """
    return data - np.mean(data)


def apply_standard_scaler(vector_list, with_mean=True, with_std=True, *args):
    """
    Parameters
    ----------
    vector_list : two-dimensional array-like, shape (n_samples, n_features)
        The data used to scale along the features axis.

    with_mean : bool, default=True
    with_std : bool, default=True
    args

    Returns
    -------
    transformed array : array-like, shape (n_samples, n_features)
    """
    vector_list = prepare_input_shape(vector_list)

    return StandardScaler(with_mean=with_mean, with_std=with_std).fit_transform(vector_list)


def apply_minimum_maximum_normalization(vector_list, full_data=None):
    vector_list = prepare_input_shape(vector_list)

    if full_data is None:
        full_data = vector_list

    scaler = MinMaxScaler()
    scaler.fit(full_data)
    return scaler.transform(vector_list)


def apply_zscore_normalization(data_array):
    return stats.zscore(data_array)


def prepare_input_shape(vector_list):
    if isinstance(vector_list, pd.Series):
        vector_list = vector_list.to_numpy()

    if isinstance(vector_list, pd.DataFrame):
        vector_list = vector_list.to_numpy()

    if len(vector_list.shape) == 1:
        # only one feature -> reshape(-1, 1)
        vector_list = vector_list.reshape(-1, 1)

    return vector_list


SCALING_MAPPING = {
    SignalScaler.ZERO_NORMALIZATION: apply_zero_mean_normalization,
    SignalScaler.MIN_MAX_SCALER    : apply_minimum_maximum_normalization,
    SignalScaler.Z_SCORE_NORMALIZER: apply_zscore_normalization,
    SignalScaler.STANDARD_SCALER   : apply_standard_scaler,
}


###### TESTING #####


def test_zero_normalization():
    np.random.seed(TEST_SEED)
    data = get_test_data_frame(9, 2)
    result = apply_zero_mean_normalization(data)
    expected_result = np.array([[0.195004, -0.000416], [-0.373087, 0.415081], [0.306504, 0.114421], [-0.246624, 0.186100], [0.362021, 0.020230],
                                [0.097623, 0.343213], [0.100966, -0.243373], [-0.124133, -0.375979], [-0.318275, -0.459278]])
    assert np.allclose(expected_result, result, 0.001)


def test_minmax_normalization():
    np.random.seed(TEST_SEED)
    data = get_test_data_frame(9, 2)
    result = apply_minimum_maximum_normalization(data)
    expected_result = np.array([[0.77279933, 0.5247983], [0., 1.], [0.92447686, 0.65613682], [0.17203395, 0.73811474], [1., 0.5484111],
                                [0.64032707, 0.917805], [0.64487568, 0.24692888], [0.33866327, 0.09526846], [0.07456419, 0.]])
    assert np.allclose(expected_result, result, 0.001)


def test_zscore_normalization():
    np.random.seed(TEST_SEED)
    data = get_test_data_frame(9, 2)
    result = apply_zscore_normalization(data)
    expected_result = np.array([[7.55575511e-01, -1.44016197e-03], [-1.44558735e+00, 1.43812389e+00], [1.18759834e+00, 3.96433445e-01],
                                [-9.55583389e-01, 6.44775267e-01], [1.40271075e+00, 7.00918669e-02], [3.78255028e-01, 1.18912444e+00],
                                [3.91210825e-01, -8.43210798e-01], [-4.80973410e-01, -1.30264703e+00], [-1.23320630e+00, -1.59125091e+00]])
    assert np.allclose(expected_result, result, 0.001)


def test_standard_scaling_normalization():
    np.random.seed(TEST_SEED)
    data = get_test_data(9, 2)
    data_frame = pd.DataFrame(data=data)
    result1 = apply_standard_scaler(data)
    result2 = apply_standard_scaler(data[:, 0].reshape(-1, 1))
    result3 = apply_standard_scaler(data[:, 0].reshape(1, -1))
    result4 = apply_standard_scaler(data_frame)
    expected_result = np.array([[7.55575511e-01, -1.44016197e-03], [-1.44558735e+00, 1.43812389e+00],
                                [1.18759834e+00, 3.96433445e-01], [-9.55583389e-01, 6.44775267e-01], [1.40271075e+00, 7.00918669e-02],
                                [3.78255028e-01, 1.18912444e+00], [3.91210825e-01, -8.43210798e-01], [-4.80973410e-01, -1.30264703e+00],
                                [-1.23320630e+00, -1.59125091e+00]])
    np.testing.assert_allclose(result1, expected_result)
    np.testing.assert_allclose(result2.reshape(len(result1)), result1[:, 0])
    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, result2, result3)
    np.testing.assert_allclose(result4, result1)


def test_unknown_method_exception_raised():
    with np.testing.assert_raises(UnknownMethodException):
        SignalScaler(scaling_method="Unknown").fit_transform(get_test_data_frame(9, 2))


def test_signal_scaler():
    np.random.seed(TEST_SEED)
    expected_result = np.array([[7.55575511e-01, -1.44016197e-03], [-1.44558735e+00, 1.43812389e+00],
                                [1.18759834e+00, 3.96433445e-01], [-9.55583389e-01, 6.44775267e-01], [1.40271075e+00, 7.00918669e-02],
                                [3.78255028e-01, 1.18912444e+00], [3.91210825e-01, -8.43210798e-01], [-4.80973410e-01, -1.30264703e+00],
                                [-1.23320630e+00, -1.59125091e+00]])

    signal_scaler = SignalScaler(scaling_method=SignalScaler.STANDARD_SCALER)
    result = signal_scaler.fit_transform(get_test_data_frame(9, 2))
    np.testing.assert_allclose(result, expected_result)


def get_test_data(number_of_rows, number_of_columns):
    return np.random.rand(number_of_rows, number_of_columns)


def get_test_data_frame(number_of_rows, number_of_columns):
    columns = ["column_{}".format(counter) for counter in range(number_of_columns)]
    return pd.DataFrame(data=get_test_data(number_of_rows, number_of_columns), columns=columns)
