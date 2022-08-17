import numpy as np
import scipy

from src.core.base_classes import FeatureCalculator
from src.core.error_handling.exceptions import MissingSensorDataException


class FourierCoefficientsMagnitudeCalculator(FeatureCalculator):

    def __init__(self, number_of_coefficients=40, skip_first=False, **kwargs):
        self.number_of_coefficients = number_of_coefficients
        self.skip_first = skip_first
        super().__init__(feature_name="fourier-magnitude", **kwargs)

    def _transform(self, data):
        # raw=True not usable as number of rows is altered
        return data.apply(calculate_fourier_transform_coefficients_magnitude, args=[self.number_of_coefficients, 1 if self.skip_first else 0])


def calculate_fourier_transform_coefficients_magnitude(data, number_of_coefficients=40, start=0):
    coefficients = np.fft.fft(data)
    # coefficients[0] contains the zero-frequency term (the sum of the signal)
    # abs(coefficients) is its amplitude (magnitude in the paper) spectrum
    return abs(coefficients[start:number_of_coefficients])


# Discrete Cosine Transform
class DiscreteCosineCoefficientsMagnitude(FeatureCalculator):

    def __init__(self, number_of_coefficients=40, **kwargs):
        self.number_of_coefficients = number_of_coefficients
        super().__init__(feature_name="discrete-cosine", **kwargs)

    def _transform(self, data):
        return data.apply(calculate_discrete_cosine_transform_coefficients, args=[self.number_of_coefficients])


def calculate_discrete_cosine_transform_coefficients(series, number_of_coefficients=40, dct_type=3, norm=None):
    if len(series) == 0:
        raise MissingSensorDataException()
    # Refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html for more insides on the type
    # In the Paper 'Adaptive Cross-Device Gait Recognition Using a Mobile Accelerometer' it can be seen that type 3 is the one used by the authors
    coefficients = scipy.fft.dct(series.to_numpy(), type=dct_type, norm=norm)
    # In their code from github (https://github.com/thanghoang/GaitAuth) the coefficients are normalized by the size of the input data:
    #   DCT = dct(curData(:,j))/length(size(curData,1))
    return coefficients[:number_of_coefficients] / len(series)
