from itertools import chain

import numpy as np
import pywt

from src.core.base_classes import FeatureCalculator


class MultiLevelWaveletEnergyCalculator(FeatureCalculator):
    """
    Wavelet domain feature using multilevel wavelets. The default values are taken from 'Performance evaluation of implicit smartphones
    authentication via sensor-behavior analysis' by Shen et al.

    "The energy is the sum of the square of the absolute values": https://math.stackexchange.com/questions/1086522/how-to-calculate-wavelet-energy
    https://en.wikipedia.org/wiki/Energy_(signal_processing)

    Parameters
        ----------
        wavelet_name : Wavelet object or name string, default='db3'
            Wavelet to use
        mode : str, default='periodic'
            # Question: not mentioned in data_set, default of library is symmetric
            Signal extension mode, see `pywt.Modes.modes`.
        levels : array-like, default=[4, 5]
            Decomposition level used for calculating the energy (must be > 0).
    Returns
        ----------
         Filtered data frame.
    """

    FEATURE_NAME = "wavelet-energy"

    def __init__(self, wavelet_name='db3', mode='periodic', levels=None, **kwargs):
        self.wavelet_name = wavelet_name
        self.mode = mode
        if levels is None:
            levels = [4, 5]
        self.levels = levels
        super().__init__(feature_name=self.FEATURE_NAME, **kwargs)

    def _transform(self, data):
        return data.apply(calculate_wavelet_energy, args=(self.wavelet_name, self.mode, self.levels))


def calculate_wavelet_energy(data_series: np.ndarray, wavelet_name, mode, levels):
    # coefficients are an array with the coarse coefficients from level self.level at [0] and
    # detailed coefficients in reversed level-order from then on.

    # cA, cD_L, cD_L-1, .... , cD_1 with L=level
    coefficients = pywt.wavedec(data=data_series, wavelet=wavelet_name, mode=mode, level=max(levels))
    detailed_coefficients = [pow(abs(coefficients[-level]), 2) for level in levels]

    return np.sum(list(chain(*detailed_coefficients)))
