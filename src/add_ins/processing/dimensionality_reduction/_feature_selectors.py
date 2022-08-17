# Feature selection
import numpy as np
from scipy.stats import ks_2samp

from src.core.base_classes import SafeTransformer


class KolmogorovSmirnovFeatureSelector(SafeTransformer):
    # @formatter:off
    # sample_sizes n1: 4-12, n2: 2-12
    CRITICAL_VALUES = [
        [(None, None), (None, None), (None, None), (None, None), ((16, 16), None), ((18, 18), None), ((20, 20), None), ((22, 22), None), ((24, 24), None)],
        [(None, None), ((15, 15), None), ((18, 18), None), ((21, 21), None), ((21, 24), (24, 24)), ((24, 27), (27, 27)), ((27, 30), (30, 30)), ((30, 33), (33, 33)), ((30, 36), (36, 36))],
        [((16, 16), None), ((20, 20), None), ((20, 24), (24, 24)), ((24, 28), (28, 28)), ((28, 32), (32, 32)), ((28, 36), (32, 36)), ((30, 40), (36, 40)), ((33, 44), (40, 44)), ((36, 48), (44, 48))],
        [None, (None, None), ((24, 30), (30, 30)), ((30, 35), (35, 35)), ((30, 40), (35, 40)), ((35, 45), (40, 45)), ((40, 50), (45, 50)), ((39, 55), (45, 55)), ((43, 60), (50, 60))],
        [None, None, ((30, 36), (36, 36)), ((30, 42), (36, 42)), ((34, 48), (40, 48)), ((39, 54), (45, 54)), ((40, 60), (48, 60)), ((43, 66), (54, 66)), ((48, 72), (60, 72))],
        [None, None, None, ((42, 49), (42, 49)), ((40, 56), (48, 56)), ((42, 63), (49, 63)), ((46, 70), (53, 70)), ((48, 77), (59, 77)), ((53, 84), (60, 84))],
        [None, None, None, None, ((48, 64), (56, 64)), ((46, 72), (55, 72)), ((48, 80), (60, 80)), ((53, 88), (64, 88)), ((60, 96), (68, 96))],
        [None, None, None, None, None, ((54, 81), (63, 81)), ((53, 90), (70, 90)), ((59, 99), (70, 99)), ((63, 108), (75, 108))],
        [None, None, None, None, None, None, ((70, 100), (80, 100)), ((60, 110), (77, 110)), ((66, 120), (80, 120))],
        [None, None, None, None, None, None, None, ((77, 121), (88, 121)), ((72, 132), (86, 132))],
        [None, None, None, None, None, None, None, None, ((96, 144), (84, 144))]]
    # @formatter:on

    def __init__(self, alpha=0.05, use_p_value=True, error_handler=None, allowed_exceptions=None, default_item=None):
        self.use_p_value = use_p_value
        self.alpha = alpha
        super().__init__(error_handler=error_handler, allowed_exceptions=allowed_exceptions, default_item=default_item)

    def _get_critical_value(self, size_a, size_b, alpha):
        if (alpha == 0.05 or alpha == 0.01) and size_a < 13 and size_b < 13:
            if size_a < size_b:
                size_a, size_b = size_b, size_a
            if size_a < 4:
                return None
            if size_b < 2:
                return None
            return self.CRITICAL_VALUES[size_b - 2][size_a - 4]
        else:
            return KolmogorovSmirnovFeatureSelector._calculate_critical_value(size_a, size_b, alpha)

    @staticmethod
    def _calculate_critical_value(size_a, size_b, alpha):
        return KolmogorovSmirnovFeatureSelector._get_alpha_coefficient(alpha) * np.sqrt((size_a + size_b) / (size_a * size_b))

    @staticmethod
    def _get_alpha_coefficient(alpha):
        if alpha == .1:
            return 1.22
        elif alpha == .05:
            return 1.36
        elif alpha == .025:
            return 1.48
        elif alpha == .01:
            return 1.63
        elif alpha == .005:
            return 1.73
        elif alpha == 0.001:
            return 1.95
        raise Exception("Unknown alpha value ({}) for statistical tests".format(alpha))

    def _transform(self, data):
        # TODO: split depending on feature domain
        # select these dimensions whose maximal mean p-values are less than α.

        distances = dict()
        for i in range(len(data.columns) - 1):
            for j in range(i + 1, len(data.columns)):
                data1 = data.iloc[:, 0]
                data2 = data.iloc[:, 1]

                result = ks_2samp(data1, data2)

                if i not in distances:
                    distances[i] = list()
                if j not in distances:
                    distances[j] = list()

                if self.use_p_value:
                    distances[i].append(result[1])
                    distances[j].append(result[1])
                else:
                    distances[i].append(result[0])
                    distances[j].append(result[0])
        if self.use_p_value:
            means = [(key, np.mean(values)) for key, values in distances.items()]
            columns_to_remove = set()
            for key, value in means:
                if value > self.alpha:
                    columns_to_remove.add(key)
            return data.drop(columns_to_remove, axis=1)
        else:
            # TODO: critical values way
            pass
