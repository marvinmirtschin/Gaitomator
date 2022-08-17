import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import auc, roc_curve

from src.core.visualization import visualizer


class DET:
    """
    Store decision error trade off curves for each cross validation split and computes the average after CV.
    Example: <a href="https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html">ROC for CV</a>

    Usage:
    -   'add_curve_and_metrics' in scoring_functions: Store false positives, true negatives, false matches, false non matches and eer as well as the
        DET curve for each cv split.
    -   'plot_operating_characteristics': in e.g. validate_cross_validation: After cross validation/ grid search, the average curve and classification
        metrics are computed for all cv splits. Curves of same labels are in same color.
    """

    def __init__(self):
        self.label_list = list()
        self.mean_curve = None
        self.curve_list = list()
        # auc
        self.area_under_the_curve_list = list()
        self.false_positive_list = list()
        self.true_positive_list = list()
        # eer
        self.false_match_list = list()
        self.false_non_match_list = list()
        self.equal_error_list = list()

        self.threshold_variations = np.linspace(0, 1, 100)

    def add_curve_and_metrics(self, y_true, y_score, label):
        self.label_list.append(label)

        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true=y_true, y_score=y_score)
        false_match_rate = false_positive_rate
        false_non_match_rate = 1 - true_positive_rate

        self.false_match_list.append(false_match_rate)
        self.false_non_match_list.append(false_non_match_rate)

        self.equal_error_list.append(compute_interpolated_equal_error_rate(y_true=y_true, y_score=y_score))

        self.curve_list.append(np.interp(self.threshold_variations, false_match_rate, false_non_match_rate))
        self.curve_list[-1][0] = 1.0

        self.area_under_the_curve_list.append(auc(false_match_rate, false_non_match_rate))

    def get_average_curve_and_metrics(self):
        if len(self.curve_list) == 0:
            raise Exception("Curve list for plotting DET is not available. Might be caused by parallel processing.")
        self.mean_curve = np.mean(self.curve_list, axis=0)

        # area under the curve; approx.
        mean_area_under_the_curve = auc(self.threshold_variations, self.mean_curve)
        std_area_under_the_curve = np.float(np.std(self.area_under_the_curve_list))
        # mean_equal_error
        mean_equal_error_rate = np.float(np.mean(self.equal_error_list))
        std_equal_error_rate = np.float(np.std(self.equal_error_list))
        return mean_area_under_the_curve, std_area_under_the_curve, mean_equal_error_rate, std_equal_error_rate

    def get_curves_deviation_area(self):
        std_curve_values = np.std(self.curve_list, axis=0)
        upper_bound = np.minimum(self.mean_curve + std_curve_values, 1)
        lower_bound = np.maximum(self.mean_curve - std_curve_values, 0)
        return lower_bound, upper_bound

    def plot_operating_characteristics(self):
        if len(self.curve_list) == 0:
            raise Exception("Curve list for plotting DET is not available. Might be caused by parallel processing.")

        mean_auc, std_auc, mean_eer, std_eer = self.get_average_curve_and_metrics()
        lower_bound, upper_bound = self.get_curves_deviation_area()

        # create visualization traces
        trace_list = [visualizer.get_line_by_chance_trace(),
                      visualizer.get_trace_for_average_characteristic_curve(thresholds=self.threshold_variations, average_curve=self.mean_curve,
                                                                            legend_label=r'$\text{Mean DET (AUC} = %0.2f \pm %0.2f; \text{EER} = '
                                                                                         r'%0.2f $\pm$ %0.2f\)$' % (
                                                                                             mean_auc, std_auc, mean_eer, std_eer))]

        trace_list.extend(visualizer.get_trace_for_characteristic_curves(x_values_list=self.false_match_list, y_values_list=self.false_non_match_list,
                                                                         characteristic_metric_list=self.equal_error_list, label_list=self.label_list,
                                                                         legend_name="EER", align_same_labels_to_colors=True))
        trace_list.extend(visualizer.get_filled_between_traces(thresholds=self.threshold_variations, lower_bound=lower_bound,
                                                               upper_bound=upper_bound))

        # plot vis
        visualizer.visualize_trace_list(trace_list=trace_list, title='Decision Error Trade Off Curve',
                                        x_axis='False Match Rate', y_axis='False None Match Rate')

    def clear(self):
        self.label_list = list()
        self.mean_curve = None
        self.curve_list = list()
        # auc
        self.area_under_the_curve_list = list()
        self.false_positive_list = list()
        self.true_positive_list = list()
        # eer
        self.false_match_list = list()
        self.false_non_match_list = list()
        self.equal_error_list = list()


def compute_interpolated_equal_error_rate(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer
