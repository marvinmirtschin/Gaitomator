import itertools
import math
import random

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import confusion_matrix, roc_curve

from src.core.constants import LABEL_USER_KEY


def _chain(_list):
    return list(itertools.chain(*_list))


def split_data(data_frame, labels, genuine_labels, k, label_key=LABEL_USER_KEY, use_one_class=False, used_for_identification=False):
    """


    Parameters
    ----------
    data_frame : pd.DataFrame
        Data Frame containing the data to be evaluated.
    labels : array-like

    genuine_labels
    k : float
        This portion of the data should be used as training data for each user. If k is an integer it will be used as number of training samples.
    label_key : str, default=src.constants.LABEL_USER_KEY
        Key used for the row which contains the labels in the actual_data_frame
    use_one_class : bool, default=False
        If True, uses only data of the genuine user in the train set.

    Returns
    -------

    """
    splits = list()

    if not isinstance(genuine_labels, list):
        genuine_labels = [genuine_labels]

    for label in labels:
        splits.append(_get_train_and_test_data_for_label(data_frame, label, label_key, label in genuine_labels, k,
                                                         use_one_class=use_one_class, used_for_identification=used_for_identification))

    splits = _chain(splits)
    training_data = pd.DataFrame(_chain(splits[0::4]))
    test_data = pd.DataFrame(_chain(splits[1::4]))
    training_labels = _chain(splits[2::4])
    test_labels = _chain(splits[3::4])
    return training_data, test_data, training_labels, test_labels


def _get_train_and_test_data_for_label(data_frame, label, label_key, is_genuine, k, use_one_class=False, used_for_identification=False):
    # select data frame according to given label
    selected_data_frame = data_frame[data_frame[label_key] == label]

    # number of rows to select training
    if use_one_class and not is_genuine:
        number_of_training_samples = 0
    else:
        assert k > 0
        if k < 1:
            number_of_training_samples = int(math.ceil(k * len(selected_data_frame)))
        else:
            number_of_training_samples = round(k)

    indices = np.arange(len(selected_data_frame))
    random.shuffle(indices)

    training_data = list()
    test_data = list()
    training_labels = list()
    test_labels = list()

    for i, index in enumerate(indices):
        if i < number_of_training_samples:
            if used_for_identification:
                training_labels.append(label)
            else:
                training_labels.append(1 if is_genuine else -1)
            training_data.append(selected_data_frame.iloc[index])
        else:
            if used_for_identification:
                test_labels.append(label)
            else:
                test_labels.append(1 if is_genuine else -1)
            test_data.append(selected_data_frame.iloc[index])

    return training_data, test_data, training_labels, test_labels


def calculate_confusion_matrix_values(predictions, test_labels, labels=None):
    if labels is not None and len(labels) > 2:
        conf_mat = confusion_matrix(test_labels, predictions, labels)
        number_of_correct_classifications = sum(conf_mat.diagonal())
        number_of_classifications = len(predictions)
        return number_of_correct_classifications, number_of_classifications
    else:
        tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
        return tn, fp, fn, tp


def calculate_eer(decision_values, expected_labels, return_intermediate_values=False, **kwargs):
    """
    Calculate Equal Error Rate (EER) for the given decision_vales and their real labels.

    Parameters
    ----------
    decision_values : array-like
        Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions (as
        returned by "decision_function" on some classifiers).
    expected_labels : array-like
        Real labels for the decision_values. Should be either {0, 1} or {-1, 1}.
    return_intermediate_values : bool, default=False
        Also return false acceptance rate, false rejection rate and the associated thresholds.

    Returns
    -------
    eer : Float
        Equal Error Rate (EER) for the given decisions and test labels.
    eer_threshold : Float
        Decision value threshold for the eer value.
    far : array-like
        False Acceptance Rate (FAR) for the associated thresholds.
    frr : array-like
        False Rejection Rate (FRR) for the associated thresholds.
    thresholds : array-like
        Thresholds used to distinguish the decision_values to calculate FAR and FRR.
    """
    false_positive_rate, true_positive_rate, thresholds = roc_curve(expected_labels, decision_values)
    eer, eer_threshold = calculate_equal_error_rate(false_positive_rate, true_positive_rate, thresholds, **kwargs)
    far = false_positive_rate
    frr = 1 - true_positive_rate

    if return_intermediate_values:
        return eer, eer_threshold, far, frr, thresholds
    return eer, eer_threshold


def calculate_equal_error_rate(false_positive_rate, true_positive_rate, thresholds, method="normal"):
    # methods: mean, interpolation, normal
    eer = None
    threshold = None
    if method == "interpolation":
        # way 1: using interpolation
        eer = brentq(lambda x: 1. - x - interp1d(false_positive_rate, true_positive_rate)(x), 0., 1.)
        threshold = interp1d(false_positive_rate, thresholds)(eer)
    else:
        false_negative_rate = 1 - true_positive_rate  # also the frr, whereas the false_positive_rate is also the far
        result_distances = np.absolute((false_negative_rate - false_positive_rate))
        smallest_difference_index = np.nanargmin(result_distances)

        # default way is based on the assumption that the smallest index is not in that range; therefore if the value is in that range we will
        # default to this way
        is_border_index = smallest_difference_index < 0 or smallest_difference_index > len(false_positive_rate) - 2
        if method == "normal" and is_border_index:
            print("Unable to calculate EER with the given way; Using the 'mean' approach instead")

        if method == "mean" or is_border_index:
            # way 2: using the mean of the closest two values in far and frr
            eer_p = false_positive_rate[smallest_difference_index]
            # as a sanity check the value should be close to
            eer_n = false_negative_rate[smallest_difference_index]

            # according to https://stats.stackexchange.com/questions/221562/calculate-eer-from-far-and-frr the mean of those two values is an
            # acceptable value
            # -> after inspection this seems not correct as both value arrays do not behave like curves
            eer = (eer_p + eer_n) / 2
            threshold = thresholds[smallest_difference_index]
        else:  # method == "normal"
            # way 3: check which of eer_n or eer_p is the correct value; find value where far_i is smaller than frr_i but far_i+1 is bigger than
            # frr_+1
            relevant_far = false_positive_rate[smallest_difference_index - 1: smallest_difference_index + 2]
            relevant_frr = false_negative_rate[smallest_difference_index - 1: smallest_difference_index + 2]

            old_far_value = relevant_far[0]
            for i in range(1, len(relevant_far)):
                if relevant_far[i] > relevant_frr[i]:
                    # values are chosen so that this is bound to be found
                    if relevant_far[i] == old_far_value:
                        eer = relevant_far[i]
                    else:
                        eer = relevant_frr[i]
                    threshold = thresholds[smallest_difference_index - 1 + i]
                    break
                old_far_value = relevant_far[i]

    if eer is None or threshold is None:
        raise Exception(
            f"Unable to calculate EER for false_positive_rate={false_positive_rate}, true_positive_rate={true_positive_rate} and thresholds="
            f"{thresholds} with method={method}")
    return eer, threshold
