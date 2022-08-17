import numpy as np
import pandas as pd
from prettytable import PrettyTable

from src.add_ins.classification.classifier import DistanceClassifier
from src.add_ins.processing.cleaning import CycleCleaner
from src.core.base_classes import NestedDataFrameTransformer, split_labels
from src.core.classification import run_classification_for_classifier
from src.core.classification.evaluation import split_data
from src.core.constants import LABEL_USER_KEY
from src.core.utility.data_frame_transformer import expand_data_frame
from src.implementations.cross_pocket_paper.hot_spots._cycle_handling import AverageCycleCreator


# HOT SPOT
def run_classification(data_frame):
    # Note: Performance metric is not mentioned in paper; we will use the Equal Error Rate (EER)
    k = 0.5

    users = data_frame[LABEL_USER_KEY].unique()

    data_frame.reset_index(inplace=True, drop=True)
    results = {
        "scenario_1": dict(),
        "scenario_2": dict(),
        "scenario_3": dict(),
        "scenario_4": dict(),
        "scenario_5": dict(),
        "scenario_6": dict()
    }
    for value in results.values():
        for classifier in get_classifiers():
            value[str(classifier)] = list()

    cycle_cleaner = NestedDataFrameTransformer(CycleCleaner())
    average_mean_transformer = NestedDataFrameTransformer(AverageCycleCreator(averaging_method="mean", include_std_cycle=False))

    cleaned_data = cycle_cleaner.transform(data_frame)

    average_mean_no_cleaning = average_mean_transformer.transform(data_frame)
    average_mean_cleaning = average_mean_transformer.transform(cleaned_data)

    del cleaned_data
    del cycle_cleaner
    del average_mean_transformer

    for genuine_user in users:
        data = data_frame[data_frame[LABEL_USER_KEY] == genuine_user]
        imposter = data_frame[data_frame[LABEL_USER_KEY] != genuine_user]

        data = expand_data_frame(data)
        train_data, test_data, _, y_test = split_data(data, users, genuine_user, k=k)
        train_data, _ = split_labels(train_data)

        if len(train_data) < 2:
            # need at least 2 cycles to calculate the standard deviation cycle
            continue
        if len(test_data) < 1:
            # need at least 1 test cycle to calculate roc_curve
            continue

        cleaner = CycleCleaner()
        cleaned_train_data = cleaner.transform(train_data)
        mean_creator = AverageCycleCreator(averaging_method="mean")
        median_creator = AverageCycleCreator(averaging_method="median")

        mean_cleaned = mean_creator.transform(cleaned_train_data)
        mean_uncleaned = mean_creator.transform(train_data)
        median_cleaned = median_creator.transform(cleaned_train_data)

        imposter_mean = average_mean_no_cleaning[data_frame[LABEL_USER_KEY] != genuine_user]
        imposter_mean_clean = average_mean_cleaning[data_frame[LABEL_USER_KEY] != genuine_user]
        imposter = expand_data_frame(imposter)
        imposter_mean = expand_data_frame(imposter_mean)
        imposter_mean_clean = expand_data_frame(imposter_mean_clean)

        mean_creator = AverageCycleCreator(averaging_method="mean", include_std_cycle=False)
        test_data, _ = split_labels(test_data)
        cleaned_test_data = cleaner.transform(test_data)
        test_mean = mean_creator.transform(cleaned_test_data)
        test_mean[LABEL_USER_KEY] = [genuine_user] * len(test_mean)
        test_mean_uncleaned = mean_creator.transform(test_data)
        test_mean_uncleaned[LABEL_USER_KEY] = [genuine_user] * len(test_mean_uncleaned)
        test_data[LABEL_USER_KEY] = [genuine_user] * len(test_data)

        test_data = pd.concat([test_data, imposter])
        test_mean = pd.concat([test_mean, imposter_mean])
        test_mean_uncleaned = pd.concat([test_mean_uncleaned, imposter_mean_clean])

        test_data, test_labels = split_labels(test_data)
        y_test = test_labels[LABEL_USER_KEY].apply(lambda user: 1 if user == genuine_user else -1)

        test_mean_uncleaned, test_mean_uncleaned_labels = split_labels(test_mean_uncleaned)
        y_mean_uncleaned_test = test_mean_uncleaned_labels[LABEL_USER_KEY].apply(lambda user: 1 if user == genuine_user else -1)

        test_mean, test_mean_labels = split_labels(test_mean)
        y_mean_test = test_mean_labels[LABEL_USER_KEY].apply(lambda user: 1 if user == genuine_user else -1)

        for classifier in get_classifiers():
            # scenario 1
            y_train = len(mean_cleaned) * [1]
            eer = run_classification_for_classifier(classifier, mean_cleaned, test_data, y_train, y_test,
                                                    method="interpolation")
            if eer is not np.nan:
                results["scenario_1"][str(classifier)].append(eer)
            else:
                print(f"S1 EER was nan: {1 in y_test.values} - {-1 in y_test.values}")

            # scenario 2
            y_train = len(mean_uncleaned) * [1]
            eer = run_classification_for_classifier(classifier, mean_uncleaned, test_data, y_train, y_test,
                                                    method="interpolation")
            if eer is not np.nan:
                results["scenario_2"][str(classifier)].append(eer)
            else:
                print(f"S2 EER was nan: {1 in y_test.values} - {-1 in y_test.values}")

            # scenario 3
            y_train = len(median_cleaned) * [1]
            eer = run_classification_for_classifier(classifier, median_cleaned, test_data, y_train, y_test,
                                                    method="interpolation")
            if eer is not np.nan:
                results["scenario_3"][str(classifier)].append(eer)
            else:
                print(f"S3 EER was nan: {1 in y_test.values} - {-1 in y_test.values}")

            # scenario 4
            y_train = len(mean_uncleaned) * [1]
            eer = run_classification_for_classifier(classifier, mean_uncleaned, test_mean_uncleaned, y_train, y_mean_uncleaned_test,
                                                    method="interpolation")
            if eer is not np.nan:
                results["scenario_4"][str(classifier)].append(eer)
            else:
                print(f"S4 EER was nan: {1 in y_mean_uncleaned_test.values} - {-1 in y_mean_uncleaned_test.values}")

            # scenario 5
            y_train = len(mean_cleaned) * [1]
            eer = run_classification_for_classifier(classifier, mean_cleaned, test_mean_uncleaned, y_train, y_mean_uncleaned_test,
                                                    method="interpolation")
            if eer is not np.nan:
                results["scenario_5"][str(classifier)].append(eer)
            else:
                print(f"S5 EER was nan: {1 in y_mean_uncleaned_test.values} - {-1 in y_mean_uncleaned_test.values}")

            # scenario 6
            eer = run_classification_for_classifier(classifier, mean_cleaned, test_mean, y_train, y_mean_test,
                                                    method="interpolation")
            if eer is not np.nan:
                results["scenario_6"][str(classifier)].append(eer)
            else:
                print(f"S6 EER was nan: {1 in y_mean_test.values} - {-1 in y_mean_test.values}")

    _scenario_1(results)
    _scenario_2(results)
    _scenario_3(results)


def _scenario_1(results: dict):
    # template with vs without clean
    print("\n\nTemplate creation with Outlier Removal vs without")
    with_cleaning = results["scenario_1"]
    classifiers = list(with_cleaning.keys())
    eers_clean = np.mean(list(with_cleaning.values()), axis=1)
    without_cleaning = results["scenario_2"]
    assert list(without_cleaning.keys()) == classifiers
    eers_without_clean = np.mean(list(without_cleaning.values()), axis=1)

    _print_results(["Classifier", "EER with Outlier Removal", "EER without Outlier Removal"], classifiers, eers_clean, eers_without_clean)


def _scenario_2(results: dict):
    # mean vs median metric -> template_clean=True, test_clean=False, test_avg=False
    print("\n\nTemplate creation with mean vs with median")
    with_mean = results["scenario_1"]  # we already have this
    classifiers = list(with_mean.keys())
    eers_mean = np.mean(list(with_mean.values()), axis=1)
    with_median = results["scenario_3"]
    assert list(with_median.keys()) == classifiers
    eers_median = np.mean(list(with_median.values()), axis=1)

    _print_results(["Classifier", "EER with Mean", "EER with Median"], classifiers, eers_mean, eers_median)


def _scenario_3(results: dict):
    # test as template with all outlier options -> average_metric=mean
    test_avg = True
    average_metric = "mean"
    print("\n\nTemplate creation with mean vs with median")
    d = results["scenario_4"]
    classifiers = list(d.keys())
    eers_1 = np.mean(list(d.values()), axis=1)
    d = results["scenario_5"]
    assert list(d.keys()) == classifiers
    eers_2 = np.mean(list(d.values()), axis=1)
    d = results["scenario_6"]
    assert list(d.keys()) == classifiers
    eers_3 = np.mean(list(d.values()), axis=1)

    _print_results(["Classifier", "Test Template without Outlier Removal", "Test Template with only Template Outlier Removal",
                    "Test Template with Outlier Removal"], classifiers, eers_1, eers_2, eers_3)


def _print_results(column_names, classifiers, eers_1, eers_2, eers_3=None):
    my_table = PrettyTable()
    my_table.field_names = column_names

    classifiers = list(_get_readable_classifier_names(classifiers))

    for i in range(len(classifiers)):
        if eers_3 is None:
            my_table.add_row([classifiers[i], eers_1[i], eers_2[i]])
        else:
            my_table.add_row([classifiers[i], eers_1[i], eers_2[i], eers_3[i]])
    print(my_table)


def _get_readable_classifier_names(classifiers):
    for classifier in classifiers:
        if "manhattan" in classifier:
            if "scaled=True" in classifier:
                yield "Scaled Manhattan"
            else:
                yield "Manhattan"
        else:
            if "scaled=True" in classifier:
                yield "Scaled Euclidean"
            else:
                yield "Euclidean"


# HOT_SPOT
def get_classifiers():
    return [
        DistanceClassifier(distance_method="manhattan", scaled=True),
        DistanceClassifier(distance_method="euclidean", scaled=True),
        DistanceClassifier(distance_method="manhattan"),
        DistanceClassifier(distance_method="euclidean")
    ]
