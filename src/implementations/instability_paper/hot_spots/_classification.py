import copy

import numpy as np
import pandas as pd
from prettytable import PrettyTable
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve

from src.add_ins.classification.classifier import KnnClassifier, SvmClassifier
from src.core.base_classes import DataLabelSplitter, split_labels
from src.core.classification import ClassificationRunner
from src.core.classification.evaluation import calculate_confusion_matrix_values, calculate_equal_error_rate, split_data
from src.core.constants import LABEL_USER_KEY
from src.implementations.performance_evaluation.hot_spots import ShenDistanceClassifier


class VerificationRunner(ClassificationRunner):

    def __init__(self, k=0.4, seed=None, use_given_number_of_principal_components=True, normalize_before_pca=False, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.seed = seed
        self.use_given_number_of_principal_components = use_given_number_of_principal_components
        self.normalize_before_pca = normalize_before_pca

    def print_classifications(self, data):
        users = set(data[LABEL_USER_KEY])

        eers = dict()
        for classifier in self.get_classifiers():
            eers[str(classifier)] = list()

        # for user in users -> genuine user
        for genuine_user in users:
            imposters = copy.copy(users)
            imposters.remove(genuine_user)

            # get user data -> split according to k
            training_data, test_data, training_labels, test_labels = split_data(data.copy(deep=True), users, genuine_user, self.k)
            training_data.reset_index(inplace=True, drop=True)
            test_data.reset_index(inplace=True, drop=True)

            # pca.fit_transform train_part => template
            # pca.transform test_part with imposter data
            # normalize train and test
            training_data, test_data = reduce_and_normalize_data(training_data, test_data, seed=self.seed,
                                                                 use_given_number_of_principal_components=self.use_given_number_of_principal_components,
                                                                 normalize_before_pca=self.normalize_before_pca)
            training_data, _ = split_labels(training_data)
            test_data, _ = split_labels(test_data)

            for classifier in self.get_classifiers():
                classifier.fit(training_data, training_labels)
                distances = classifier.transform(test_data)
                false_positive_rate, true_positive_rate, thresholds = roc_curve(test_labels, distances)
                eer, eer_threshold = calculate_equal_error_rate(false_positive_rate, true_positive_rate, thresholds)
                eers[str(classifier)].append(eer)

        for classifier in self.get_classifiers():
            eers[str(classifier)] = np.mean(eers[str(classifier)])

        print("\nEqual Error Rates for Verification Scenario")
        print_result(eers)

    def get_classifiers(self):
        # NOTE: specified nu is infeasible -> use normal SVM
        return [ShenDistanceClassifier(distance_method="euclidean"), SvmClassifier(kernel="linear", return_decision_values=True, nu=None)]


class IdentificationRunner(ClassificationRunner):

    def __init__(self, k=0.4, seed=None, use_given_number_of_principal_components=True, normalize_before_pca=False, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.seed = seed
        self.use_given_number_of_principal_components = use_given_number_of_principal_components
        self.normalize_before_pca = normalize_before_pca

    def print_classifications(self, data):
        users = set(data[LABEL_USER_KEY])

        # for user in users -> genuine user

        # get user data -> split according to k
        training_data, test_data, training_labels, test_labels = split_data(data.copy(deep=True), users, users, self.k, used_for_identification=True)
        training_data.reset_index(inplace=True, drop=True)
        test_data.reset_index(inplace=True, drop=True)

        # pca.fit_transform train_part => template
        # pca.transform test_part with imposter data
        # normalize train and test
        training_data, test_data = reduce_and_normalize_data(training_data, test_data, seed=self.seed,
                                                             use_given_number_of_principal_components=self.use_given_number_of_principal_components,
                                                             normalize_before_pca=self.normalize_before_pca)

        training_data, _ = split_labels(training_data)
        test_data, _ = split_labels(test_data)

        accuracies = dict()
        for classifier in self.get_classifiers():
            classifier.fit(training_data, training_labels)
            predictions = classifier.transform(test_data)
            number_of_correct_classifications, number_of_classifications = calculate_confusion_matrix_values(predictions, test_labels,
                                                                                                             labels=list(users))
            accuracy = number_of_correct_classifications / number_of_classifications
            accuracies[str(classifier)] = accuracy

        for classifier in self.get_classifiers():
            accuracies[str(classifier)] = np.mean(accuracies[str(classifier)])

        print("\nAccuracies for Identification Scenario")
        print_result(accuracies)

    def get_classifiers(self):
        return [KnnClassifier(k=1, distance_method="euclidean"), SvmClassifier(kernel="linear", return_decision_values=False)]


def print_result(results_dict):
    my_table = PrettyTable()
    my_table.field_names = list(results_dict.keys())  # TODO: more readable
    my_table.add_row(list(results_dict.values()))
    print(my_table)


def reduce_and_normalize_data(training_data, test_data, seed, use_given_number_of_principal_components, normalize_before_pca):
    # PCA + NORMALIZATION
    # The input data will be centered but not scaled for each feature before applying the SVD.
    # OG: By default, pca centers the data and uses the singular value decomposition (SVD) algorithm.
    #  see: https://de.mathworks.com/help/stats/pca.html#d123e644689, formerly `[coeff,score, latent] = princomp(X)`
    if use_given_number_of_principal_components:
        # use this variant only when using the original data set
        # number of principal components returned for `alpha` = 0.995
        pca = DataLabelSplitter(PCA(n_components=42, random_state=seed))
    else:
        # Paper: `alpha` = 0.995
        pca = DataLabelSplitter(PCA(n_components=0.995, random_state=seed))

    # only seen in code, not mentioned in paper
    normalizer = Normalizer(should_split_labels=True)

    if normalize_before_pca:
        normalizer_before = Normalizer(should_split_labels=True)
        # In the paper the data is not normalized before the pca. As it is all based on acceleration data this might be correct but the features
        # use different unit which should normally be accounted for. Therefore one should normalize (correlate) but this could also remove
        # important information (the variance of data).
        # see: https://stats.stackexchange.com/questions/53/pca-on-correlation-or-covariance
        transformed_train_data = normalizer_before.fit_transform(training_data)
        transformed_test_data = normalizer_before.transform(test_data)
    else:
        transformed_train_data = training_data
        transformed_test_data = test_data

    # train
    transformed_train_data = pca.fit_transform(transformed_train_data)
    # Normalize the training data to -1...1 scale
    transformed_train_data = normalizer.fit_transform(transformed_train_data)

    # test
    transformed_test_data = pca.transform(transformed_test_data)
    # normalize test data based on max min values extracted from the training phase
    transformed_test_data = normalizer.transform(transformed_test_data)

    return transformed_train_data, transformed_test_data


class Normalizer(BaseEstimator, TransformerMixin):

    def __init__(self, should_split_labels=True):
        self.should_split_labels = should_split_labels
        self.max_value_vector = None
        self.min_value_vector = None

    def fit(self, data, y=None):
        data_new = data.copy(deep=True)
        if self.should_split_labels:
            data_new, _ = split_labels(data=data_new)

        self.max_value_vector = data_new.apply(max)
        self.min_value_vector = data_new.apply(min)

    def fit_transform(self, data, y=None, **fit_params):
        self.fit(data, y)
        return self.transform(data)

    def transform(self, data):
        data_new = data.copy(deep=True)
        y_new = None
        if self.should_split_labels:
            data_new, y_new = split_labels(data=data)

        transformed_data = data_new.apply(lambda x: ((x - self.min_value_vector) / (self.max_value_vector - self.min_value_vector) - 0.5) * 2, axis=1)
        return pd.concat([transformed_data, y_new], axis=1)
