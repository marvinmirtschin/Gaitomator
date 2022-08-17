# KNN (euclidean, mahalanobis)
#   k = 3
# SVM (linear, rbf kernel)
#   linear: g = 0.04, nu = 0.5
#   rbf: g = 0.0625, nu = 0.1
# Euclidean (classic, should_normalize_output)
#   should_normalize_output: divide detection score by ||template||_2 * ||tests||_2
# Manhattan (classic, should_normalize_output)
#   should_normalize_output: divide detection score by ||template||_1 * ||tests||_1
# Mahalanobis (classic, should_normalize_output)
#   should_normalize_output: divide detection score by ||template||_2 * ||tests||_2

# 10 fold cross validation (as explained below)
# calculate FAR and FRR for EER
# each user is used as genuine user:
#   split legitimate data into 10 parts -> each is used once for training
#   choose 30 % illegitimate data as testing dataset
#   calculate FAR, FRR, EER
import random

import numpy as np
import pandas as pd
import scipy as sp
from prettytable import PrettyTable
from sklearn.model_selection import KFold

from src.add_ins.classification.classifier import DistanceClassifier, KnnClassifier, SvmClassifier
from src.core.base_classes import split_labels
from src.core.classification import ClassificationRunner, run_classification_for_classifier
from src.core.constants import LABEL_USER_KEY


class ShenDistanceClassifier(DistanceClassifier):
    DISTANCE_METHODS = ["euclidean", "manhattan", "mahalanobis"]

    def __init__(self, distance_method, should_normalize_output=False, use_paper_formula=True, scaled=False, **kwargs):
        # TODO: improve usage to allow avg of distances instead of ONLY average of template
        assert distance_method in ShenDistanceClassifier.DISTANCE_METHODS
        self.should_normalize_output = should_normalize_output
        self.use_paper_formula = use_paper_formula
        self.full_template = None
        super().__init__(distance_method=distance_method, scaled=scaled, **kwargs)

    def fit(self, data_frame: pd.DataFrame, y=None, **kwargs):
        super().fit(data_frame, y)
        self.full_template = data_frame
        mean = data_frame.mean()

        if self.distance_method == "mahalanobis":
            if self.template.shape[0] < self.template.shape[1]:
                print(f"Unable to calculate covariance matrix needed for the mahalanobis distance calculation, as the template has shape " +
                      f"{self.template.shape} Defaulting back to euclidean distance.")
                self.distance_method = "euclidean"

        self.template = mean.to_frame().transpose()
        return self

    def _transform(self, data):
        import scipy.spatial.distance as distance
        assert self.template is not None

        if self.distance_method == "mahalanobis":
            inverse_covariance_matrix = np.cov(self.full_template.values.T)
            inverse_covariance_matrix = sp.linalg.inv(inverse_covariance_matrix)
            if self.use_paper_formula:
                result = data.apply(lambda column: distance.mahalanobis(self.template, column, inverse_covariance_matrix), axis=1)
            else:
                result = data.apply(lambda column: distance.mahalanobis(column, self.template, inverse_covariance_matrix), axis=1)
        else:
            # make sure to to revert negation for now
            result = super()._transform(data).apply(lambda x: -1 * x)

        if self.should_normalize_output:
            # norm by dividing with ||vmean ||_ord ||vtest ||_ord .
            if self.distance_method == "manhattan":
                _ord = 1
            else:
                _ord = 2
            # norm_train = np.linalg.norm(self.template.mean(), ord=_ord)
            norm_train = np.linalg.norm(self.template, ord=_ord)
            norm_test = data.apply(np.linalg.norm, axis=1, ord=_ord)
            norm = norm_train * norm_test
            result = result / norm

        # The greater the distance, the less the probability of the sample belonging to the trained user
        # invert distance to be used in roc_curve
        return result.apply(lambda x: -1 * x)


def test_distance_methods():
    filepath = 'https://raw.githubusercontent.com/selva86/datasets/master/diamonds.csv'
    df = pd.read_csv(filepath).iloc[:, [0, 4, 6]]
    df.head()
    distance_classifier = ShenDistanceClassifier(distance_method="euclidean")

    df_x = df[['carat', 'depth', 'price']].head(500)
    distance_classifier.fit(data_frame=df[['carat', 'depth', 'price']])
    df_x['euclidean'] = distance_classifier.transform(df_x[['carat', 'depth', 'price']])
    assert not df_x['euclidean'].isna().any()

    distance_classifier.distance_method = "manhattan"
    distance_classifier.fit(data_frame=df[['carat', 'depth', 'price']])
    df_x['manhattan'] = distance_classifier.transform(df_x[['carat', 'depth', 'price']])
    assert not df_x['manhattan'].isna().any()

    distance_classifier.distance_method = "mahalanobis"
    distance_classifier.fit(data_frame=df[['carat', 'depth', 'price']])
    df_x['mahalanobis'] = distance_classifier.transform(df_x[['carat', 'depth', 'price']])
    assert not df_x['mahalanobis'].isna().any()


class Shen2017ClassificationRunner(ClassificationRunner):
    def get_classifiers(self):
        return [KnnClassifier(k=3, distance_method="euclidean", use_one_class=True),
                KnnClassifier(k=3, distance_method="mahalanobis", use_one_class=True),
                SvmClassifier(kernel="linear", gamma=0.04, nu=0.5, use_one_class=True, return_decision_values=True),
                SvmClassifier(kernel="rbf", gamma=0.0625, nu=0.1, use_one_class=True, return_decision_values=True),
                ShenDistanceClassifier(distance_method="euclidean", should_normalize_output=False),
                ShenDistanceClassifier(distance_method="euclidean", should_normalize_output=True),
                ShenDistanceClassifier(distance_method="manhattan", should_normalize_output=False),
                ShenDistanceClassifier(distance_method="manhattan", should_normalize_output=True),
                ShenDistanceClassifier(distance_method="mahalanobis", should_normalize_output=False),
                ShenDistanceClassifier(distance_method="mahalanobis", should_normalize_output=True)]

    def print_classifications(self, data):
        # input: extended data frame, labels =[activity, user, record, type]
        # Note: number of folds is originally 10 but that does not work for some data sets as they have to little data per participant
        number_of_folds = 2

        data.dropna(inplace=True, axis=0, how="any")

        users = list(data[LABEL_USER_KEY].unique())

        results = dict()
        for classifier in self.get_classifiers():
            results[str(classifier)] = list()

        for genuine_user in users:
            genuine_data = data[data[LABEL_USER_KEY] == genuine_user]
            imposter_data = data[data[LABEL_USER_KEY] != genuine_user]

            cv = KFold(n_splits=number_of_folds, random_state=1, shuffle=True)
            splits = cv.split(genuine_data)

            for split in splits:
                train_data, y_train = split_labels(genuine_data.iloc[split[0]])
                genuine_test_data, genuine_y_test = split_labels(genuine_data.iloc[split[1]])

                index = list(imposter_data.index)
                random.shuffle(index)
                number_of_used_indices = int(0.3 * len(index))
                index = index[:number_of_used_indices]
                imposter_test_data, imposter_y_test = split_labels(imposter_data.loc[index])

                test_data = pd.concat([genuine_test_data, imposter_test_data])

                y_train = len(train_data) * [1]
                y_test = len(imposter_test_data) * [-1] + len(genuine_test_data) * [1]

                for classifier in self.get_classifiers():
                    eer = run_classification_for_classifier(classifier, train_data=train_data, test_data=test_data, y_train=y_train, y_test=y_test)
                    results[str(classifier)].append(eer)

        for classifier in self.get_classifiers():
            results[str(classifier)] = np.mean(results[str(classifier)])

        print("\nEqual Error Rates for Classifiers")
        print_result(results)


def print_result(results_dict):
    my_table = PrettyTable()
    my_table.field_names = list(results_dict.keys())  # TODO: more readable
    my_table.add_row(list(results_dict.values()))
    print(my_table)
