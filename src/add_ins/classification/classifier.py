import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.base import ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier

from src.core.base_classes import SafeTransformer
from src.core.classification._distance_metrics import calculate_scaled_euclidean_distance_for_cycles, calculate_scaled_manhattan_distance_for_cycles
from src.core.error_handling.exceptions import UnknownMethodException


class ClassificationTransformer(SafeTransformer, ClassifierMixin):

    def __init__(self, **kwargs):
        SafeTransformer.__init__(self, **kwargs)
        ClassifierMixin.__init__(self)

    def _transform(self, data):
        raise NotImplementedError("Implement the '_transform(self, data)' method in  your class.")

    def predict(self, data, threshold):
        # Predict the class labels for the provided data.
        raise NotImplementedError("Implement the '_transform(self, data)' method in  your class.")


class DistanceClassifier(SafeTransformer):
    DISTANCE_METHODS = ["euclidean", "manhattan"]
    SCALED_DISTANCE_METHODS = ["euclidean", "manhattan"]

    def __init__(self, distance_method, scaled=False, **kwargs):
        super().__init__(**kwargs)
        self.distance_method = distance_method
        self.scaled = scaled
        self.template = None

    def fit(self, data_frame: pd.DataFrame, y=None, **kwargs):
        self.template = data_frame
        return super().fit(data_frame, y)

    def _transform(self, data):
        import scipy.spatial.distance as distance
        assert self.template is not None
        if self.scaled:
            if self.distance_method == "euclidean":
                template = self.template.filter(regex="avg", axis=1)
                scaling_cycle = self.template.filter(regex="std", axis=1)
                result = pd.Series(calculate_scaled_euclidean_distance_for_cycles(template.values, scaling_cycle.values, data.values))
            elif self.distance_method == "manhattan":
                template = self.template.filter(regex="avg", axis=1)
                scaling_cycle = self.template.filter(regex="std", axis=1)
                result = pd.Series(calculate_scaled_manhattan_distance_for_cycles(template.values, scaling_cycle.values, data.values))
            else:
                raise UnknownMethodException(f"Unknown scaled distance method [{self.distance_method}] given." +
                                             f"Try one of the following {DistanceClassifier.SCALED_DISTANCE_METHODS}")
        else:
            try:
                template = self.template[data.columns]
            except KeyError:
                # handle bours_2018 paper with cycle as avg and std
                template = self.template.filter(regex="avg", axis=1)
            if self.distance_method == "euclidean":
                result = data.apply(lambda row: distance.euclidean(template.values, row), axis=1, raw=True)
            elif self.distance_method == "manhattan":
                result = data.apply(lambda row: distance.cityblock(template.values, row), axis=1, raw=True)
            else:
                raise UnknownMethodException(f"Unknown distance method [{self.distance_method}] given." +
                                             f"Try one of the following {DistanceClassifier.DISTANCE_METHODS}")
        return result.apply(lambda x: -1 * x)


class KnnClassifier(SafeTransformer):
    # @see https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/
    def __init__(self, k=3, distance_method="euclidean", use_one_class=False, **kwargs):
        if k < 1:
            raise Exception("k needs to be greater than 0 for the KNN Classifier.")
        self.k = k
        self.distance_method = distance_method
        self.use_one_class = use_one_class

        if distance_method == "euclidean":
            self.classifier = KNeighborsClassifier(n_neighbors=k, p=2)
        elif distance_method == "mahalanobis":
            self.classifier = KNeighborsClassifier(n_neighbors=k, metric="mahalanobis")
        else:
            raise Exception("Distance function used in KNN Classifier should be either 'euclidean' or 'mahalanobis', not {}".format(distance_method))

        super().__init__(**kwargs)

    def fit(self, data_frame, y=None, **kwargs):
        self.classifier.fit(X=data_frame, y=y)
        return super().fit(data_frame, y)

    def _transform(self, data):
        if self.use_one_class:
            distances, _ = self.classifier.kneighbors(data, return_distance=True)
            distances = np.average(distances, axis=1)
            return - distances
        else:
            return self.classifier.predict(data)


class SvmClassifier(SafeTransformer):
    # @see https://scikit-learn.org/stable/modules/svm.html

    def __init__(self, kernel='linear', gamma='scale', nu=0.5, return_decision_values=False, use_one_class=False, **kwargs):
        assert kernel in ['linear', 'rbf']
        if use_one_class:
            self.classifier = svm.OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
        elif nu is not None:
            self.classifier = svm.NuSVC(kernel=kernel, gamma=gamma, nu=nu)
        else:
            self.classifier = svm.SVC(kernel=kernel, gamma=gamma)
        self.kernel = kernel
        self.gamma = gamma
        self.nu = nu
        self.return_decision_values = return_decision_values
        self.use_one_class = use_one_class
        super().__init__(**kwargs)

    def fit(self, data_frame, y=None, **kwargs):
        self.classifier.fit(X=data_frame, y=y)
        return super().fit(data_frame, y)

    def _transform(self, data):
        if self.return_decision_values:
            return self.classifier.decision_function(data)
        else:
            return self.classifier.predict(data)
