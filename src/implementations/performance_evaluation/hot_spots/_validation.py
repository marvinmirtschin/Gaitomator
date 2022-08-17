import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from src.core.base_classes import SafeTransformer
from src.core.constants import LABEL_USER_KEY, RANDOM_SEED


class ValidationTransformer(SafeTransformer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _transform(self, data):
        raise NotImplementedError("Implement the '_transform(self, data)' method in  your class.")


class CrossValidationTransformer(ValidationTransformer):

    def __init__(self, classifier, genuine_user=None, n_splits=10, shuffle=True, random_state=RANDOM_SEED, **kwargs):
        self.classifier = classifier
        self.genuine_user = genuine_user
        self.cross_validator = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        super().__init__(**kwargs)

    def validate(self, data):
        return self.transform(data)

    def _transform(self, data):
        assert self.genuine_user is not None

        genuine_data = data[data[LABEL_USER_KEY] == self.genuine_user]
        imposter_data = data[data[LABEL_USER_KEY] != self.genuine_user]
        splits = self.cross_validator.split(genuine_data)

        # Paper: randomly choose 30% illegitimate data as the illegitimate testing dataset.
        _, imposter_test_data = train_test_split(imposter_data, test_size=0.3)

        scores = list()
        for train_indices, test_indices in splits:
            train_data = genuine_data.iloc[train_indices]
            genuine_test_data = genuine_data.iloc[test_indices]

            test_data = pd.concat(genuine_test_data, imposter_test_data)
            test_labels = np.hstack(np.zeros(len(genuine_test_data)), np.ones(len(imposter_test_data)))

            self.classifier.fit(train_data)
            scores.append(self.classifier.score(test_data, test_labels))

        return np.mean(scores)
