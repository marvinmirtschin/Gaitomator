import numpy as np
from sklearn.metrics import roc_curve

from src.core.classification.evaluation import calculate_equal_error_rate


def run_classification_for_classifier(classifier, train_data, test_data, y_train, y_test, method="normal"):
    try:
        classifier.fit(train_data, y_train)
        distances = classifier.transform(test_data)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, distances)
        eer, threshold = calculate_equal_error_rate(false_positive_rate, true_positive_rate, thresholds, method=method)
        return eer
    except ValueError as e:
        import traceback
        traceback.print_exc()
        return np.nan
