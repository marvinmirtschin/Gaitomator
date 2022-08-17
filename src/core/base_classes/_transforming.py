import warnings

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.core.constants import DATA_FRAMES_KEY, LABEL_RECORD_KEY, LABEL_USER_KEY
from src.core.error_handling import PipelineErrorHandler
from src.core.error_handling.exceptions import EmptyDataFrameException


class SafeTransformer(BaseEstimator, TransformerMixin):
    """
    Class used for sklearn.Pipeline to enable dynamic error handling. Each extending class needs to implement the _transform method. If
    allowed_exceptions are provided, all exceptions of these types are excepted in _transform and the default_item will be returned.

    TODO: Error handling is only implemented in transform, not in fit for now.
    """

    def __init__(self, error_handler=None, allowed_exceptions=None, default_item=None):
        self.allowed_exceptions = allowed_exceptions
        self.default_item = default_item
        if error_handler is None:
            self.error_handler = PipelineErrorHandler.get_instance()
        else:
            self.error_handler = error_handler

    def fit(self, data_frame, y=None, **fit_params):
        return self.safe_fit(data=data_frame, y=y, **fit_params)

    def safe_fit(self, data, y=None, **fit_params):
        if self.error_handler:
            with self.error_handler.new_case(caller=self, allowed_exceptions=self.allowed_exceptions):
                return self._fit(data, y, **fit_params)
        return self._fit(data, y, **fit_params)

    def _fit(self, data, y=None, **fit_params):
        return self

    def transform(self, data):
        return self.safe_transform(data)

    def safe_transform(self, data):
        if self.error_handler:
            with self.error_handler.new_case(caller=self, allowed_exceptions=self.allowed_exceptions):
                return self._transform(data)
            print("Resulting to default item in {}".format(self.__class__.__name__))
            return self.default_item
        return self._transform(data)

    def _transform(self, data):
        raise NotImplementedError("Implement the '_transform(self, data)' method in your class.")

    def __json__(self):
        return __dict__


class FeatureCalculator(SafeTransformer):

    def __init__(self, feature_name, raw_output=False, **kwargs):
        self.feature_name = feature_name
        self.raw_output = raw_output
        SafeTransformer.__init__(self, **kwargs)

    def transform(self, data):
        if self.raw_output:
            result = SafeTransformer.transform(self, data=data)
            if isinstance(result, (pd.DataFrame, pd.Series)):
                return result.to_numpy()
            else:
                return result
        else:
            return self._prepare_result(SafeTransformer.transform(self, data=data))

    def _prepare_result(self, result):
        if isinstance(result, pd.DataFrame):
            result.columns = [self.feature_name + "_{}".format(column) for column in result.columns]
            return result
        elif isinstance(result, pd.Series):
            result.index = [self.feature_name + "_{}".format(column) for column in result.index]
            return result.to_frame().T

    def _transform(self, data):
        raise NotImplementedError("Implement the '_transform(self, data)' method in your class.")


# noinspection PyTypeChecker
class NestedDataFrameTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, transformer: SafeTransformer, axis=1, parameter=None):
        self.transformer = transformer
        self.axis = axis
        self.parameter = parameter

    def fit(self, data: pd.DataFrame, y=None, **fit_params):
        if self.parameter:
            if data.empty:
                raise EmptyDataFrameException()

            user_id = self.parameter[LABEL_USER_KEY] if LABEL_USER_KEY in self.parameter.keys() else None
            record_id = self.parameter[LABEL_RECORD_KEY] if LABEL_RECORD_KEY in self.parameter.keys() else None

            if user_id:
                data = data[data[LABEL_USER_KEY] == user_id]
            if record_id:
                data = data[data[LABEL_RECORD_KEY] == record_id]

        dfs = list()
        for i in range(len(data)):
            row = data.iloc[i]
            df = row[DATA_FRAMES_KEY].copy(deep=True)
            if LABEL_USER_KEY in data.columns:
                df[LABEL_USER_KEY] = [row[LABEL_USER_KEY]] * len(df)
            if LABEL_RECORD_KEY in data.columns:
                df[LABEL_RECORD_KEY] = [row[LABEL_RECORD_KEY]] * len(df)
            dfs.append(df)

        if len(dfs) == 0:
            raise ValueError(f"No objects to concatenate for results of {self.transformer}")
        data = pd.concat(dfs)
        self.transformer.fit(data, y, **fit_params)  # this returns the self.transformer
        return self

    def transform(self, data: pd.DataFrame):
        data, labels = split_labels(data)
        result = data[DATA_FRAMES_KEY].apply(self.transformer.transform)
        labels[DATA_FRAMES_KEY] = result
        return labels


class DataLabelSplitter(BaseEstimator):

    def __init__(self, transformer):
        self.transformer = transformer
        self.last_fitted_label = None

    # TODO: use_y could be changed into a class parameter
    def fit(self, data, y=None, use_y=False):
        data_new, y_new = split_labels(data)
        self.last_fitted_label = y_new

        if y is None:
            # pass y (fitting without labels)
            self.transformer.fit(data_new, y=None)
        elif use_y:
            self.transformer.fit(data_new, y=y)
        else:
            # if y is not None, use y_new
            self.transformer.fit(data_new, y_new)
        return self

    def fit_transform(self, data, y=None):
        data_new, y_new = split_labels(data)
        self.last_fitted_label = y_new

        if y is None:
            # pass y (fitting without labels)
            transformed_data = self.transformer.fit_transform(data_new, y)
        else:
            # if y is not None, use y_new
            transformed_data = self.transformer.fit_transform(data_new, y_new)

        # if indices are different the resulting label will contain 'nan' in all fields where only one of them has data for the index
        # append labels
        return pd.concat([pd.DataFrame(transformed_data, index=y_new.index), y_new], axis=1)

    def transform(self, data):
        data_new, y_new = split_labels(data)
        transformed_data = self.transformer.transform(data_new)
        # if indices are different the resulting label will contain 'nan' in all fields where only one of them has data for the index
        try:
            return pd.concat([pd.DataFrame(transformed_data, index=y_new.index), y_new], axis=1)
        except ValueError:
            return pd.concat([transformed_data, y_new], axis=1)

    def score(self, data, y=None):
        data_new, y_new = split_labels(data)

        # score = self.transformer.score(data_new, y_new, fitted_labels=self.last_fitted_label)
        score = self.transformer.score(data_new, y)
        # self.last_fitted_label = None
        return score

    def fit_predict(self, data, y):
        data_new, y_new = split_labels(data)
        self.last_fitted_label = y_new

        if y is None:
            y_pred = self.transformer.fit_predict(data_new, y)
        else:
            y_pred = self.transformer.fit_predict(data_new, y_new)

        return y_pred

    def predict(self, data):
        data_new, y_new = split_labels(data)
        y_pred = self.transformer.predict(data_new)
        return y_pred

    def decision_function(self, data):
        # TODO: this is tailor-made for svm, need a more general way to use methods from contained transformer
        data_new, y_new = split_labels(data)
        y_pred = self.transformer.decision_function(data_new)
        return y_pred


def split_labels(data):
    copy = data.copy(deep=True)

    # slice labels and data
    labels = copy.filter(regex='label')
    copy = copy[copy.columns.drop(list(copy.filter(regex='label')))]

    if labels.empty:
        warnings.warn("Data frame does not have any columns named by a substring 'label'. Column names are {}".format(data.columns))

    return copy, labels


class RowwiseTransformer(BaseEstimator, TransformerMixin):
    """
    Base class for transformers which should be used in sklearn.Pipeline's. Each extending class needs to implement the _transform method.
    Extending this will lead to _transform being called for each column of the pandas.DataFrame.

    TODO: Rowwise handling is only implemented in transform, not in fit for now.
    """

    def fit(self, data_frame: pd.DataFrame, y=None, **fit_params):
        """

        Parameters
        ----------
        data_frame
        y

        Returns
        -------

        """
        return self

    def transform(self, data):
        return self._perform_per_row(self._transform, data)

    def _perform_per_row(self, method, data):
        """
        This method is based on the 2nd attempt of the sktime.transformers.compose.RowwiseTransformer.transform method. It takes a pd.DataFrame as
        input, which contains multiple pd.DataFrames in its cells. Then, for each pd.DataFrame the given method is called, in a column-wise order.
        The result will then wield the transformed pd.DataFrames in its cells.

        Parameters
        ----------
        method : method
            Method which should be performed for each column of the data frame.
        data : pd.DataFrame
            Data frame to be transformed. Contains data frames as columns for which the method should be called separately.

        Returns
        -------
            Data Frame with data frames as rows which were used individually.
        """
        m_list = []
        for _, col in data.items():
            # To pass self into the method, the method must be called on self with __getattribute__
            series = pd.Series(col.apply(self.__getattribute__(method.__name__)))
            m_list.append(series)

        return pd.concat(m_list, axis=1)

    def _transform(self, data):
        raise NotImplementedError("Implement the '_transform(self, data)' method in your class.")


class SafeRowwiseTransformer(RowwiseTransformer, SafeTransformer):
    """
    Base class for transformers which should be used in sklearn.Pipeline's. Each extending class needs to implement the _transform method.
    Extending this will lead to the _transform being for each column of the pandas.DataFrame. If allowed_exceptions are provided, all exceptions of
    these types are excepted in _transform and the default_item will be returned.

    TODO: Rowwise handling is only implemented in transform, not in fit for now.
    TODO: Error handling is only implemented in transform, not in fit for now.
    """

    def __init__(self, should_split_labels=True, **kwargs):
        self.should_split_labels = should_split_labels
        SafeTransformer.__init__(self, **kwargs)

    def transform(self, data: pd.DataFrame):
        data_copy = data.copy(deep=True)
        y_new = None
        if self.should_split_labels:
            data_copy, y_new = split_labels(data=data_copy)

        transformed_data = self._perform_per_row(SafeTransformer.safe_transform, data_copy)

        # if indices are different the resulting label will contain 'nan' in all fields where only one of them has data for the index
        transformed_data.index = y_new.index
        return pd.concat([transformed_data, y_new], axis=1)

    def fit(self, data_frame: pd.DataFrame, y=None, **fit_params):
        # TODO: rowwise fitting?
        return SafeTransformer.fit(self, data_frame, y, **fit_params)

    def _transform(self, data):
        # as it is private it needs to be duplicated to show 'Implement abstract method' warning
        raise NotImplementedError("Implement the '_transform(self, data)' method in your class.")
