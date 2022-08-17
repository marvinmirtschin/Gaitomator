from itertools import chain

import pandas as pd
from sklearn.pipeline import FeatureUnion

from src.core.base_classes import SafeTransformer, split_labels
from src.core.constants import DATA_FRAMES_KEY, LABEL_USER_KEY
from src.core.error_handling.exceptions import IncorrectInputTypeException


class DataFrameTransposer(SafeTransformer):
    # need of this class may hint to incorrect handling in the step before as it should be done before returning the results

    def _transform(self, data_frame):
        return data_frame.transpose()


class Tabularizer(SafeTransformer):

    def __init__(self, check_input=True, **kwargs):
        self.check_input = check_input
        super().__init__(**kwargs)

    def _transform(self, data_frame):
        """Based on sktime.transformers.compose.Tabularizer:
        Transform nested pandas dataframe into tabular data frame.

        Parameters
        ----------
        data_frame : pandas DataFrame
            Nested data frame with pandas data frames, series or numpy arrays in cells.

        Returns
        -------
        Xt : pandas DataFrame
            Transformed data frame with only primitives in cells.
        """
        if not isinstance(data_frame, pd.DataFrame):
            raise IncorrectInputTypeException(input_parameter=data_frame, expected_type=pd.DataFrame)

        # split actual_data_frame and y
        data, y = split_labels(data=data_frame)

        data_list = []
        label_list = []
        for dimension in range(data.shape[1]):
            df_list = data.iloc[:, dimension].tolist()
            for i, df in enumerate(df_list):
                label_list.extend([y.iloc[i,]] * len(df))
            data_list.extend(df_list)

        labels = pd.DataFrame(label_list).reset_index(drop=True)
        data = pd.concat(data_list).reset_index(drop=True)
        return pd.concat([data, labels], axis=1)


class DataFrameFeatureUnion(FeatureUnion):
    """
    Transforms the normal FeatureUnion output into an pd.DataFrame to match the structure of the base transformer classes.
    """

    def __init__(self, transformer_list, use_y=True):
        self.use_y = use_y
        super().__init__(transformer_list)

    def fit_transform(self, data, y=None, **fit_params):
        # TODO: usage of y_new
        data_new, y_new = split_labels(data)
        if self.use_y and y is None:
            y = y_new
        transformed_data = super().fit(data_new, y=y, **fit_params).transform(data_new)
        result = pd.DataFrame(transformed_data, index=data_new.index)
        return pd.concat([result, y], axis=1)

    def transform(self, data):
        data_new, y_new = split_labels(data)
        transformed_data = super().transform(data_new)
        result = pd.DataFrame(transformed_data, index=data_new.index)
        return pd.concat([result, y_new], axis=1)


class NestedDataFrameFeatureUnion(FeatureUnion):

    def __init__(self, transformer_list):
        from src.core.base_classes import NestedDataFrameTransformer

        nested_transformer_list = list()
        for i in range(len(transformer_list)):
            transformer_item = transformer_list[i]
            if len(transformer_item) == 2:
                name, transformer = transformer_item
                nested_transformer_list.append((name, NestedDataFrameTransformer(transformer=transformer)))
            else:
                name, transformer, parameter = transformer_item
                nested_transformer_list.append((name, NestedDataFrameTransformer(transformer=transformer, parameter=parameter)))

        transformer_list = nested_transformer_list
        FeatureUnion.__init__(self, transformer_list)

    def transform(self, data):
        data, y = split_labels(data)
        result = pd.DataFrame(FeatureUnion.transform(self, data), index=data.index)
        return pd.concat([result, y], axis=1)

    def fit_transform(self, data, y=None, **fit_params):
        FeatureUnion.fit(self, data, y, **fit_params)
        return self.transform(data)


# Utility Classes
class FeatureFlattener(SafeTransformer):
    """
    The output of the FeatureUnion is a data frame with a column for each feature. Each of the entries contains an np.ndarray with the values for
    all dimensions and sometimes multiple rows. This class is used to combine these features into one row with a column for each value.
    """

    def _transform(self, data):
        """
        This method is based on the 2nd attempt of the sktime.transformers.compose.RowwiseTransformer.transform method.
        Note that this method iterates over row instead of columns and does not convert the result to pd.Series.

        Parameters
        ----------
        data : pd.DataFrame
            Data frame to be transformed. Contains data frames as columns for which the method should be called separately.

        Returns
        -------
            Data Frame with data frames as columns which were used individually.
        """
        data, y_new = split_labels(data=data)

        m_list = []
        for _, row in data.iterrows():
            # To pass self into the method, the method must be called on self with __getattribute__
            # Note: No conversion to pd.series like in SafeRowwiseTransformer
            m_list.append(self.flatten_values(row))

        flattened_data = pd.concat(m_list, axis=1).transpose()
        return pd.concat([flattened_data, y_new], axis=1)

    @staticmethod
    def flatten_values(data):
        values = list()
        index = list()
        for data_frame in data:
            for counter, row in data_frame.iterrows():
                if len(data_frame) > 1:
                    index += [i + "_" + str(counter) for i in row.index]
                else:
                    index += list(row.index)
                for dimensional_value in row:
                    values.append(dimensional_value)
        return pd.Series(values, index=index)


def expand_data_frame(data_frame, key=DATA_FRAMES_KEY):
    """
    Takes nested data frames contained in input and tries to create a single big data frame from them.

    Parameters
    ----------
    data_frame : pd.DataFrame
        DataFrame containing further data frames in the column given by key.
    key: string, default=DATA_FRAMES_KEY
        Key for column containing the nested data frames

    Returns
    -------
    result : pd.DataFrame
        DataFrame with new rows and columns taken from the before nested data frames
    """
    if isinstance(data_frame, pd.Series):
        print("Expanding series instead of data frame")
        return expand_series(data_frame, key=key)
    data_frames = data_frame[key]
    x = data_frames.values
    result = pd.concat(data_frames.values)
    # if not uses_custom_index:
    #    data_frame_start_indices = list(np.where(result.index == 0)[0])
    #    data_frame_start_indices.append(len(result))
    #    data_frame_lengths = [data_frame_start_indices[index] - data_frame_start_indices[index - 1] for index in
    #                          range(1, len(data_frame_start_indices))]
    # else:
    data_frame_lengths = list()
    for df in data_frames:
        data_frame_lengths.append(len(df))
    remaining_columns = list(data_frame.columns.values)
    remaining_columns.remove(key)

    for column in remaining_columns:
        expanded_column = list()
        for index, length in enumerate(data_frame_lengths):
            expanded_column.append([data_frame[column].iloc[index]] * length)

        result[column] = list(chain(*expanded_column))
    return result


def expand_series(series, key=DATA_FRAMES_KEY):
    data_frames = series[key]
    result = pd.DataFrame(data_frames.values)
    data_frame_length = len(data_frames)
    remaining_columns = list(series.index.values)
    remaining_columns.remove(key)

    for column in remaining_columns:
        expanded_column = list()
        expanded_column.append([series[column]] * data_frame_length)

        result[column] = list(chain(*expanded_column))
    return result


def shrink_data_frame(data_frame, label_columns=None):
    """
    Shrink the given data frame by creating nested data frames containing the data and leaving unique combinations of identifying label columns for
    each nested data frame.

    Parameters
    ----------
    data_frame : pd.DataFrame
        Data frame for which some columns should be combined into a nested data frame.
    label_columns : iterable, optional
        Iterable containing column keys for columns for which the combination of values will be made unique and all rows matching these identifiers
        will be shrunken into the nested data frames.
    Returns
    -------
    pd.DataFrame
        Shrunken data frame containing nested data frames for data and unique rows of label combinations.
    """
    if label_columns is None:
        label_columns = [column for column in data_frame.columns if "label" in str(column)]
    label_df = data_frame[label_columns]
    label_df.drop_duplicates(inplace=True)
    label_df.reset_index(drop=True, inplace=True)

    data_frame = pd.DataFrame({DATA_FRAMES_KEY: list(shrink(data_frame, label_df))})  # TODO: use index
    return pd.concat([data_frame, label_df], axis=1)


def shrink(data_frame, label_df, reset_index=True):
    # used to avoid problems with duplications in indices
    copy_df = data_frame.copy(deep=True)
    copy_df.reset_index(inplace=True, drop=True)
    for _, row in label_df.iterrows():
        index = None
        for name, value in row.items():
            i = copy_df[copy_df[name] == value].index
            if index is None:
                index = i
            else:
                index = index.intersection(i)
        df = data_frame.iloc[index]
        df, _ = split_labels(df)
        if reset_index:
            df.reset_index(inplace=True, drop=True)
        yield df


def test_expand():
    columns = ["a", "b"]
    df1 = pd.DataFrame(data=[[2, 3], [0, 1], [4, 5]], columns=columns)
    df2 = pd.DataFrame(data=[[2, 3], [7, 8]], columns=columns)
    df = pd.DataFrame(data=[df1, df2], columns=[DATA_FRAMES_KEY])
    df[LABEL_USER_KEY] = ["user1", "user2"]
    assert expand_data_frame(df).shape == (5, 3)
    assert expand_data_frame(df.iloc[0]).shape == (3, 3)


def test_shrink():
    df = pd.DataFrame(data=[[1, 2, "a", "b"], [1, 2, "a", "b"], [3, 4, "c", "d"], [5, 6, "e", "f"]],
                      columns=["value1", "value2", "label_1", "label_2"])
    result = shrink_data_frame(df)
    assert result.shape == (3, 3)
