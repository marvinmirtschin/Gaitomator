import numpy as np
import pandas as pd

from src.core import constants


def expand_dict_to_data_frame(data_frame_dict, *, data_key=constants.DATA_FRAMES_KEY, label_keys=None, keep_dead_end_branches=False):
    """
    Expand tree-like structure of a dict into a data frame by building rows according to path from root to leaf. The depth of the tree will be
    given by label_keys.

    Parameters
    ----------
    data_frame_dict : dict
        Dictionary to be transformed into a data frame.
    data_key : str, default=src.constants.DATA_FRAMES_KEY
        Column name for leaf objects.
    label_keys : list, optional
        Depth of dict will be determent based on the number of label keys. Once the max depth is reached, remaining object will be seen as leaf.
        The keys themself will be used as column names for the keys of the associated depth. Keys will be used in order of front to back for root
        to leaf.
    keep_dead_end_branches : bool, default=False
        If True, will fill rows with no leaves with np.nan for all missing columns. If False, those rows will be removed.

    Returns
    -------
    pd.DataFrame
        Data frame containing all (duplicated) keys of the dict in their respective column given by label_keys and the leaf objects in the column
        given by data_key.
    """
    result_matrix = dict()
    if label_keys is None:
        label_keys = [constants.LABEL_USER_KEY, constants.LABEL_RECORD_KEY]
    keys = label_keys + [data_key]

    _run_dict_to_matrix(result_matrix, data_frame_dict, keys, keep_dead_end_branches)
    data_frame = pd.DataFrame({data_key: result_matrix[data_key] if data_key in result_matrix.keys() else []})
    for key in label_keys:
        if key in result_matrix.keys():
            data_frame[key] = result_matrix[key]
        else:
            data_frame[key] = np.nan
    return data_frame[keys]


def _run_dict_to_matrix(result_dict, input_dict, result_keys, keep_dead_end_branches):
    list(_dict_to_matrix(result_dict, input_dict, result_keys, keep_dead_end_branches))


def _dict_to_matrix(result_dict, input_dict, result_keys, keep_dead_end_branches):
    if len(result_keys) > 1 and hasattr(input_dict, "keys"):
        if len(input_dict.keys()) == 0 and keep_dead_end_branches:
            # Branches ending before leaf -> ignore or fill with np.nan
            for key in result_keys:
                if key not in result_dict.keys():
                    result_dict[key] = list()
                result_dict[key].append(np.nan)
            yield 1
        else:
            for key in input_dict.keys():
                # Traverse down the path and add current key for as many paths it lies on
                result = list(_dict_to_matrix(result_dict, input_dict[key], result_keys[1:], keep_dead_end_branches))
                number_of_elements = sum(result)
                if result_keys[0] not in result_dict.keys():
                    result_dict[result_keys[0]] = list()
                result_dict[result_keys[0]] += number_of_elements * [key]
                yield number_of_elements
    elif len(result_keys) > 1 and not hasattr(input_dict, "keys"):
        # leaf found on branch before max depth
        if result_keys[0] not in result_dict.keys():
            result_dict[result_keys[0]] = list()
        result_dict[result_keys[0]].append(input_dict)
        for key in result_keys[1:]:
            if key not in result_dict.keys():
                result_dict[key] = list()
            result_dict[key].append(np.nan)
        yield 1
    else:
        # leaf found
        if result_keys[-1] not in result_dict.keys():
            result_dict[result_keys[-1]] = list()
        result_dict[result_keys[-1]].append(input_dict)
        yield 1


def test_expand_map_to_data_frame():
    map_to_transform = {"outer_key1": {"inner_key1": "leaf", "inner_key2": "leaf", "inner_key3": "leaf"}, "outer_key2": {"inner_key1": "leaf"}}
    result = expand_dict_to_data_frame(map_to_transform, data_key="leaf_key", label_keys=["outer_keys", "inner_keys"])
    assert result.shape == (4, 3)
    assert_equal_list(result.columns, ["leaf_key", "outer_keys", "inner_keys"])
    assert expand_dict_to_data_frame({}).empty

    dict_with_dead_ends = {"outer_key1": {"inner_key1": "leaf", "inner_key2": "leaf", "inner_key3": "leaf"}, "outer_key2": {}}
    df_with_dead_ends = expand_dict_to_data_frame(dict_with_dead_ends, label_keys=["outer_keys", "inner_keys"], keep_dead_end_branches=True)
    df_without_dead_ends = expand_dict_to_data_frame(dict_with_dead_ends, label_keys=["outer_keys", "inner_keys"])
    assert df_with_dead_ends.shape == (4, 3)
    assert df_without_dead_ends.shape == (3, 3)

    df_with_empty_data_row = expand_dict_to_data_frame(map_to_transform, label_keys=["1", "2", "3"])
    assert df_with_empty_data_row.shape == (4, 4)
    assert all(df_with_empty_data_row[constants.DATA_FRAMES_KEY].isna())


def assert_equal_list(list1, list2, allow_duplicates=False):
    assert len(list1) == len(list2) and (allow_duplicates or len(set(list1)) == len(set(list2)) == len(list1))
    for s in list1:
        assert s in list2
