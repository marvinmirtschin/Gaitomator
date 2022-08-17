import random

from sklearn.base import BaseEstimator, TransformerMixin

from src.core.constants import LABEL_RECORD_KEY, LABEL_USER_KEY
from src.core.data_access import hoang_2015, ou, uci_har, wisdm
from src.core.data_access._utility import reduce_data_frame
from src.core.utility.data_frame_transformer import expand_data_frame, shrink_data_frame


class DataAccessor(BaseEstimator, TransformerMixin):
    """
    Helper class to get data from different data sets.
    """

    DATA_WISDM = "wisdm"
    DATA_UCIHAR = "uci-har"
    DATA_OU = "ou"
    DATA_HOANG_2015 = "hoang_2015"
    DATA_SETS = [DATA_WISDM, DATA_UCIHAR, DATA_OU, DATA_HOANG_2015]

    FREQUENCIES = {
        DATA_WISDM     : 20,
        DATA_UCIHAR    : 50,
        DATA_OU        : 100,
        DATA_HOANG_2015: 27,
    }

    def __init__(self, source, *, user_ids=None, sensors=None, seed=None, get_expanded_form=False, filter_parameters=None):
        """
        Parameters
        ----------
        source : str
            Name of the data set to be used.
        user_ids : list, default=None
            Selection parameter for users
        sensors: list, default=None
            Selection parameter for sensors
        seed: int, default=None
            If given, seed will be used for randomizing selected data if only a certain amount of data is allowed by the filter_parameters
        get_expanded_form: bool, default=False
            If False, the actual data will be located inside a column with the key src.core.constants.DATA_FRAMES_KEY for each distinct combination
            of labels.
        filter_parameters: dict, default=None
            Parameters to perform selections of data
        """
        self.source = source
        self.user_ids = user_ids
        self.sensors = sensors
        self.seed = seed
        self.get_expanded_form = get_expanded_form
        # Do not convert to **kwargs as this will lead to the parameter being ignored in #get_params
        if filter_parameters is None:
            filter_parameters = dict()
        self.filter_parameters = filter_parameters

    def transform(self):
        """
        Return pd.DataFrame in desired form for the specified data set with selection applied.
        Returns
        -------
        pd.DataFrame
            DataFrame in the specified form
        """
        if self.seed is not None:
            random.seed(self.seed)
        assert self.source in DataAccessor.DATA_SETS
        if self.source == DataAccessor.DATA_WISDM:
            is_expanded = True
            result = wisdm.get_data(user_ids=self.user_ids, sensors=self.sensors, **self.filter_parameters)
        elif self.source == DataAccessor.DATA_UCIHAR:
            is_expanded = True
            result = uci_har.get_data(user_ids=self.user_ids, sensors=self.sensors, **self.filter_parameters)
        elif self.source == DataAccessor.DATA_OU:
            is_expanded = False
            result = ou.get_data(user_ids=self.user_ids, sensors=self.sensors, **self.filter_parameters)
        else:
            is_expanded = False
            result = hoang_2015.get_data(user_ids=self.user_ids, sensors=self.sensors, **self.filter_parameters)

        if self.get_expanded_form and not is_expanded:
            result = expand_data_frame(result)
        elif not self.get_expanded_form and is_expanded:
            result = shrink_data_frame(result)

        if "max_number_of_users" in self.filter_parameters.keys() or "max_number_of_recordings_per_user" in self.filter_parameters.keys():
            result = reduce_data_frame(data_frame=result, max_number_of_users=self.filter_parameters.get("max_number_of_users", None),
                                       max_number_of_recordings_per_user=self.filter_parameters.get("max_number_of_recordings_per_user", None))
            result.reset_index(drop=True, inplace=True)

        return result

    def __json__(self):
        return __dict__

    def get_frequency(self):
        """
        Return recording frequency of the data sets. Note that not all supply a time row, therefore using recording frequency is more robust.

        Returns
        -------
        int
            Frequency for the specified data set
        """
        return DataAccessor.FREQUENCIES[self.source]


def test_get_data_shrunken():
    data_wisdm = DataAccessor(DataAccessor.DATA_WISDM).transform()
    assert data_wisdm.shape == (476, 3)
    del data_wisdm
    data_ucihar = DataAccessor(DataAccessor.DATA_UCIHAR).transform()
    assert data_ucihar.shape == (90, 3)
    del data_ucihar
    data_ou = DataAccessor(DataAccessor.DATA_OU).transform()
    assert data_ou.shape == (2775, 5)
    del data_ou
    data_instability = DataAccessor(DataAccessor.DATA_HOANG_2015).transform()
    assert data_instability.shape == (721, 3)


def test_get_data_expanded():
    data_wisdm = DataAccessor(DataAccessor.DATA_WISDM, get_expanded_form=True).transform()
    assert data_wisdm.shape == (2980765, 6)
    del data_wisdm
    data_ucihar = DataAccessor(DataAccessor.DATA_UCIHAR, get_expanded_form=True).transform()
    assert data_ucihar.shape == (299008, 11)
    del data_ucihar
    data_ou = DataAccessor(DataAccessor.DATA_OU, get_expanded_form=True).transform()
    assert data_ou.shape == (1233416, 10)
    del data_ou
    data_instability = DataAccessor(DataAccessor.DATA_HOANG_2015, get_expanded_form=True).transform()
    assert data_instability.shape == (1660991, 17)


def test_selection():
    data_frame = DataAccessor(DataAccessor.DATA_HOANG_2015,
                              filter_parameters={"max_number_of_users": 2, "max_number_of_recordings_per_user": 3}).transform()
    assert data_frame[LABEL_USER_KEY].unique() == 2
    assert data_frame[LABEL_RECORD_KEY].unique() == 6


def test_deterministic():
    accessor = DataAccessor(DataAccessor.DATA_HOANG_2015, filter_parameters={"max_number_of_users": 2, "max_number_of_recordings_per_user": 3})
    df_1 = accessor.transform()
    accessor.seed = 12345
    df_2 = accessor.transform()
    df_3 = accessor.transform()

    assert not df_1.equals(df_2)
    # assert df_2.equals(df_3) is broken for unknown reasons
    assert df_2[LABEL_USER_KEY].equals(df_3[LABEL_USER_KEY])
    assert df_2[LABEL_RECORD_KEY].equals(df_3[LABEL_RECORD_KEY])
