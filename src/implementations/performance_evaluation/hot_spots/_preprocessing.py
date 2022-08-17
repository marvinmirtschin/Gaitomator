import pandas as pd

from src.add_ins.preprocessing import SavitzkyGolayFilter
from src.core.base_classes import SafeTransformer
from src.core.error_handling.exceptions import EmptyDataFrameException, UnknownFilterException


class SplitFilter(SafeTransformer):
    AVAILABLE_FILTERS = ["savgol", "kalman"]

    def __init__(self, sensors=None, filter_params=None, **kwargs):
        if not sensors or len(sensors) == 0:
            sensors = ["acc", "gyro"]
        self.sensors = sensors
        self.filters = {}
        self.filter_params = filter_params
        if not filter_params or len(filter_params) == 0:
            filter_params = {"savgol": {}, "kalman": {}}
        for i, (filter_name, filter_parameters) in enumerate(filter_params.items()):
            if filter_name == "savgol":
                new_filter = SavitzkyGolayFilter(filter_parameters)
            elif filter_name == "kalman":
                # Note: the kalman filter takes very long, especially for huge data sets, without performing significantly better then the savgol
                #  filter. Therefore we change the default filter to be the savgol filter.
                new_filter = SavitzkyGolayFilter(filter_parameters)
            else:
                raise UnknownFilterException()

            self.filters[sensors[i]] = new_filter
        super(SplitFilter, self).__init__(**kwargs)

    def _transform(self, data: pd.DataFrame):
        result = []
        for sensor in self.filters.keys():
            current_data_frame: pd.DataFrame = data.loc[:, [sensor in column.lower() for column in data.columns]]
            if current_data_frame.empty:
                raise EmptyDataFrameException()
            filtered_df = self.filters[sensor].transform(current_data_frame)
            result.append(filtered_df)
        return pd.concat(result, axis=1)
