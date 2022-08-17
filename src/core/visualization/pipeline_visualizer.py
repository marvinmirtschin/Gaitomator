import copy
import itertools
import os
import time
import warnings

import pandas as pd
import plotly
import plotly.graph_objs as go

from src.core.base_classes import NestedDataFrameTransformer, SafeRowwiseTransformer, SafeTransformer, split_labels
from src.core.constants import DIMENSIONS_KEY_LIST, LABEL_RECORD_KEY
from src.core.utility import cycle_data
from src.core.utility.file_handling import get_or_create_html_visualizations_directory
from src.core.visualization import visualizer
from src.core.visualization.visualizer import get_hex_color


class CycleDataVisualizer(SafeRowwiseTransformer):

    def __init__(self, cycle_detection_method=None, max_number_of_plots=-1, sub_title=None, plot_concatenated_cycles=True, **kwargs):
        self.cycle_detection_method = cycle_detection_method
        self.max_number_of_plots = max_number_of_plots
        self.sub_title = sub_title if sub_title else ""
        self.plot_concatenated_cycles = plot_concatenated_cycles
        self.counter = None
        self.label_iterator = None
        SafeRowwiseTransformer.__init__(self, **kwargs)

    def transform(self, data):
        self.counter = 0

        data_new = data.copy(deep=True)
        y_new = None
        if self.should_split_labels:
            data_new, y_new = split_labels(data=data)

        self.label_iterator = iter(y_new[LABEL_RECORD_KEY].values)
        transformed_data = self._perform_per_row(SafeTransformer.safe_transform, data_new)

        # append y to transformed_data
        return pd.concat([transformed_data, y_new], axis=1)

    def _transform(self, data_frame_list):
        if self.max_number_of_plots == -1 or self.counter < self.max_number_of_plots:
            self.counter += 1
            title = "Cycle Evaluation: {}".format(next(self.label_iterator))
            if self.cycle_detection_method:
                title += " ({})".format(self.cycle_detection_method)
            if self.plot_concatenated_cycles:
                visualizer.visualize_concatenated_cycles_from_list(cycle_list=data_frame_list, title=title, sensor="", subtitle=self.sub_title)
            else:
                visualizer.visualize_single_cycles_from_list(cycle_list=data_frame_list, title=title, sensor="", subtitle=self.sub_title)
        return data_frame_list


class DataSampleVisualizer(SafeRowwiseTransformer):

    def __init__(self, cycle_detection_method=None, max_number_of_plots=-1, error_handler=None, allowed_exceptions=None, default_item=None,
                 should_split_labels=True):
        self.cycle_detection_method = cycle_detection_method
        self.max_number_of_plots = max_number_of_plots
        self.counter = None
        self.label_iterator = None
        SafeRowwiseTransformer.__init__(self, should_split_labels=should_split_labels, error_handler=error_handler,
                                        allowed_exceptions=allowed_exceptions,
                                        default_item=default_item)

    def transform(self, data):
        self.counter = 0

        data_new = data.copy(deep=True)
        y_new = None
        if self.should_split_labels:
            data_new, y_new = split_labels(data=data)

        try:
            # Backwards compatibility
            self.label_iterator = iter(y_new["label_recordings"].values)
        except KeyError:
            self.label_iterator = iter(y_new[LABEL_RECORD_KEY].values)
        transformed_data = self._perform_per_row(SafeTransformer.safe_transform, data_new)

        # append y to transformed_data
        return pd.concat([transformed_data, y_new], axis=1)

    def _transform(self, data_frame):
        if self.max_number_of_plots == -1 or self.counter < self.max_number_of_plots:
            self.counter += 1
            title = "Cycle Evaluation: {}".format(next(self.label_iterator))
            if self.cycle_detection_method:
                title += " ({})".format(self.cycle_detection_method)
            visualizer.visualize_single_cycles_from_list(cycle_list=[data_frame, ], title=title, sensor="")
        visualizer.visualize_single_cycles_from_list(cycle_list=[data_frame, ], title="Sample Visualization", sensor="")
        return data_frame


class AccelerometerSampleVisualizer(SafeRowwiseTransformer):
    def __init__(self, title=""):
        super().__init__()
        self.title = title

    def _transform(self, data_frame):
        _regex = "{sensor_name}_{dimension}".format(sensor_name="accelerometer", dimension=DIMENSIONS_KEY_LIST)
        accelerometer_data = data_frame.filter(regex=_regex, axis=1)
        visualizer.visualize_single_cycles_from_list(cycle_list=[accelerometer_data.dropna(how='all'), ], title="Sample Visualization", sensor="",
                                                     subtitle=self.title)
        return data_frame


class CleaningDifferenceVisualizer(SafeTransformer):
    """
    Will run the normal and the interpolated cleaning methods and visualize the raw cycles and interpolated cycles with colors based on whether the
    cycle was never cleaned, exclusively cleaned by one or cleaned by both.
    """

    def __init__(self, column_index=None, deviation_threshold=2, interpolated_deviation_threshold=3, interpolation_method="linear", cycle_length=100,
                 **kwargs):
        self.column_index = column_index
        self.deviation_threshold = deviation_threshold
        self.interpolated_deviation_threshold = interpolated_deviation_threshold

        self.interpolation_method = interpolation_method
        self.cycle_length = cycle_length
        super().__init__(**kwargs)

    def _transform(self, data):
        """
        Will perform cleaning methods on the raw cycles given to it. The data will be both cleaned raw and first interpolated and than cleaned. The
        resulting plot will use colors based on whether the cycle was never cleaned, exclusively cleaned by one or cleaned by both. The input will
        be returned untransformed.

        Parameters
        ----------
        data: pd.DataFrame(list(pd.DataFrame))
            Data frame with list of raw cycles as rows.

        Returns
        -------
        data: pd.DataFrame(list(pd.DataFrame))
            Data frame with list of raw cycles as rows.
        """
        # normal cleaning
        cleaning_copy = copy.deepcopy(data)
        cleaner = cycle_data.CycleCleaner(uses_interpolated_data=False, column_index=self.column_index,
                                          cycle_deviation_threshold=self.deviation_threshold, error_handler=self.error_handler,
                                          allowed_exceptions=self.allowed_exceptions, default_item=self.default_item)

        cleaned_data = cleaner.transform(cleaning_copy)

        # interpolation + cleaning
        cleaning_copy = copy.deepcopy(data)
        # TODO: is this always nested? This Class is not nested and cycle cleaner is also not nested
        interpolator = NestedDataFrameTransformer(
            cycle_data.CycleInterpolator(interpolation_method=self.interpolation_method, cycle_length=self.cycle_length,
                                         error_handler=self.error_handler, allowed_exceptions=self.allowed_exceptions,
                                         default_item=self.default_item))

        interpolated_data = interpolator.transform(cleaning_copy)
        cleaner.uses_interpolated_data = True
        cleaner.cycle_deviation_threshold = self.interpolated_deviation_threshold
        cleaned_interpolated_data = cleaner.transform(interpolated_data)

        for index in range(len(data)):
            try:
                recording_id = data.iloc[index]['label_recordings']
            except KeyError:
                try:
                    recording_id = data.iloc[index][LABEL_RECORD_KEY]
                except KeyError:
                    warnings.warn("No 'label_recordings' or '{}' column provided to print recording ids.".format(LABEL_RECORD_KEY))
                    recording_id = "unknown"
            normal = data.iloc[index, 0]
            normal_cleaned = cleaned_data.iloc[index, 0]
            interpolated = interpolated_data.iloc[index, 0]
            interpolated_cleaned = cleaned_interpolated_data.iloc[index, 0]

            only_normal = self._get_indices(normal, normal_cleaned)
            only_interpolated = self._get_indices(interpolated, interpolated_cleaned)

            # order indices into intersecting and exclusive indices
            both = only_normal.intersection(only_interpolated)
            only_normal = only_normal.difference(both)
            only_interpolated = only_interpolated.difference(both)

            # plot cycles with colors based on their index group
            self._visualize_cycles_cleaning_difference(cycle_list=normal, excl1=only_normal, excl2=only_interpolated, intersect=both,
                                                       current_label=recording_id)
            self._visualize_cycles_cleaning_difference(cycle_list=interpolated, excl1=only_normal, excl2=only_interpolated, intersect=both,
                                                       current_label=recording_id)
        return data

    @staticmethod
    def _get_indices(raw_data, cleaned_data):
        removed_indices = set()
        if len(cleaned_data) != len(raw_data):
            counter = 0
            for i, df in enumerate(raw_data):
                if counter >= len(cleaned_data) or not df.equals(cleaned_data[counter]):
                    removed_indices.add(i)
                else:
                    counter += 1
        return removed_indices

    @staticmethod
    def _get_index_group(i, excl1, excl2, intersect):
        index = 0
        index_name = "Uncleaned"
        if i in excl1:
            index = 1
            index_name = "Normal Clean"
        elif i in excl2:
            index = 2
            index_name = "Interpolated Clean"
        elif i in intersect:
            index = 3
            index_name = "Both"
        return index, index_name

    def _visualize_cycles_cleaning_difference(self, cycle_list, excl1, excl2, intersect, current_label):
        traces = list()

        number_of_cycles = len(cycle_list)
        known_indices = set()

        for i in range(0, number_of_cycles):
            cycle_df = cycle_list[i]
            cycle_df = cycle_df.reset_index()
            cycle_df.drop("index", axis=1, inplace=True)

            index, index_name = CleaningDifferenceVisualizer._get_index_group(i, excl1, excl2, intersect)

            x = cycle_df.index.tolist()
            y = list(itertools.chain.from_iterable(cycle_df.values))
            color = get_hex_color(number_of_distinct_colors=4, index=index)
            show_legend = index not in known_indices

            trace = go.Scatter(x=x, y=y, name=index_name, legendgroup=index, line={'color': color}, showlegend=show_legend)
            known_indices.add(index)
            traces.append(trace)

        fig = go.Figure(data=traces)
        title = "{}: cycles: {}, deviation normal: {}, - interpolated: {}".format(current_label, number_of_cycles, self.deviation_threshold,
                                                                                  self.interpolated_deviation_threshold)
        fig['layout'].update(title=str(title))
        file_name = os.path.join(get_or_create_html_visualizations_directory(),
                                 title.replace(" ", "_") + "_{time}.html".format(time=round(time.time())))
        plotly.offline.plot(fig, filename=file_name)
