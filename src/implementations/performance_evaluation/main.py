from sklearn.pipeline import Pipeline

from src.add_ins.preprocessing import AccelerationUnitTransformer
from src.add_ins.processing.features import (CorrelationCalculator, FourierCoefficientsMagnitudeCalculator,
                                             InterquartileRangeCalculator, MeanCalculator, MinMaxDifferenceCalculator,
                                             MultiLevelWaveletEnergyCalculator,
                                             StandardDeviationCalculator)
from src.add_ins.processing.scaling import SignalScaler
from src.add_ins.segmentation.hoang_2015 import CycleDetectionTransformer
from src.core import constants
from src.core.base_classes import DataLabelSplitter, NestedDataFrameTransformer
from src.core.data_access import DataAccessor
from src.core.runner import ImplementationRunner
from src.core.utility.cycle_data import CycleCleaner
from src.core.utility.data_frame_transformer import FeatureFlattener, NestedDataFrameFeatureUnion, Tabularizer
from src.core.utility.sensor_fusion import SensorSynchronizer
from src.core.utility.testing_helper import run_deterministic_test
from src.core.visualization.pipeline_visualizer import CycleDataVisualizer
from src.implementations.performance_evaluation.hot_spots import FeatureSelector, Shen2017ClassificationRunner, SplitFilter

SEED = constants.RANDOM_SEED


class Shen2017Runner(ImplementationRunner):

    def print_classification_report(self, transformed_data_frame):
        classification = Shen2017ClassificationRunner()
        classification.print_classifications(transformed_data_frame)

    @staticmethod
    def get_paper_name():
        return "shen_2018"

    def get_pipeline(self, recording_frequency, **kwargs):
        reference_sensor = "accelerometer"  # OG: "linearAcceleration", should be adjustable to data set
        column_to_use = "accelerometer_z"  # OG: "linearAcceleration_z", should be adjustable to data set
        visualize_segments = False
        values_per_second = 20  # 20 Hz
        use_performance_boost = True

        preprocessing_filter = NestedDataFrameTransformer(SplitFilter())

        # OG: does not work for data sets given
        # segmentationTransformer = Shen2017SegmentationTransformer(values_per_second=values_per_second, column_to_use=column_to_use,
        #                                                           use_performance_boost=use_performance_boost)
        segmentation_transformer = CycleDetectionTransformer(cut_at_dimension=column_to_use)
        segmentation_transformer = NestedDataFrameTransformer(segmentation_transformer)
        feature_pipeline = self.get_feature_pipeline()

        pipe = Pipeline([
            ('sensor_fusion',
             NestedDataFrameTransformer(
                 SensorSynchronizer(reference_sensor=reference_sensor, frequency=values_per_second, old_frequency=recording_frequency))),
            ('filtering', preprocessing_filter),
            ('segmentation', segmentation_transformer),
            ('tabularizer', Tabularizer()),
            ('feature_extraction', feature_pipeline),
            ('feature_flattener', FeatureFlattener()),
            ('feature_normalization', DataLabelSplitter(SignalScaler(scaling_method=SignalScaler.STANDARD_SCALER))),
            ('feature_selection', DataLabelSplitter(FeatureSelector(must_include_all_features=False)))  # This is the normal way
        ])

        if visualize_segments:
            pipe.steps.insert(3, ('cycle_visualizer', CycleDataVisualizer(sub_title=column_to_use)))
        if use_performance_boost:
            cycle_cleaner = CycleCleaner(column_index=0, cycle_deviation_threshold=2, use_std=False, use_mean=False)
            cycle_cleaner = NestedDataFrameTransformer(cycle_cleaner)
            if visualize_segments:
                pipe.steps.insert(4, ('cycle_cleaning', cycle_cleaner))
                pipe.steps.insert(5, ('cycle_visualizer', CycleDataVisualizer(sub_title=column_to_use)))
            else:
                pipe.steps.insert(3, ('cycle_cleaning', cycle_cleaner))

        if self.data_accessor.source == DataAccessor.DATA_OU or self.data_accessor.source == DataAccessor.DATA_UCIHAR:
            # data is given in G instead of m/s^2
            pipe.steps.insert(0, ('acceleration_unit_transformer', AccelerationUnitTransformer()))
        return pipe

    def get_feature_pipeline(self):
        return NestedDataFrameFeatureUnion([
            ('mean', MeanCalculator()),
            ('standard_deviation', StandardDeviationCalculator()),
            ('min_max_difference', MinMaxDifferenceCalculator()),
            ('correlation', CorrelationCalculator()),
            ('interquartile_range', InterquartileRangeCalculator()),
            # ('dynamic_time_warping', DynamicTimeWarpingCalculator()),
            ('wavelet_coefficients', MultiLevelWaveletEnergyCalculator()),
            ('fourier_coefficients', FourierCoefficientsMagnitudeCalculator(number_of_coefficients=9, skip_first=True))
        ])

    def get_configuration(self, pipeline):
        configuration = super().get_configuration(pipeline)
        feature_transformer_list = configuration['feature_extraction__transformer_list']
        feature_transformer_names = [name for name, value in feature_transformer_list]
        feature_steps = [f"feature_extraction__{name}" for name in feature_transformer_names]
        keys_to_remove = [key for key in configuration.keys() if key in feature_steps]

        for key in keys_to_remove:
            del configuration[key]

        del configuration["feature_extraction__transformer_list"]
        del configuration["filtering__transformer__filter_params"]

        return configuration


def test_classification():
    data_accessor = DataAccessor(DataAccessor.DATA_UCIHAR,
                                 filter_parameters={"max_number_of_users": 3, "max_number_of_recordings_per_user": 3, "activities": ["walking"]}
                                 , sensors=["accelerometer", "gyroscope"])
    runner = Shen2017Runner(data_accessor)
    runner.run()


def test_deterministic():
    data_accessor = DataAccessor(DataAccessor.DATA_UCIHAR,
                                 filter_parameters={"max_number_of_users": 3, "max_number_of_recordings_per_user": 3, "activities": ["walking"]}
                                 , sensors=["accelerometer", "gyroscope"])
    runner = Shen2017Runner(data_accessor)
    run_deterministic_test(runner.process_data(), "performance_linAccz_boosted")
