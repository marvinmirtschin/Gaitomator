import random

from sklearn.pipeline import Pipeline

from src.add_ins.processing.features import (AverageGaitCycleLengthCalculator, AverageMaximumAccelerationCalculator,
                                             AverageMinimumAccelerationCalculator, BinHistogramDistributionCalculator,
                                             DiscreteCosineCoefficientsMagnitude, FourierCoefficientsMagnitudeCalculator,
                                             MeanAbsoluteDifferenceCalculator, RootMeanSquareCalculator, StandardDeviationCalculator,
                                             WaveformLengthCalculator)
from src.add_ins.segmentation.hoang_2015 import CycleDetectionTransformer, SegmentationTransformer
from src.core import constants
from src.core.base_classes import DataLabelSplitter, NestedDataFrameTransformer
from src.core.data_access import DataAccessor
from src.core.runner import ImplementationRunner
from src.core.utility.data_frame_transformer import FeatureFlattener, NestedDataFrameFeatureUnion, Tabularizer
from src.core.utility.sensor_fusion import SensorSynchronizer
from src.core.utility.signal_filter_methods import MultiLevelWaveletFilter
from src.core.utility.testing_helper import run_deterministic_test
from src.core.visualization.pipeline_visualizer import CycleDataVisualizer
from src.implementations.instability_paper import (CoordinateTransformer, SensorInputTransformer)
from src.implementations.instability_paper.hot_spots import (IdentificationRunner, VerificationRunner)

SEED = constants.RANDOM_SEED


class Hoang2015Runner(ImplementationRunner):
    MATCHING_DATA_SETS = [DataAccessor.DATA_HOANG_2015]

    def print_classification_report(self, transformed_data_frame):
        verification = VerificationRunner()
        verification.print_classifications(transformed_data_frame)
        identification = IdentificationRunner()
        identification.print_classifications(transformed_data_frame)

    @staticmethod
    def get_paper_name():
        return "hoang_2015"

    def get_pipeline(self, recording_frequency, cycle_detection="original", visualize_cycles=False, frequency=100, **kwargs):
        pipeline = Pipeline([
            # Note: We first perform the sensor synchronization than the coordinate transformation as it does not change the result but helps with
            #  processing;
            ('sensor_fusion', NestedDataFrameTransformer(SensorSynchronizer(reference_sensor="linearAcceleration", frequency=27))),
            ('acceleration_transformation', NestedDataFrameTransformer(CoordinateTransformer())),
            ('new_input', NestedDataFrameTransformer(SensorInputTransformer())),
            ('filtering', NestedDataFrameTransformer(MultiLevelWaveletFilter(wavelet_name='db6', level=2))),
            ('cycle_detection', NestedDataFrameTransformer(CycleDetectionTransformer(cut_at_dimension='accelerometer_z', window_size=10))),
            ('segmentation', NestedDataFrameTransformer(SegmentationTransformer(cycles_per_segment=4, number_of_overlapping_cycles=2))),
            ('tabularizer', Tabularizer()),
            ('feature_extraction', DataLabelSplitter(self.get_feature_pipeline())),
            ('feature_flattener', FeatureFlattener()),
        ])

        if visualize_cycles:
            pipeline.steps.insert(5, ('cycle_visualizer', CycleDataVisualizer()))
        return pipeline

    @staticmethod
    def get_feature_pipeline():
        return NestedDataFrameFeatureUnion([
            ('average_maximum_acceleration', AverageMaximumAccelerationCalculator()),
            ('average_minimum_acceleration', AverageMinimumAccelerationCalculator()),
            ('mean_absolute_difference', MeanAbsoluteDifferenceCalculator()),
            ('root_mean_square', RootMeanSquareCalculator()),
            ('standard_deviation', StandardDeviationCalculator()),
            ('waveform_length', WaveformLengthCalculator()),
            ('histogram_distribution', BinHistogramDistributionCalculator(number_of_bins=10)),
            ('average_gait_cycle_length', AverageGaitCycleLengthCalculator()),
            ('discrete_fourier_coefficients', FourierCoefficientsMagnitudeCalculator(number_of_coefficients=40)),
            ('discrete_cosine_coefficients', DiscreteCosineCoefficientsMagnitude(number_of_coefficients=40))
        ])

    def get_configuration(self, pipeline):
        configuration = super().get_configuration(pipeline)

        feature_transformer_list = configuration['feature_extraction__transformer__transformer_list']
        feature_transformer_names = [name for name, value in feature_transformer_list]
        feature_steps = [f"feature_extraction__transformer__{name}" for name in feature_transformer_names]
        keys_to_remove = [key for key in configuration.keys() if key in feature_steps]

        for key in keys_to_remove:
            del configuration[key]

        del configuration["feature_extraction__transformer__transformer_list"]
        return configuration


def test_run_classification():
    data_accessor = DataAccessor(DataAccessor.DATA_HOANG_2015, sensors=["Linear Acceleration Sensor", "Rotation Matrix"])
    runner = Hoang2015Runner(data_accessor)
    runner.run()


def test_deterministic():
    random.seed(SEED)
    data_accessor = DataAccessor(DataAccessor.DATA_HOANG_2015, filter_parameters={"max_number_of_users": 4, "max_number_of_recordings_per_user": 3},
                                 sensors=["Linear Acceleration Sensor", "Rotation Matrix"])
    runner = Hoang2015Runner(data_accessor)
    run_deterministic_test(runner.run(), "instability")
