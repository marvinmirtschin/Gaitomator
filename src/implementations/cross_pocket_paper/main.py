import random

from sklearn.pipeline import Pipeline

from src.add_ins.preprocessing import AccelerationUnitTransformer, WeightedMovingAverageFilter
from src.add_ins.processing.features import MagnitudeCalculator
from src.add_ins.segmentation.bours_2018 import CycleDetectionTransformer as CrossPocketCycleDetectionTransformer
from src.add_ins.segmentation.hoang_2015 import CycleDetectionTransformer as HoangCycleDetectionTransformer
from src.core import constants
from src.core.base_classes import NestedDataFrameTransformer
from src.core.data_access import DataAccessor
from src.core.runner import ImplementationRunner
from src.core.utility.cycle_data import CycleInterpolator
from src.core.utility.data_frame_transformer import DataFrameTransposer, expand_data_frame, shrink_data_frame
from src.core.utility.sensor_fusion import SensorSynchronizer
from src.core.utility.testing_helper import run_deterministic_test
from src.core.visualization.pipeline_visualizer import CycleDataVisualizer
from src.implementations.cross_pocket_paper.hot_spots import run_classification

# Note: Related Papers:
#  - Gait Identification Using Accelerometer on Mobile Phone
#  - Adaptive Cross-Device Gait Recognition Using a Mobile Accelerometer (contains Feature Formulas)

SEED = constants.RANDOM_SEED


class Bours2018Runner(ImplementationRunner):
    # Implementation is able to run with these data sets
    MATCHING_DATA_SETS = DataAccessor.DATA_SETS

    # Here you can add parameter changes for the pipeline which need to be done for the data set to work
    DATA_SET_ADJUSTMENTS = {
        DataAccessor.DATA_WISDM     : dict(),
        DataAccessor.DATA_OU        : dict(),
        DataAccessor.DATA_UCIHAR    : dict(),
        DataAccessor.DATA_HOANG_2015: dict()
    }

    def print_classification_report(self, transformed_data_frame):
        run_classification(transformed_data_frame)

    def read_data_frame(self, data_file):
        transformed_data_frame = super().read_data_frame(data_file)
        return shrink_data_frame(transformed_data_frame)

    def save_data_frame(self, transformed_data_frame, data_file):
        transformed_data_frame = expand_data_frame(transformed_data_frame)
        super().save_data_frame(transformed_data_frame, data_file)

    @staticmethod
    def get_paper_name():
        return "bours_2018"

    def get_pipeline(self, recording_frequency, cycle_detection="original", visualize_cycles=False, frequency=100, **kwargs):
        if cycle_detection == "instability":
            cycle_transformer = NestedDataFrameTransformer(
                HoangCycleDetectionTransformer(cut_at_dimension="{}_{}".format(MagnitudeCalculator.KEY, "accelerometer")))
        else:
            cycle_transformer = NestedDataFrameTransformer(
                CrossPocketCycleDetectionTransformer(cut_at_dimension="{}_{}".format(MagnitudeCalculator.KEY, "accelerometer"),
                                                     frequency=frequency,
                                                     neighborhood_factor=0.3))  # Note that this value in OG is 0.2

        pipeline = Pipeline([
            ('time_equalization', NestedDataFrameTransformer(SensorSynchronizer(frequency=frequency, old_frequency=recording_frequency))),
            ('magnitude', NestedDataFrameTransformer(MagnitudeCalculator(sensor_names="accelerometer"))),
            ('filtering', NestedDataFrameTransformer(WeightedMovingAverageFilter())),
            ('cycle_detection', cycle_transformer),
            ('cycle_interpolation', NestedDataFrameTransformer(CycleInterpolator(interpolation_method="linear", cycle_length=100))),
            ('data_frame_transposer', NestedDataFrameTransformer(DataFrameTransposer())),
        ])

        if visualize_cycles:
            pipeline.steps.insert(4, ('cycle_visualizer', CycleDataVisualizer(plot_concatenated_cycles=False)))

        if self.data_accessor.source == DataAccessor.DATA_OU or self.data_accessor.source == DataAccessor.DATA_UCIHAR:
            # data is given in G instead of m/s^2
            pipeline.steps.insert(0, ('acceleration_unit_transformer', AccelerationUnitTransformer()))
        return pipeline


def test_for_all_matching_data_sets():
    matching_data_sets = Bours2018Runner.MATCHING_DATA_SETS
    for data_set in matching_data_sets:
        data_accessor = DataAccessor(data_set, sensors=["accelerometer"], filter_parameters={"activities": ["Walking"]})
        runner = Bours2018Runner(data_accessor)
        print(f"\nRunning {runner.get_paper_name()} with data set {data_set}")
        runner.run()


def test_deterministic():
    random.seed(SEED)
    data_accessor = DataAccessor(DataAccessor.DATA_HOANG_2015, filter_parameters={"max_number_of_users": 4, "max_number_of_recordings_per_user": 3},
                                 sensors=["accelerometer"])
    runner = Bours2018Runner(data_accessor)

    result = runner.process_data()
    result.columns = [str(column) for column in result.columns]
    run_deterministic_test(result, runner.get_paper_name())
