import pandas as pd

from src.core.base_classes import SafeTransformer


class FeatureSelector(SafeTransformer):
    ORIGINAL_SELECTED_FEATURES = ["iqr_linearAcceleration_x", "fourier-magnitude_linearAcceleration_x_4", "fourier-magnitude_linearAcceleration_x_5",
                                  "mean_linearAcceleration_y", "fourier-magnitude_linearAcceleration_y_1", "fourier-magnitude_linearAcceleration_y_2",
                                  "fourier-magnitude_linearAcceleration_y_3", "std_gyroscope_x", "wavelet-energy_gyroscope_x",
                                  "fourier-magnitude_gyroscope_x_0", "fourier-magnitude_gyroscope_x_1", "fourier-magnitude_gyroscope_x_2",
                                  "fourier-magnitude_gyroscope_x_6", "fourier-magnitude_gyroscope_x_7", "iqr_gyroscope_y",
                                  "wavelet-energy_gyroscope_y", "fourier-magnitude_gyroscope_y_0", "fourier-magnitude_gyroscope_y_3",
                                  "fourier-magnitude_gyroscope_y_4", "fourier-magnitude_gyroscope_y_6", "mean_gyroscope_z", "std_gyroscope_z",
                                  "iqr_gyroscope_z", "wavelet-energy_gyroscope_z", "fourier-magnitude_gyroscope_z_5",
                                  "fourier-magnitude_gyroscope_z_6", "fourier-magnitude_gyroscope_z_7", "dtw_upstairs", "dtw_walking", "dtw_jogging",
                                  "dtw_jumping", "correlation_linearAcceleration_y_z", "correlation_gyroscope_x_z", "mmDiff_linearAcceleration_y",
                                  "mmDiff_gyroscope_x", "mmDiff_gyroscope_z"]

    def __init__(self, selected_features=None, must_include_all_features=True, **kwargs):
        self.selected_features = selected_features
        self.must_include_all_features = must_include_all_features
        super().__init__(**kwargs)

    def _transform(self, data):
        selected_features = self.selected_features
        if not selected_features:
            selected_features = FeatureSelector.ORIGINAL_SELECTED_FEATURES

        if self.must_include_all_features:
            assert all([column in data.columns for column in selected_features])

        selected_features = list(set(selected_features).intersection(set(data.columns)))
        selected_features.sort()

        return data[selected_features]


def test_feature_selector():
    selected_features = ["f", "b", "c"]
    df = pd.DataFrame(columns=["a", "b", "c", "d", "e", "f"])
    feature_selector = FeatureSelector(selected_features=selected_features)
    assert list(feature_selector.transform(df).columns) == selected_features
