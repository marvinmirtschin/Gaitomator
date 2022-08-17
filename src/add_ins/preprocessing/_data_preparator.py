from src.core.base_classes import SafeTransformer


class AccelerationUnitTransformer(SafeTransformer):
    G = 9.81

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _transform(self, data):
        _regex = "cceler"
        reference_sensor_data = data.filter(regex=_regex, axis=1)
        transformed_df = reference_sensor_data.apply(lambda x: x * AccelerationUnitTransformer.G)
        data[transformed_df.columns] = transformed_df
        return data
