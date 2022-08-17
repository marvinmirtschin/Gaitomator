class InvalidConfigurationException(Exception):
    pass


class UnknownFilterException(Exception):
    pass


class IncorrectInputTypeException(Exception):
    def __init__(self, input_parameter, expected_type=None):
        error_message = "Input parameter is of wrong type ({})".format(type(input_parameter).__name__)
        if expected_type:
            error_message += ". The expected type was {}".format(expected_type.__name__)
        super().__init__(error_message)


class NoCyclesDetectedException(Exception):
    pass


class EmptyDataFrameException(Exception):
    pass


class CycleCleaningException(Exception):
    pass


class InterpolationException(Exception):
    pass


class MissingSensorDataException(Exception):
    pass


class RotationCalculationException(Exception):
    pass


class SegmentationException(Exception):
    pass


class NotFittedException(Exception):
    pass


class UnknownMethodException(Exception):
    pass
