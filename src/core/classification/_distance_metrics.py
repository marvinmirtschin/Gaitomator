import numpy as np


def calculate_scaled_euclidean_distance_for_cycles(reference_cycle, scaling_cycle, test_cycle):
    """

    Parameters
    ----------
    reference_cycle : np.array
        Template cycle, in this approach it is usually the mean of all (cleaned) reference cycles
    scaling_cycle : np.array
        Standard deviation for each point from the references cycles
    test_cycle : np.array
        Cycle to calculate the distance to the reference from

    Returns
    -------
        Distance from test_cycle to reference_cycle with scaling being performed

    """
    cycle_difference = test_cycle - reference_cycle
    cycle_difference = np.square(cycle_difference)
    scaled_difference_cycle = cycle_difference / scaling_cycle
    result = np.sqrt(np.sum(scaled_difference_cycle, axis=1))
    return result


def calculate_scaled_manhattan_distance_for_cycles(reference_cycle, scaling_cycle, test_cycle):
    """

    Parameters
    ----------
    reference_cycle : np.array
        Template cycle, in this approach it is usually the mean of all (cleaned) reference cycles
    scaling_cycle : np.array
        Standard deviation for each point from the references cycles
    test_cycle : np.array
        Cycle to calculate the distance to the reference from

    Returns
    -------
        Distance from test_cycle to reference_cycle with scaling being performed

    """
    cycle_difference = test_cycle - reference_cycle
    cycle_difference = np.abs(cycle_difference)
    scaled_difference_cycle = cycle_difference / scaling_cycle
    result = np.sum(scaled_difference_cycle, axis=1)
    return result
