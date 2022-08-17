import math
import warnings
from copy import copy

import numpy as np
import pandas as pd

from src.core.base_classes import SafeRowwiseTransformer
from src.core.error_handling.exceptions import MissingSensorDataException, RotationCalculationException


class AccelerationTransformer(SafeRowwiseTransformer):
    """
    Perform Coordinate Transformation from Hoang et al. 2015 with accerelation, gravity and magnetometer data instead of the rotation matrix.
    """

    def __init__(self, append_to_data_frame=False, **kwargs):
        self.append_to_data_frame = append_to_data_frame
        super().__init__(**kwargs)

    def _transform(self, data):
        try:
            # TODO: this only works if we assume the correct order of the row
            transformed_df = data.apply(lambda data_frame_row: transform_acceleration_vector_to_earth_coordinate_per_row(*data_frame_row), axis=1)
        except TypeError as e:
            raise MissingSensorDataException(e)

        # handle cases where rotation matrix was not successfully calculated
        transformed_df.interpolate(inplace=True)

        if not self.append_to_data_frame:
            return pd.DataFrame(transformed_df.values.tolist(), index=data.index, columns=['accelerometer_x', 'accelerometer_y', 'accelerometer_z'])

        new_columns = ["transformed_" + str(column) for column in ['accelerometer_x', 'accelerometer_y', 'accelerometer_z']]
        transformed_df.columns = new_columns
        return pd.concat([data, transformed_df])


def transform_acceleration_vector_to_earth_coordinate_per_row(acc_x, acc_y, acc_z, grav_x, grav_y, grav_z, mag_x, mag_y, mag_z):
    """
    Helper function to transform a data frame row into the desired input.
    """
    acceleration = np.array(acc_x, acc_y, acc_z)
    gravity = np.array(grav_x, grav_y, grav_z)
    geomagnetic = np.array(mag_x, mag_y, mag_z)
    return transform_acceleration_vector_to_earth_coordinate(acceleration, gravity, geomagnetic)


def transform_acceleration_vector_to_earth_coordinate(acceleration, gravity, geomagnetic, use_angles=True):
    """
    Transforms the given acceleration values from the mobile coordinate system to the Earth coordinate system by multiplying it with a rotation
    matrix obtained from the device's orientation.

    Parameters
    ----------
    acceleration : array-like
        Acceleration sensor values to be transformed.
    gravity : array-like
        Gravity sensor values used to calculated the device's orientation
    geomagnetic : array-like
        Geomagnetic field sensor values used to calculated the device's orientation
    use_angles : bool, Default=True
        If True, orientation will be calculated using euler angles

    Returns
    -------
    transformed_acceleration_values : array-like
        Transformed acceleration values.
    """
    try:
        acceleration -= gravity
        if use_angles:
            orientation = _calculate_orientation_in_euler_angles(gravity, geomagnetic, return_16_values=False)
            rotation_matrix = _calculate_paper_rotation_matrix(*orientation)
        else:
            rotation_matrix, inclination_matrix = _calculate_rotation_matrices(gravity, geomagnetic)
        return _transform_acceleration_vector(acceleration, rotation_matrix)
    except RotationCalculationException as e:
        warnings.warn(
            "Some orientations could not be calculated. These values will be set to 'np.nan' and may later be interpolated: {}".format(str(e)))
        return pd.Series([np.nan] * len(acceleration))


def _calculate_rotation_matrices(gravity, geomagnetic, return_16_values=False):
    """
    See https://developer.android.com/reference/android/hardware/SensorManager#getRotationMatrix.
    Inline comments taken from here:
    https://stackoverflow.com/questions/32372847/android-algorithms-for-sensormanager-getrotationmatrix-and-sensormanager-getori

    Computes the inclination matrix I as well as the rotation matrix R transforming a vector from the device coordinate system to the world's
    coordinate system which is defined as a direct orthonormal basis, where:

        - actual_data_frame is defined as the vector product Y.Z (It is tangential to the ground at the device's current location and roughly
        points East).
        - Y is tangential to the ground at the device's current location and points towards the magnetic North Pole.
        - Z points towards the sky and is perpendicular to the ground.
    By definition:
        - [0 0 g] = R * gravity (g = magnitude of gravity)
        - [0 m 0] = I * R * geomagnetic (m = magnitude of geomagnetic field)

    R is the identity matrix when the device is aligned with the world's coordinate system, that is, when the device's actual_data_frame axis
    points toward East,
    the Y axis points to the North Pole and the device is facing the sky.
    I is a rotation matrix transforming the geomagnetic vector into the same coordinate space as gravity (the world's coordinate space). I is a
    simple rotation around the actual_data_frame axis. The inclination angle in radians can be computed with {@link #getInclination}.

    Each matrix is returned either as a 3x3 or 4x4 row-major matrix depending on the length of the passed array.

    Note that the returned matrices always have this form:

        /  M[ 0]   M[ 1]   M[ 2]   0  \
        |  M[ 4]   M[ 5]   M[ 6]   0  |
        |  M[ 8]   M[ 9]   M[10]   0  |
        \      0       0       0   1  /

    If the array length is 9:

        /  M[ 0]   M[ 1]   M[ 2]  \
        |  M[ 3]   M[ 4]   M[ 5]  |
        \  M[ 6]   M[ 7]   M[ 8]  /

    The inverse of each matrix can be computed easily by taking its transpose.
    The matrices returned by this function are meaningful only when the device is not free-falling and it is not close to the magnetic north. If
    the device is accelerating, or placed into a strong magnetic field, the returned matrices may be inaccurate.

    Parameters
    ----------
    gravity : nd.Array
        Should be the values obtained from an gravity sensor or low pass filter of an accelerometer.
    geomagnetic : nd.Array
        Should be the values from an magnetometer.
    return_16_values : bool, default=False
        If True, return 4 x 4 matrices instead of 3 x 3.

    Returns
    -------
    rotation_matrix : array-like

    inclination_matrix : array_like
    """

    # We assume that gravity points toward the center of the Earth and magnet to the north Pole. But in real cases these vectors are
    # non-perpendicular, that's why we firstly calculate vector H that is orthogonal to E and A and belong to tangential plane. H is a
    # cross-product (E x A) and is orthogonal to E and A.
    H = np.cross(geomagnetic, gravity)
    H_norm = np.linalg.norm(H)

    # check conditions
    gravity_norm_square = pow(np.linalg.norm(gravity), 2)
    g = 9.81
    free_fall_gravity_squared = 0.01 * g * g
    if gravity_norm_square < free_fall_gravity_squared:
        # gravity less than 10 % of normal value
        raise RotationCalculationException("Gravity is too low")

    if H_norm < 0.1:
        # device is close to free fall ( or in space?), or close to magnetic north pole.Typical values are > 100
        raise RotationCalculationException("Device may have been in free fall (or maybe in space? :o).")

    # normalize acceleration and H vector (because these vectors will compose a basis of ENU coordinate system)
    H_inverted_norm = 1.0 / H_norm
    H *= H_inverted_norm

    gravity_norm = np.linalg.norm(gravity)
    inverted_gravity_norm = 1.0 / gravity_norm
    gravity *= inverted_gravity_norm

    # Find last basis vector (M) as cross-product of H and A:
    M = np.cross(gravity, H)

    # Coordinates of the arbitrary vector (a) in body frame expresses through NED coordinates as a = Ra' R - Transformation matrix matrix,
    # whose columns are the coordinates of the new basis vectors in the old basis
    #
    # But coordinates in NED frame are calculated as a' = T^(-1) * a. For orthogonal transformation matrix inverse is equal to transposed matrix.
    # Thus we have:
    if return_16_values:
        # 4 x 4 matrix
        rotation_matrix = np.concatenate((H, [0], M, [0], gravity, [0, 0, 0, 0, 1]), axis=0)
    else:
        # 3 x 3 matrix
        rotation_matrix = np.concatenate((H, M, gravity), axis=0)

    # compute the inclination matrix by projecting the geomagnetic vector onto the Z (gravity) and actual_data_frame (horizontal component of
    # geomagnetic vector)
    # axes.
    magnetic_norm = np.linalg.norm(geomagnetic)
    magnetic_inverted_norm = 1.0 / magnetic_norm
    c = sum(geomagnetic * M) * magnetic_inverted_norm
    s = sum(geomagnetic * gravity) * magnetic_inverted_norm
    if return_16_values:
        inclination_matrix = [1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1]
    else:
        inclination_matrix = [1, 0, 0, 0, c, s, 0, -s, c]

    return rotation_matrix, inclination_matrix


def _calculate_orientation_in_euler_angles(gravity, geomagnetic, return_16_values=False):
    """
    Calculate the device orientation for the given sensor readings as euler angles around the device axes in the world coordinate base.

    Parameters
    ----------
    gravity
    geomagnetic
    return_16_values

    Returns
    -------
    azimuth : float
        Rotation around the -Z axis, i.e. the opposite direction of Z axis.
    pitch : float
        Rotation around the -actual_data_frame axis, i.e the opposite direction of actual_data_frame axis.
    roll : float
        Rotation around the Y axis.
    """
    rotation_matrix, inclination_matrix = _calculate_rotation_matrices(gravity, geomagnetic, return_16_values)

    # Once we have rotation matrix we can convert it to Euler angles representation. Formulas of conversion depend on convention that we use.
    # These formulas are correct for Tiat Bryan angles with convention Y-actual_data_frame-Z.
    azimuth = math.atan2(rotation_matrix[1], rotation_matrix[4])
    pitch = math.asin(-rotation_matrix[7])
    roll = math.atan2(-rotation_matrix[6], rotation_matrix[8])
    return azimuth, pitch, roll


def _calculate_paper_rotation_matrix(azimuth, pitch, roll):
    """
    Taken from the paper 'On the Instability of Sensor Orientation in Gait Verification on Mobile Phone'. Used to transform an acceleration vector
    in the mobile coordinate system to the Earth coordinate system.

    Parameters
    ----------
    azimuth : float
        Rotation around the -Z axis, i.e. the opposite direction of Z axis.
    pitch : float
        Rotation around the -actual_data_frame axis, i.e the opposite direction of actual_data_frame axis.
    roll : float
        Rotation around the Y axis.

    Returns
    -------
    rotation_matrix : array_like
        Rotation matrix to transform an acceleration vector from the mobile coordinate system to the Earth coordinate system.
    """
    rotation_matrix = [(math.cos(azimuth) * math.cos(roll) - math.sin(azimuth) * math.sin(pitch) * math.sin(roll)),
                       (math.sin(azimuth) * math.cos(roll)),
                       (math.cos(azimuth) * math.sin(roll) + math.sin(azimuth) * math.sin(pitch) * math.cos(roll)),
                       (-math.sin(azimuth) * math.cos(roll) - math.cos(azimuth) * math.sin(pitch) * math.sin(roll)),
                       (math.cos(azimuth) * math.cos(roll)),
                       (-math.sin(roll) * math.sin(roll) + math.cos(azimuth) * math.sin(pitch) * math.cos(roll)),
                       (-math.cos(pitch) * math.sin(roll)),
                       (-math.sin(pitch)),
                       (math.cos(pitch) * math.cos(roll))]
    return rotation_matrix


def transform_acceleration_vector_row(acceleration_x, acceleration_y, acceleration_z, rotation_0, rotation_1, rotation_2, rotation_3, rotation_4,
                                      rotation_5, rotation_6, rotation_7, rotation_8):
    return _transform_acceleration_vector(np.array([acceleration_x, acceleration_y, acceleration_z]),
                                          np.array([rotation_0, rotation_1, rotation_2, rotation_3, rotation_4, rotation_5, rotation_6, rotation_7,
                                                    rotation_8]))


def _transform_acceleration_vector(acceleration, rotation_matrix):
    return np.matmul(acceleration.T, rotation_matrix.reshape(3, 3))


def test_orientation_calculation():
    gravity = np.array([3.17, 14, 9.81])
    magnetometer = np.array([26.1, 33.2, 13.9])
    result1 = _calculate_orientation_in_euler_angles(copy(gravity), copy(magnetometer))
    result2 = _calculate_orientation_in_euler_angles(copy(gravity), copy(magnetometer), True)

    # values calculated in android with SensorManager.getRotationMatrix() for the same input
    expected1 = [-1.4462736, -0.9360627, -0.31254837]
    expected2 = [-0.57223904, -0.0, 1.1433687]

    for index in range(len(result1)):
        assert abs(abs(result1[index]) - abs(expected1[index])) < 0.01
        assert abs(abs(result2[index]) - abs(expected2[index])) < 0.01
