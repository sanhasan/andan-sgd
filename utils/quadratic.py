import numpy as np


START_POINT = np.array([2.0, 2.0])
MINIMUM_POINT = np.zeros(2)


def quadratic_value(point):
    return 50 * point[0] ** 2 + point[1] ** 2


def quadratic_gradient(point):
    return np.array([100 * point[0], 2 * point[1]])
