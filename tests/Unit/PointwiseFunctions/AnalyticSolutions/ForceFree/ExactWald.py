# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def TildeE(x, t, B0):
    result = x * 0.0

    r_squared = np.sum(x * x)
    r0 = 2.0

    if (r_squared) < r0**2:
        result[0] = -2.0 * B0 * x[1] / r0**2
        result[1] = 2.0 * B0 * x[0] / r0**2
    else:
        result[0] = -2.0 * B0 * x[1] / r_squared
        result[1] = 2.0 * B0 * x[0] / r_squared

    return result


def TildeB(x, t, B0):
    result = x * 0.0
    result[2] = B0

    return result


def TildePsi(x, t, wave_speed):
    return x[0] * 0.0


def TildePhi(x, t, wave_speed):
    return x[0] * 0.0


def TildeQ(x, t, wave_speed):
    return x[0] * 0.0
