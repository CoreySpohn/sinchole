"""Independent finite-sum oracle using only NumPy and rational floor truth."""

import math
from fractions import Fraction

import numpy as np


def reference_read(data, query, product=0, shift_y=0):
    """Return the raw finite sum and absolute weighted sum per detector pixel."""
    query = np.asarray(query, dtype=float).copy()
    query[np.abs(query) < np.finfo(float).tiny] = 0.0
    origin = np.asarray(data["declared_lattice"].origin_xy_lod)
    spacing = np.asarray(data["declared_lattice"].spacing_xy_lod)
    taps, weights = [], []
    for axis in range(2):
        difference = float(query[axis] - origin[axis])
        floor = math.floor(Fraction(difference) / Fraction(float(spacing[axis])))
        shift = shift_y if axis == 1 else 0
        indices = np.arange(floor - 15 + shift, floor + 17 + shift)
        phase = difference / spacing[axis] - indices
        active = np.abs(phase) < 16
        weight = np.zeros(32)
        u = phase[active]
        weight[active] = (
            np.sinc(u) * np.i0(12 * np.sqrt(1 - (u / 16) ** 2)) / np.i0(12.0)
        )
        taps.append(indices)
        weights.append(weight)
    shape = data["products"].pixels.shape_yx
    nodes = np.asarray(data["intensity_nodes"])
    cube = nodes.reshape(nodes.shape[0], *data["declared_lattice"].shape_yx, *shape)
    reflected = data["reflection"].kind == "two_axis_pixel_flip_v1"
    result, absolute = np.zeros(shape), np.zeros(shape)
    for j, iy in enumerate(taps[1]):
        row = np.zeros(shape)
        for i, ix in enumerate(taps[0]):
            sample = cube[
                product, abs(iy) if reflected else iy, abs(ix) if reflected else ix
            ].astype(float)
            if reflected and ix < 0:
                sample = sample[:, ::-1]
            if reflected and iy < 0:
                sample = sample[::-1, :]
            row += weights[0][i] * sample
            absolute += np.abs(weights[1][j] * weights[0][i] * sample)
        result += weights[1][j] * row
    return result, absolute
