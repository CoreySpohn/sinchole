"""Shared axes and precision validation at the plotting boundary."""

import jax
import matplotlib.pyplot as plt
import numpy as np


def single_ax(ax):
    """Resolve one axes, rejecting arrays with an explicit shape error."""
    if ax is None:
        return plt.subplots(layout="constrained")[1]
    if not isinstance(ax, plt.Axes):
        shape = np.asarray(ax, dtype=object).shape
        raise ValueError(f"expected a single ax, got axes shape {shape}")
    return ax


def require_x64():
    """Refuse precision-sensitive kernel arithmetic without changing JAX."""
    if not jax.config.x64_enabled:
        raise ValueError("Enable jax_enable_x64 before plotting sinchole stencils")


def query_xy(query):
    """Convert one finite coordinate pair to a host array."""
    point = np.asarray(query, dtype=float)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError("query must be a finite (x, y) pair")
    return point


def lattice_parts(lattice):
    """Return validated ``(origin, spacing, shape_yx)`` host arrays of a lattice.

    Accepts a Lattice or an ``(origin_xy, spacing_xy, shape_yx)`` tuple.
    """
    if hasattr(lattice, "origin_xy_lod"):
        origin, spacing, shape = (
            lattice.origin_xy_lod,
            lattice.spacing_xy_lod,
            lattice.shape_yx,
        )
    else:
        origin, spacing, shape = lattice
    origin, spacing = np.asarray(origin, dtype=float), np.asarray(spacing, dtype=float)
    shape = np.asarray(shape)
    if (
        origin.shape != (2,)
        or spacing.shape != (2,)
        or shape.shape != (2,)
        or not np.all(np.isfinite(origin))
        or not np.all(np.isfinite(spacing))
        or np.any(spacing <= 0)
        or np.any(shape < 1)
        or np.any(shape != np.floor(shape))
    ):
        raise ValueError(
            "lattice needs finite origin, positive spacing, and integer shape_yx"
        )
    return origin, spacing, shape
