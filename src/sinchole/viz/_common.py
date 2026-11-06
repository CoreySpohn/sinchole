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
