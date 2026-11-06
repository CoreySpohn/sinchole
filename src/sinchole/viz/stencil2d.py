"""The 32-by-32 tap stencil of one read, drawn over the node lattice."""

import jax.numpy as jnp
import numpy as np
from matplotlib.colors import to_rgba

from sinchole.kernel import stencil
from sinchole.viz import _require
from sinchole.viz._common import lattice_parts, query_xy, require_x64, single_ax


def plot_stencil_2d(
    query,
    lattice,
    *,
    quadrant=False,
    order="spatial",
    n_taps=None,
    fold=0.0,
    max_area=120.0,
    ax=None,
):
    """Draw the 1024 taps of a read, marker area proportional to the weight.

    Each tap sits at its node, with area ``max_area * |w_x w_y|`` (a weight of
    one, a read exactly on a node, fills ``max_area``) and the color of its
    sign (the ``positive`` and ``negative`` roles). The stored lattice is
    scenery behind it. Taps not yet counted are faint ghost dots, so the whole
    stencil is visible before it accumulates.

    Args:
        query: Finite ``(x, y)`` source position.
        lattice: Lattice or ``(origin_xy, spacing_xy, shape_yx)`` tuple.
        quadrant: The lattice stores only the positive quadrant. Taps at
            negative nodes are hollow (they exist only by symmetry) and mirrored
            nodes are drawn as dim dots. Requires a zero lattice origin.
        order: ``"spatial"`` counts taps row by row, ``"weight"`` by decreasing
            weight magnitude, so the first taps counted are the ones that matter.
        n_taps: Number of taps counted so far (0 to 1024); None counts all.
        fold: Quadrant only. 0 draws taps at their signed nodes, 1 draws taps
            at negative nodes on the stored node they read, 0 to 1 slides them.
        max_area: Marker area in points squared of a weight of one.
        ax: Single axes; None creates a figure. Limits stay fixed on updates.

    Returns:
        PlotResult whose ``scatter`` holds, in order, the stored nodes, the
        mirrored nodes (quadrant only), the ghost dots, the counted taps and
        the query marker. ``update(query=None, n_taps=None, fold=None)`` moves
        the existing artists; an argument left None keeps its current value.

    Raises:
        ValueError: Invalid lattice, query, order, counts, axes shape, or
            disabled x64.

    Example::

        from sinchole.viz import plot_stencil_2d
        result = plot_stencil_2d((1.0, 2.0), ((0, 0), (0.38, 0.38), (31, 31)),
                                 quadrant=True, order="weight", n_taps=100)
        result.update((0.2, 2.0), n_taps=1024, fold=1.0)
    """
    origin, spacing, shape = lattice_parts(lattice)
    if quadrant and np.any(origin != 0):
        raise ValueError("quadrant stencils require zero lattice origin")
    if order not in ("spatial", "weight"):
        raise ValueError("order must be 'spatial' or 'weight'")
    ep = _require.eyepiece()
    require_x64()
    nx, ny = shape[::-1].astype(int)
    positive, negative = ep.style.role("positive"), ep.style.role("negative")
    scenery = ep.style.neutral(0.35)
    state = {"query": query_xy(query), "n": n_taps, "fold": float(fold)}
    if not 0.0 <= state["fold"] <= 1.0 or (state["fold"] > 0 and not quadrant):
        raise ValueError("fold must lie in [0, 1] and needs a quadrant lattice")

    def tap_table():
        """Return positions, weights, validity and count rank of all 1024 taps."""
        taps, weight = [], []
        for axis in (0, 1):
            t, w = stencil(
                jnp.asarray([state["query"][axis]], dtype=jnp.float64),
                float(origin[axis]),
                float(spacing[axis]),
            )
            taps.append(np.asarray(t[0]))
            weight.append(np.asarray(w[0]))
        tx, ty = np.meshgrid(taps[0], taps[1])
        weights = (weight[1][:, None] * weight[0][None, :]).ravel()
        tx, ty = tx.ravel(), ty.ravel()
        rank = np.arange(1024)
        if order == "weight":
            rank[np.argsort(-np.abs(weights), kind="stable")] = np.arange(1024)
        reach_x, reach_y = (np.abs(tx), np.abs(ty)) if quadrant else (tx, ty)
        valid = (reach_x >= 0) & (reach_x < nx) & (reach_y >= 0) & (reach_y < ny)
        f = state["fold"]
        fx = tx + f * (np.abs(tx) - tx) if quadrant else tx
        fy = ty + f * (np.abs(ty) - ty) if quadrant else ty
        xy = np.column_stack((origin[0] + fx * spacing[0], origin[1] + fy * spacing[1]))
        return xy, weights, valid, rank

    ax = single_ax(ax)
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))
    stored = np.column_stack(
        (origin[0] + xx.ravel() * spacing[0], origin[1] + yy.ravel() * spacing[1])
    )
    points = [ax.scatter(*stored.T, s=3, color=scenery, zorder=1)]
    if quadrant:
        off_x, off_y = stored[:, 0] != 0, stored[:, 1] != 0
        mirrored = np.concatenate(
            (
                stored[off_x] * (-1, 1),
                stored[off_y] * (1, -1),
                stored[off_x & off_y] * (-1, -1),
            )
        )
        points.append(
            ax.scatter(*mirrored.T, s=2, color=ep.style.neutral(0.2), zorder=1)
        )
    xy = tap_table()[0]
    ghost = ax.scatter(*xy.T, s=2.5, color=ep.style.neutral(0.5), zorder=2)
    taps = ax.scatter([], [], zorder=3, linewidths=0.8)
    marker = ax.scatter([], [], marker="x", s=40, color=ep.style.role("text"), zorder=4)
    points += [ghost, taps, marker]

    def redraw():
        xy, weights, valid, rank = tap_table()
        limit = 1024 if state["n"] is None else int(state["n"])
        if not 0 <= limit <= 1024:
            raise ValueError("n_taps must lie between 0 and 1024")
        counted = rank < limit
        sign = np.where(weights >= 0, 0, 1)
        edge = np.array([to_rgba(positive), to_rgba(negative)])[sign]
        face = edge.copy()
        face[~valid, 3] = 0.0
        ghost.set_offsets(xy)
        taps.set_offsets(xy[counted])
        taps.set_sizes(max_area * np.abs(weights[counted]))
        taps.set_facecolors(face[counted])
        taps.set_edgecolors(edge[counted])
        marker.set_offsets(state["query"].reshape(1, 2))

    redraw()
    low = np.minimum(stored.min(0), -stored.max(0) if quadrant else stored.min(0))
    high = stored.max(0)
    pad = 4 * spacing
    ax.set_xlim(low[0] - pad[0], high[0] + pad[0])
    ax.set_ylim(low[1] - pad[1], high[1] + pad[1])
    ax.set_aspect("equal")
    ep.plot.label_lod(ax)

    def update(query=None, n_taps=None, fold=None):
        if query is not None:
            state["query"] = query_xy(query)
        if n_taps is not None:
            state["n"] = n_taps
        if fold is not None:
            if not 0.0 <= float(fold) <= 1.0 or (float(fold) > 0 and not quadrant):
                raise ValueError("fold must lie in [0, 1] and needs a quadrant lattice")
            state["fold"] = float(fold)
        redraw()

    return ep.plot.PlotResult(ax, {"scatter": points}, update)
