"""Extended-scene rendering from a fixed set of prefiltered coefficient images."""

import equinox as eqx
import jax
import jax.numpy as jnp

from sinchole._types import F64, I32, check_array, check_static
from sinchole.domain import coverage
from sinchole.evidence import BoundKind, MemoryLimit, Reason, Status
from sinchole.kernel import flush_subnormal
from sinchole.reader import ImageRead, PrefilteredReader, _require_precision


class SceneRenderer(eqx.Module):
    """Render many sources at once as one contraction of coefficient images.

    A prefiltered read is linear in the coefficient images, so the image of a
    scene, the flux-weighted sum of the reads of its sources, can be regrouped
    by lattice node: every source spreads its flux onto its taps with the
    read's own weights, and the folded node amplitudes multiply the coefficient
    images once. The result equals the sum of the individual reads up to
    floating-point rounding, at the cost of one pass over the touched nodes.

    The touched nodes are fixed at construction, so every shape is static under
    jit. ``table`` holds their coefficient images [nodes, pixels] in the storage
    dtype with mirror flips already applied, and ``lookup`` maps a signed node
    to its row (the node count marks an absent node).

    Rendering refuses the whole image when any contributing source cannot be
    read. A source contributes when its flux, or its raster value, is nonzero;
    zero-flux entries are padding and may hold any position.
    """

    reader: PrefilteredReader
    table: jax.Array
    lookup: I32
    nodes: I32
    product_index: int = eqx.field(static=True)
    margin_nodes: int = eqx.field(static=True)
    _corner_xy: tuple[int, int] = eqx.field(static=True)

    def __init__(self, reader, product_index, source_xy_lod, *, margin_nodes=1):
        """Gather the coefficient images that reads near the given positions touch.

        Args:
            reader: A PrefilteredReader; its policy, domain and coverage apply.
            product_index: Python int, the one product this renderer serves.
            source_xy_lod: Host float64 positions [Q, 2] whose stencils define the
                touched nodes: the sources themselves, or the centers of the
                raster pixels that can be lit. Every position must be readable.
            margin_nodes: Lattice nodes added around the touched set in every
                direction, so positions may move by that many node spacings (in
                a fit, for example) before a render refuses.
        """
        _require_precision()
        if not isinstance(reader, PrefilteredReader):
            raise TypeError("SceneRenderer needs a PrefilteredReader")
        count = reader.grid.intensity.shape[0]
        if type(product_index) is not int or not 0 <= product_index < count:
            raise ValueError("product_index must be an available Python int")
        if type(margin_nodes) is not int or margin_nodes < 0:
            raise ValueError("margin_nodes must be a nonnegative integer")
        positions = jnp.asarray(source_xy_lod)
        if positions.ndim != 2 or positions.shape[0] == 0:
            raise ValueError("Scene positions require a nonempty [source, 2] array")
        check_array(positions, (positions.shape[0], 2), (jnp.float64,))
        self.reader, self.product_index = reader, product_index
        self.margin_nodes = margin_nodes
        product = jnp.int32(product_index)
        readable = jax.vmap(lambda q: reader._validate(q, product)[1])(positions)
        if not bool(jnp.all(readable)):
            raise ValueError("Scene positions must be finite, permitted and covered")
        taps = jax.vmap(lambda q: reader.stencil(flush_subnormal(q))[0])(positions)
        # Nodes that some covered read can use, as signed (low, high) per axis.
        size = reader.grid.source.shape_yx[::-1]
        halo = reader.profile.halo
        reflected = reader.grid.reflection.kind != "none"
        usable = [(halo - (n - 1) if reflected else halo, n - 1 - halo) for n in size]
        low = [
            max(int(jnp.min(taps[:, axis])) - margin_nodes, usable[axis][0])
            for axis in range(2)
        ]
        high = [
            min(int(jnp.max(taps[:, axis])) + margin_nodes, usable[axis][1])
            for axis in range(2)
        ]
        span_x, span_y = (high[axis] - low[axis] + 1 for axis in range(2))
        touched = jnp.zeros((span_y, span_x), dtype=jnp.bool_)
        touched = touched.at[
            (taps[:, 1, :, None] - low[1]), (taps[:, 0, None, :] - low[0])
        ].set(True)
        for _ in range(margin_nodes):
            padded = jnp.pad(touched, 1)
            touched = jnp.zeros_like(touched)
            for shift_y in range(3):
                for shift_x in range(3):
                    touched = (
                        touched
                        | padded[shift_y : shift_y + span_y, shift_x : shift_x + span_x]
                    )
        self.nodes = jnp.argwhere(touched).astype(jnp.int32)
        rows = self.nodes.shape[0]
        self.lookup = (
            jnp.full((span_y, span_x), rows, dtype=jnp.int32)
            .at[self.nodes[:, 0], self.nodes[:, 1]]
            .set(jnp.arange(rows, dtype=jnp.int32))
        )
        self._corner_xy = (low[0], low[1])
        coefficients = reader.coefficients
        pixels = coefficients.shape[3] * coefficients.shape[4]
        estimate = reader.memory_estimate(1)
        table_bytes = rows * pixels * coefficients.dtype.itemsize
        if (
            estimate["resident"] + table_bytes > reader.memory.max_resident_bytes
            or estimate["total"] + table_bytes > reader.memory.max_total_bytes
        ):
            raise MemoryLimit(
                f"scene table of {table_bytes} bytes exceeds the resident budget"
            )
        node_y = self.nodes[:, 0] + low[1]
        node_x = self.nodes[:, 1] + low[0]
        images = coefficients[product_index][jnp.abs(node_y), jnp.abs(node_x)]
        if reflected:
            images = jnp.where((node_x < 0)[:, None, None], images[..., ::-1], images)
            images = jnp.where(
                (node_y < 0)[:, None, None], images[..., ::-1, :], images
            )
        self.table = images.reshape(rows, pixels)

    def __check_init__(self):
        """Validate the declared structure before numerical use."""
        check_static(self)
        rows = self.nodes.shape[0]
        check_array(self.nodes, (rows, 2), (jnp.int32,))
        check_array(self.lookup, self.lookup.shape, (jnp.int32,))
        if self.table.ndim != 2 or self.table.shape[0] != rows:
            raise ValueError("Scene table requires one row per touched node")

    def _scratch(self, working_bytes):
        # A table stored below binary64 is widened for the contraction.
        if self.table.dtype != jnp.float64:
            working_bytes += self.table.size * 8
        limit = self.reader.memory.max_scratch_bytes
        if working_bytes > limit:
            raise MemoryLimit(f"scene scratch {working_bytes} bytes exceeds budget")

    def _axis_ok(self, values, axis):
        """Sanitize coordinates of one axis and flag the finite, covered ones."""
        finite = jnp.isfinite(values)
        clean = flush_subnormal(jnp.where(finite, values, 0.0))
        safe = self.reader._safe_query
        reflected = self.reader.grid.reflection.kind != "none"
        covered = jax.vmap(
            lambda v: coverage(
                safe.at[axis].set(v),
                self.reader.grid.source,
                reflected,
                self.reader.profile,
            )
        )(clean)
        return clean, finite, covered

    def _finish(self, amplitudes, accepted, sampling_ok, failures):
        """Contract node amplitudes with the table and attach read diagnostics."""
        reader = self.reader
        # The contraction accumulates in binary64 whatever the storage dtype, as a
        # single read does. Full precision keeps accelerator backends from a
        # reduced-precision product.
        image = jnp.matmul(
            amplitudes,
            self.table.astype(jnp.float64),
            precision=jax.lax.Precision.HIGHEST,
        )
        image = image.reshape(reader.products.pixels.shape_yx)
        reason = jnp.asarray(Reason.BOUND_UNKNOWN, dtype=jnp.uint32)
        for failed, bit in failures:
            reason = reason | jnp.where(failed, jnp.uint32(bit), jnp.uint32(0))
        product = jnp.int32(self.product_index)
        ledger = reader.errors.evaluate(
            reader._safe_query, product, reader._gain_xy, grid=reader.grid
        )
        return ImageRead(
            jnp.where(accepted, image, jnp.nan),
            accepted,
            sampling_ok,
            jnp.asarray(
                jnp.where(accepted, Status.SAMPLING_ADMITTED, Status.REFUSED),
                dtype=jnp.int32,
            ),
            reason,
            jnp.asarray(False),
            jnp.asarray(BoundKind.UNKNOWN, dtype=jnp.int32),
            jnp.asarray(jnp.inf, dtype=jnp.float64),
            jnp.asarray(jnp.nan, dtype=jnp.float64),
            ledger,
        )

    def render_points(self, source_xy_lod: F64, flux: F64) -> ImageRead:
        """Render point sources [Q, 2] with fluxes [Q] as one image.

        The image is the sum over sources of flux times that source's read, in
        delivered units per unit flux. Node amplitudes are folded and contracted
        with the coefficient images in binary64, whatever the storage dtype, so
        the image equals the summed reads to binary64 rounding. The error bound
        is unknown, as for a single read, and a point-read empirical check does
        not transfer, so ``empirical_max_abs``
        is NaN.

        Any contributing source that is non-finite, outside the domain, not
        covered or outside the touched nodes refuses the whole image. Reads
        jump by the bump's edge value where a stencil switches cell, so
        position gradients are piecewise.
        """
        _require_precision()
        queries, flux = jnp.asarray(source_xy_lod), jnp.asarray(flux)
        if queries.ndim != 2:
            raise ValueError("Scene coordinates require [source, 2]")
        count = queries.shape[0]
        check_array(queries, (count, 2), (jnp.float64,))
        check_array(flux, (count,), (jnp.float64,))
        reader, taps = self.reader, self.reader.profile.taps
        rows = self.nodes.shape[0]
        self._scratch(count * taps * taps * 32 + rows * 16 + self.table.shape[1] * 16)
        finite = jnp.all(jnp.isfinite(queries), axis=1) & jnp.isfinite(flux)
        clean = flush_subnormal(jnp.where(jnp.isfinite(queries), queries, 0.0))
        member = jax.vmap(reader.domain.contains)(clean)
        reflected = reader.grid.reflection.kind != "none"
        covered = jax.vmap(
            lambda q: coverage(q, reader.grid.source, reflected, reader.profile)
        )(clean)
        readable = finite & member & covered
        safe = jnp.where(readable[:, None], clean, reader._safe_query)
        node, weight = jax.vmap(reader.stencil)(safe)
        index_x = node[:, 0, None, :] - self._corner_xy[0]
        index_y = node[:, 1, :, None] - self._corner_xy[1]
        span_y, span_x = self.lookup.shape
        inside = (
            (index_x >= 0) & (index_x < span_x) & (index_y >= 0) & (index_y < span_y)
        )
        row = self.lookup[
            jnp.clip(index_y, 0, span_y - 1), jnp.clip(index_x, 0, span_x - 1)
        ]
        row = jnp.where(inside, row, rows)
        stored = jnp.all(row < rows, axis=(1, 2))
        contributing = flux != 0
        usable = readable & stored
        sampling_ok = jnp.all(usable | ~contributing)
        accepted = sampling_ok & (reader.policy.mode == "sampling")
        amount = jnp.where(usable & contributing, flux, 0.0)
        spread = amount[:, None, None] * weight[:, 1, :, None] * weight[:, 0, None, :]
        amplitudes = jnp.zeros(rows + 1).at[row.reshape(-1)].add(spread.reshape(-1))
        return self._finish(
            amplitudes[:rows],
            accepted,
            sampling_ok,
            (
                (jnp.any(contributing & ~finite), Reason.NONFINITE_QUERY),
                (jnp.any(contributing & finite & ~member), Reason.OUTSIDE_DOMAIN),
                (
                    # ``stored`` is meaningful only for a readable source.
                    jnp.any(contributing & finite & (~covered | (readable & ~stored))),
                    Reason.INCOMPLETE_STENCIL,
                ),
            ),
        )

    def render_raster(self, scene: F64, x_lod: F64, y_lod: F64) -> ImageRead:
        """Render a raster scene [H, W] whose pixel centers are x_lod [W], y_lod [H].

        Every raster pixel is a point source at its center with the pixel's
        value as flux, so the image equals ``render_points`` on the flattened
        raster. The fold is separable: with spreading matrices W_y and W_x that
        hold each center's tap weights, the node amplitudes are W_y^T S W_x.

        Any nonzero pixel that is non-finite, outside the domain, not covered
        or outside the touched nodes refuses the whole image.
        """
        _require_precision()
        scene, x, y = jnp.asarray(scene), jnp.asarray(x_lod), jnp.asarray(y_lod)
        if scene.ndim != 2:
            raise ValueError("Raster scenes require two axes")
        height, width = scene.shape
        check_array(scene, (height, width), (jnp.float64,))
        check_array(x, (width,), (jnp.float64,))
        check_array(y, (height,), (jnp.float64,))
        reader, rows = self.reader, self.nodes.shape[0]
        span_y, span_x = self.lookup.shape
        self._scratch(
            16 * height * width
            + 16 * (height * (span_y + 1) + width * (span_x + 1))
            + 16 * (span_y + 1) * (span_x + 1)
            + self.table.shape[1] * 16
        )
        clean_x, finite_x, covered_x = self._axis_ok(x, 0)
        clean_y, finite_y, covered_y = self._axis_ok(y, 1)
        member = jax.vmap(
            lambda v: jax.vmap(lambda u: reader.domain.contains(jnp.stack([u, v])))(
                clean_x
            )
        )(clean_y)
        finite = jnp.isfinite(scene) & finite_y[:, None] & finite_x[None, :]
        covered = covered_y[:, None] & covered_x[None, :]
        lit = scene != 0
        usable = finite & member & covered
        amount = jnp.where(usable & lit, scene, 0.0)
        safe = reader._safe_query
        spreading = []
        for axis, (clean, ok, span) in enumerate(
            (
                (clean_x, finite_x & covered_x, span_x),
                (clean_y, finite_y & covered_y, span_y),
            )
        ):
            values = jnp.where(ok, clean, safe[axis])
            node, weight = jax.vmap(
                lambda v, a=axis: reader.stencil(safe.at[a].set(v))
            )(values)
            index = node[:, axis, :] - self._corner_xy[axis]
            column = jnp.where((index >= 0) & (index < span), index, span)
            matrix = jnp.zeros((values.shape[0], span + 1))
            matrix = matrix.at[jnp.arange(values.shape[0])[:, None], column].add(
                weight[:, axis, :]
            )
            spreading.append(matrix)
        across, down = spreading
        folded = down.T @ amount @ across
        # Flux that lands on a node outside the table marks an unreadable pixel.
        mass = down.T @ jnp.abs(amount) @ across
        present = jnp.pad(self.lookup < rows, ((0, 1), (0, 1)))
        stored = ~jnp.any((mass > 0) & ~present)
        sampling_ok = jnp.all(usable | ~lit) & stored
        accepted = sampling_ok & (reader.policy.mode == "sampling")
        amplitudes = folded[self.nodes[:, 0], self.nodes[:, 1]]
        return self._finish(
            amplitudes,
            accepted,
            sampling_ok,
            (
                (jnp.any(lit & ~finite), Reason.NONFINITE_QUERY),
                (jnp.any(lit & finite & ~member), Reason.OUTSIDE_DOMAIN),
                (jnp.any(lit & finite & ~covered) | ~stored, Reason.INCOMPLETE_STENCIL),
            ),
        )
