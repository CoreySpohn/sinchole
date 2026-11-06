"""Signed permission domains and independent full-stencil coverage."""

import math
import sys

import equinox as eqx
import jax
import jax.numpy as jnp

from sinchole._types import F64, check_array, check_static
from sinchole.kernel import exact_quotient_floor_v1, flush_subnormal


class Domain(eqx.Module):
    """A box or origin-centered annulus, without tolerance-expanded boundaries."""

    kind: str = eqx.field(static=True)
    boundary: str = eqx.field(static=True)
    limits_xy_lod: F64 = eqx.field(converter=jnp.asarray)
    radii_lod: F64 = eqx.field(converter=jnp.asarray)

    def __check_init__(self):
        """Validate the declared structure before numerical use."""
        check_static(self)
        check_array(self.limits_xy_lod, (2, 2), (jnp.float64,))
        check_array(self.radii_lod, (2,), (jnp.float64,))
        if self.kind not in ("box", "annulus"):
            raise ValueError("Domain kind must be box or annulus")
        if self.boundary not in ("closed", "lower_closed_upper_open"):
            raise ValueError("Unsupported domain boundary")
        values = [*self.limits_xy_lod.tolist(), self.radii_lod.tolist()]
        if any(not normal_or_zero(v) for row in values for v in row):
            raise ValueError("Domain limits must be finite, normal or zero")
        if self.kind == "box":
            ordered = self.limits_xy_lod[0] <= self.limits_xy_lod[1]
            if self.boundary != "closed":
                ordered = self.limits_xy_lod[0] < self.limits_xy_lod[1]
            if not bool(jnp.all(ordered)):
                raise ValueError("Box limits must be ordered and nonempty")
            if bool(jnp.any(self.radii_lod != 0)):
                raise ValueError("Unused radii must be zero")
        else:
            inner, outer = self.radii_lod.tolist()
            if not 0 <= inner < outer or bool(jnp.any(self.limits_xy_lod != 0)):
                raise ValueError("Annulus requires ordered radii and zero box limits")

    def contains(self, source_xy_lod):
        """Test original signed coordinates after explicit subnormal flushing."""
        query = flush_subnormal(source_xy_lod)
        if self.kind == "box":
            low = jnp.all(query >= self.limits_xy_lod[0])
            upper = query <= self.limits_xy_lod[1]
            if self.boundary != "closed":
                upper = query < self.limits_xy_lod[1]
            return low & jnp.all(upper)
        radius2 = jnp.sum(query * query)
        lower = radius2 >= self.radii_lod[0] ** 2
        upper = radius2 <= self.radii_lod[1] ** 2
        if self.boundary != "closed":
            upper = radius2 < self.radii_lod[1] ** 2
        return lower & upper


def normal_or_zero(value):
    """Check host binary64 values without backend denormal-mode dependence."""
    return math.isfinite(value) and (value == 0 or abs(value) >= sys.float_info.min)


def coverage(query, lattice, reflected):
    """Check floating floor ranges before any potentially overflowing int cast."""
    floor = exact_quotient_floor_v1(
        query, lattice.origin_xy_lod, lattice.spacing_xy_lod
    )
    size = jnp.asarray(lattice.shape_yx[::-1], dtype=jnp.float64)
    lower = 16 - size if reflected else jnp.full((2,), 15.0)
    return jnp.all(jnp.isfinite(floor) & (floor >= lower) & (floor <= size - 17))


@eqx.filter_jit
def _candidate_membership(points, domain, lattice, reflected):
    return jax.vmap(lambda q: domain.contains(q) & coverage(q, lattice, reflected))(
        points
    )


def safe_query(domain, lattice, reflected):
    """Find and verify a point in the domain/coverage intersection.

    Test inclusive coverage endpoints before their inward neighbours, box
    corners and centers, and annular candidates on corner rays and box sides
    (including axis crossings). Every candidate uses the read-time membership
    and exact floor rule. This finite list is not a proof of nonemptiness for
    every floating-point geometry: an unresolved intersection returns None,
    causing admission to refuse with INCOMPLETE_STENCIL.
    """
    origin = lattice.origin_xy_lod.tolist()
    spacing = lattice.spacing_xy_lod.tolist()
    size = lattice.shape_yx[::-1]
    low = [
        o + h * (16 - n if reflected else 15)
        for o, h, n in zip(origin, spacing, size, strict=True)
    ]
    high = [o + h * (n - 16) for o, h, n in zip(origin, spacing, size, strict=True)]
    if domain.kind == "box":
        bounds = domain.limits_xy_lod.tolist()
        low = [max(a, b) for a, b in zip(low, bounds[0], strict=True)]
        high = [min(a, b) for a, b in zip(high, bounds[1], strict=True)]
    if any(a > b for a, b in zip(low, high, strict=True)):
        return None
    axes = [
        (
            a,
            b,
            math.nextafter(a, math.inf),
            math.nextafter(b, -math.inf),
            a / 2 + b / 2,
            max(a, min(0.0, b)),
        )
        for a, b in zip(low, high, strict=True)
    ]
    candidates = [[x, y] for x in axes[0] for y in axes[1]]
    if domain.kind == "annulus":
        inner, outer = domain.radii_lod.tolist()
        radii = (
            inner,
            outer,
            (inner + outer) / 2,
            math.nextafter(inner, math.inf),
            math.nextafter(outer, -math.inf),
        )
        directions = [[x, y] for x in (low[0], high[0]) for y in (low[1], high[1])]
        directions += [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
        for radius in radii:
            for x, y in directions:
                norm = math.hypot(x, y)
                if norm:
                    candidates.append([radius * (x / norm), radius * (y / norm)])
            # Circle/box-side intersections also cover narrow off-axis regions.
            for axis in range(2):
                for value in (low[axis], high[axis], 0.0):
                    if abs(value) <= radius:
                        other = math.sqrt((radius - abs(value)) * (radius + abs(value)))
                        for sign in (-1, 1):
                            point = [value, sign * other]
                            candidates.append(point if axis == 0 else point[::-1])
    points = jnp.asarray(candidates, dtype=jnp.float64)
    valid = _candidate_membership(points, domain, lattice, reflected).tolist()
    for index, permitted in enumerate(valid):
        if permitted:
            return points[index]
    return None
