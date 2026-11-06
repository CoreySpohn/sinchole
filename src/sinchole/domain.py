"""Signed permission domains and independent full-stencil coverage."""

import math
import sys

import equinox as eqx
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


def safe_query(domain, lattice, reflected):
    """Find a host-validated point in the domain/coverage intersection.

    The coverage region is a rectangle. For an annulus, the segment from its
    nearest point to its farthest corner spans every attainable radius.
    """
    if domain.kind == "box":
        lower, upper = domain.limits_xy_lod.tolist()
        candidates = [[a / 2 + b / 2 for a, b in zip(lower, upper, strict=True)]]
        candidates.extend(
            [[x, y] for x in (lower[0], upper[0]) for y in (lower[1], upper[1])]
        )
        for candidate in candidates:
            point = jnp.asarray(candidate, dtype=jnp.float64)
            if bool(domain.contains(point) & coverage(point, lattice, reflected)):
                return point
    origin = lattice.origin_xy_lod.tolist()
    spacing = lattice.spacing_xy_lod.tolist()
    size = lattice.shape_yx[::-1]
    low = [
        o + h * (16 - n if reflected else 15)
        for o, h, n in zip(origin, spacing, size, strict=True)
    ]
    high = [o + h * (n - 16) for o, h, n in zip(origin, spacing, size, strict=True)]
    # Move into the covered side of rounded coordinate edges.
    for _ in range(8):
        low = [math.nextafter(v, math.inf) for v in low]
        high = [math.nextafter(v, -math.inf) for v in high]
    if domain.kind == "box":
        bounds = domain.limits_xy_lod.tolist()
        low = [max(a, b) for a, b in zip(low, bounds[0], strict=True)]
        high = [min(a, b) for a, b in zip(high, bounds[1], strict=True)]
        candidate = [(a + b) / 2 for a, b in zip(low, high, strict=True)]
    else:
        nearest = [max(a, min(0.0, b)) for a, b in zip(low, high, strict=True)]
        farthest = [a if abs(a) > abs(b) else b for a, b in zip(low, high, strict=True)]
        inner, outer = domain.radii_lod.tolist()
        rlow = max(inner, math.hypot(*nearest))
        rhigh = min(outer, math.hypot(*farthest))
        if rlow > rhigh:
            return None
        target = (rlow + rhigh) / 2
        left, right = 0.0, 1.0
        for _ in range(64):
            fraction = (left + right) / 2
            candidate = [
                a + fraction * (b - a) for a, b in zip(nearest, farthest, strict=True)
            ]
            if math.hypot(*candidate) < target:
                left = fraction
            else:
                right = fraction
    if any(a > b for a, b in zip(low, high, strict=True)):
        return None
    result = jnp.asarray(candidate, dtype=jnp.float64)
    if bool(domain.contains(result) & coverage(result, lattice, reflected)):
        return result
    return None
