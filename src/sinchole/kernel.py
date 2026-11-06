"""Raw 32-tap Kaiser-sinc weights and reproducible stencil selection."""

import equinox as eqx
import jax
import jax.numpy as jnp

from sinchole._types import check_static


class KernelProfile(eqx.Module):
    """Fixed unnormalized W=16, beta=12 kernel with versioned floor semantics."""

    name: str = eqx.field(static=True, default="kaiser_w16_beta12_raw_v1")
    floor_rule: str = eqx.field(static=True, default="exact_quotient_floor_v1")

    def __check_init__(self):
        """Validate the declared structure before numerical use."""
        check_static(self)
        if self.name != "kaiser_w16_beta12_raw_v1":
            raise ValueError("Unsupported kernel profile")
        if self.floor_rule != "exact_quotient_floor_v1":
            raise ValueError("Unsupported floor rule")

    @property
    def identity(self):
        """Return the joint weight and stencil arithmetic identity."""
        return f"{self.name}:{self.floor_rule}"


def flush_subnormal(value):
    """Replace binary64 subnormals by signed zero using integer bits."""
    bits = jax.lax.bitcast_convert_type(value, jnp.int64)
    exponent = (bits >> 52) & 0x7FF
    zero = jax.lax.bitcast_convert_type(bits & jnp.int64(-(2**63)), jnp.float64)
    return jnp.where(exponent == 0, zero, value)


def exact_quotient_floor_v1(query, origin, spacing):
    """Return floating integer floors of the exact quotient of rounded d by h.

    Query subnormals become signed zero before the single subtraction d=q-o.
    Callers must check the resulting float index range before integer conversion.
    The remainder determines the side of a cell edge without reciprocal rounding.
    """
    difference = flush_subnormal(query) - origin
    remainder = jnp.fmod(difference, spacing)
    quotient = jnp.round((difference - remainder) / spacing)
    return quotient - jnp.where((remainder != 0) & (difference < 0), 1.0, 0.0)


def weights(argument):
    """Evaluate raw Kaiser-sinc weights, with safe inactive derivative lanes."""
    active = jnp.abs(argument) < 16.0
    inside = jnp.where(active, 1.0 - (argument / 16.0) ** 2, 1.0)
    window = jnp.i0(12.0 * jnp.sqrt(inside)) / jnp.i0(jnp.float64(12.0))
    return jnp.where(active, jnp.sinc(argument) * window, 0.0)


def stencil(query, origin, spacing):
    """Return signed int64 taps and float64 weights for a prevalidated query."""
    query = flush_subnormal(query)
    floor = exact_quotient_floor_v1(query, origin, spacing).astype(jnp.int64)
    taps = floor[:, None] + jnp.arange(-15, 17, dtype=jnp.int64)
    phase = (query - origin) / spacing
    return taps, weights(phase[:, None] - taps)
