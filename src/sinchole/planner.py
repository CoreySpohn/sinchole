"""Tap-count planning for prefiltered reads from delivery metadata alone."""

from dataclasses import dataclass

from sinchole.evidence import PlanRefused, Reason
from sinchole.grid import _sampling_reasons
from sinchole.kernel import PrefilteredKernelProfile
from sinchole.prefilter import design_grid, design_record

_TAP_RANGE = range(5, 11)


@dataclass(frozen=True)
class PrefilterPlan:
    """Host result of ``plan_prefilter``; call require() for the profile.

    ``tone_error_two_axis`` is the estimated worst two-axis tone error of the
    chosen profile (NaN on refusal), ``kernel_design`` its full design record,
    and ``candidates`` the (taps, two-axis tone error) pairs that were tried, in
    order. These are sampled maxima of the kernel's response on the declared
    band, not error bounds on a delivered image.
    """

    profile: PrefilteredKernelProfile | None
    tone_error_two_axis: float
    candidates: tuple[tuple[int, float], ...]
    reason_bits: int
    kernel_design: tuple[tuple[str, float], ...] = ()

    @property
    def planned(self):
        """Whether some tap count met the request."""
        return self.profile is not None

    def require(self) -> PrefilteredKernelProfile:
        """Return the profile, or raise PlanRefused carrying this plan."""
        if self.profile is None:
            raise PlanRefused(self)
        return self.profile


def plan_prefilter(*, declared_lattice, products, reflection, domain, max_tone_error):
    """Choose the fewest taps whose design tone error meets ``max_tone_error``.

    Only metadata is read: no samples are touched and nothing is admitted. For
    5 to 10 taps, in order, the planner designs the filter that admission would
    build for this lattice, band and domain, and stops at the first tap count
    whose estimated two-axis tone error (e_x + e_y + e_x * e_y) is at or below
    ``max_tone_error``. It never tries more than 10 taps: a refusal means the
    lattice needs the raw profile or a looser request.

    The tone error is a dimensionless property of the kernel on the declared
    band. It is not ``ReadPolicy.max_abs_error`` and does not include the
    rounding of coefficients to the storage dtype, which for float32 storage is
    of order 1e-7 of the local intensity scale.

    Args:
        declared_lattice: The delivery's Lattice, as passed to ``admit_grid``.
        products: The delivery's Products; their shortest declared wavelength
            sets the design band.
        reflection: The delivery's Reflection declaration.
        domain: The Domain reads will be served in; table ends it cannot reach
            do not count against a tap count.
        max_tone_error: Largest acceptable two-axis tone error, positive.

    Returns:
        A PrefilterPlan. Refusal reasons are NYQUIST_FAIL or KERNEL_MARGIN_FAIL
        when the lattice is too coarse for any prefiltered profile,
        INCOMPLETE_STENCIL when no tap count has a covered read in the domain,
        and TOLERANCE_EXCEEDED when 10 taps do not reach the request.
    """
    if not max_tone_error > 0:
        raise ValueError("max_tone_error must be positive")
    reflected = reflection.kind != "none"
    spacing = declared_lattice.spacing_xy_lod.tolist()
    margin = PrefilteredKernelProfile().margin_limit
    reasons = _sampling_reasons(products, spacing, margin)
    if reasons:
        return PrefilterPlan(None, float("nan"), (), int(reasons))
    candidates = []
    for taps in _TAP_RANGE:
        profile = PrefilteredKernelProfile(taps=taps)
        if min(declared_lattice.shape_yx) < profile.minimum_size(reflected):
            continue
        record = design_record(
            design_grid(profile, products, declared_lattice, reflected, domain)
        )
        error = dict(record)["tone_error_two_axis"]
        candidates.append((taps, error))
        if error <= max_tone_error:
            return PrefilterPlan(profile, error, tuple(candidates), 0, record)
    finite = any(error < float("inf") for _, error in candidates)
    reason = Reason.TOLERANCE_EXCEEDED if finite else Reason.INCOMPLETE_STENCIL
    return PrefilterPlan(None, float("nan"), tuple(candidates), int(reason))
