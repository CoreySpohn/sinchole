"""Immutable intensity products and host admission of complete source lattices."""

import hashlib
import math
from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp

from sinchole._types import F64, I32, Boolean, check_array, check_shape, check_static
from sinchole.domain import in_supported_range, normal_or_zero, safe_query
from sinchole.evidence import Admission, Reason, Receipt
from sinchole.kernel import KernelProfile


class Frame(eqx.Module):
    """Reference angular units: theta_rad=s*lambda_ref_nm*1e-9/diameter_m."""

    reference_wavelength_nm: F64 = eqx.field(converter=jnp.asarray)
    diameter_m: F64 = eqx.field(converter=jnp.asarray)
    diameter_convention: str = eqx.field(static=True)
    coordinate_model: str = eqx.field(static=True, default="paraxial_entrance_tilt")

    def __check_init__(self):
        """Validate the declared structure before numerical use."""
        check_static(self)
        check_array(self.reference_wavelength_nm, (), (jnp.float64,))
        check_array(self.diameter_m, (), (jnp.float64,))


class PixelGrid(eqx.Module):
    """Fixed detector geometry; basis columns map column and row displacements."""

    shape_yx: tuple[int, int] = eqx.field(static=True)
    center_xy_pix: F64 = eqx.field(converter=jnp.asarray)
    basis_xy_lod: F64 = eqx.field(converter=jnp.asarray)
    measure: str = eqx.field(static=True)

    def __check_init__(self):
        """Validate the declared structure before numerical use."""
        check_static(self)
        check_shape(self.shape_yx)
        check_array(self.center_xy_pix, (2,), (jnp.float64,))
        check_array(self.basis_xy_lod, (2, 2), (jnp.float64,))


class Normalization(eqx.Module):
    """Source-independent product conversions; reads retain delivered units."""

    reference: str = eqx.field(static=True)
    photon_conversion: F64 = eqx.field(converter=jnp.asarray)
    reference_uncertainty_fraction: F64 = eqx.field(converter=jnp.asarray)
    included_factors: tuple[str, ...] = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)

    def __check_init__(self):
        """Validate the declared structure before numerical use."""
        check_static(self)
        check_array(
            self.photon_conversion, (self.photon_conversion.size,), (jnp.float64,)
        )
        check_array(
            self.reference_uncertainty_fraction,
            self.photon_conversion.shape,
            (jnp.float64,),
        )


class Premises(eqx.Module):
    """Six ordered physical premises, evidence codes and provenance sources."""

    values: Boolean = eqx.field(converter=jnp.asarray)
    evidence: I32 = eqx.field(converter=jnp.asarray)
    source_ids: tuple[str, ...] = eqx.field(static=True)

    def __check_init__(self):
        """Validate the declared structure before numerical use."""
        check_static(self)
        check_array(self.values, (6,), (jnp.bool_,))
        check_array(self.evidence, (6,), (jnp.int32,))
        if not isinstance(self.source_ids, tuple) or len(self.source_ids) != 6:
            raise ValueError("Premises require six provenance IDs")


class Products(eqx.Module):
    """Exact monochromatic or fixed band-mean products with explicit support."""

    frame: Frame
    pixels: PixelGrid
    normalization: Normalization
    kind: str = eqx.field(static=True)
    wavelength_nm: F64 = eqx.field(converter=jnp.asarray)
    limits_nm: F64 = eqx.field(converter=jnp.asarray)
    sampling_min_nm: F64 = eqx.field(converter=jnp.asarray)
    campaign_sampling_min_nm: F64 = eqx.field(converter=jnp.asarray)
    support_extent_xy: F64 = eqx.field(converter=jnp.asarray)
    support_evidence: I32 = eqx.field(converter=jnp.asarray)
    support_source_ids: tuple[str, ...] = eqx.field(static=True)
    premises: Premises
    optical_id: str = eqx.field(static=True)
    spectral_weight_id: str = eqx.field(static=True)
    identity: str = eqx.field(static=True)

    def __check_init__(self):
        """Validate the declared structure before numerical use."""
        check_static(self)
        count = self.wavelength_nm.size
        if count < 1:
            raise ValueError("At least one product is required")
        for value in (
            self.wavelength_nm,
            self.sampling_min_nm,
            self.normalization.photon_conversion,
        ):
            check_array(value, (count,), (jnp.float64,))
        check_array(self.limits_nm, (count, 2), (jnp.float64,))
        check_array(self.support_extent_xy, (count, 2), (jnp.float64,))
        check_array(self.support_evidence, (count,), (jnp.int32,))
        check_array(self.campaign_sampling_min_nm, (), (jnp.float64,))
        if len(self.support_source_ids) != count:
            raise ValueError("Support source count differs from product count")


class Lattice(eqx.Module):
    """Authoritative canonical origin, pitch and stored rectangle shape."""

    origin_xy_lod: F64 = eqx.field(converter=jnp.asarray)
    spacing_xy_lod: F64 = eqx.field(converter=jnp.asarray)
    shape_yx: tuple[int, int] = eqx.field(static=True)

    def __check_init__(self):
        """Validate the declared structure before numerical use."""
        check_static(self)
        check_shape(self.shape_yx)
        check_array(self.origin_xy_lod, (2,), (jnp.float64,))
        check_array(self.spacing_xy_lod, (2,), (jnp.float64,))


class Reflection(eqx.Module):
    """Declared two-axis detector flips; admission computes check_passed itself.

    Reflection requires 0 < tolerance <= 1e-6 of the local peak.
    Axis evidence records exact native-value identity separately from relative
    residuals. Without a positive local peak, only exact identity passes.
    """

    kind: str = eqx.field(static=True)
    axes: Boolean = eqx.field(converter=jnp.asarray)
    declaration_id: str = eqx.field(static=True)
    argument_id: str = eqx.field(static=True)
    evidence: I32 = eqx.field(converter=jnp.asarray)
    check_report_id: str = eqx.field(static=True)
    check_passed: Boolean = eqx.field(converter=jnp.asarray)
    tolerance: F64 = eqx.field(
        default_factory=lambda: jnp.asarray(1e-6), converter=jnp.asarray
    )

    def __check_init__(self):
        """Validate the declared structure before numerical use."""
        check_static(self)
        check_array(self.axes, (2,), (jnp.bool_,))
        check_array(self.evidence, (), (jnp.int32,))
        check_array(self.check_passed, (self.check_passed.size,), (jnp.bool_,))
        check_array(self.tolerance, (), (jnp.float64,))


class SampledGrid(eqx.Module):
    """Canonical native float32/64 samples [product, sy, sx, row, column]."""

    intensity: jax.Array
    source: Lattice
    products: Products
    reflection: Reflection
    receipt_id: str = eqx.field(static=True)
    _empirical_max_abs: F64 = eqx.field(default_factory=lambda: jnp.asarray(jnp.nan))
    _has_checks: bool = eqx.field(static=True, default=False)

    def __check_init__(self):
        """Validate the declared structure before numerical use."""
        check_static(self)
        count = self.products.wavelength_nm.size
        check_array(
            self.intensity,
            (count, *self.source.shape_yx, *self.products.pixels.shape_yx),
            (jnp.float32, jnp.float64),
        )
        check_array(self.reflection.check_passed, (count,), (jnp.bool_,))
        check_array(self._empirical_max_abs, (), (jnp.float64,))


class CheckSet(eqx.Module):
    """Independent off-node images and optional independently supplied core peaks."""

    source_xy_lod: F64 = eqx.field(converter=jnp.asarray)
    intensity: jax.Array = eqx.field(converter=jnp.asarray)
    product_index: I32 = eqx.field(converter=jnp.asarray)
    core_peak: F64 = eqx.field(converter=jnp.asarray)
    provenance_id: str = eqx.field(static=True)
    reference_method_id: str = eqx.field(static=True)
    target_id: str = eqx.field(static=True)

    def __check_init__(self):
        """Validate the declared structure before numerical use."""
        check_static(self)
        count = self.product_index.size
        check_array(self.source_xy_lod, (count, 2), (jnp.float64,))
        check_array(self.product_index, (count,), (jnp.int32,))
        check_array(self.core_peak, (count,), (jnp.float64,))
        if self.intensity.ndim != 3:
            raise ValueError("Check images require [query, row, column]")
        check_array(
            self.intensity,
            (count, *self.intensity.shape[1:]),
            (jnp.float32, jnp.float64),
        )


class MirrorCheckSet(eqx.Module):
    """Independent signed node images; indices are signed about the zero anchor.

    Each product must include x-negative, y-negative and both-negative checks.
    These finite comparisons are empirical evidence, never a global bound.
    """

    source_xy_lod: F64 = eqx.field(converter=jnp.asarray)
    indices_xy: I32 = eqx.field(converter=jnp.asarray)
    intensity: jax.Array = eqx.field(converter=jnp.asarray)
    product_index: I32 = eqx.field(converter=jnp.asarray)
    provenance_id: str = eqx.field(static=True)

    def __check_init__(self):
        """Validate the declared structure before numerical use."""
        check_static(self)
        count = self.product_index.size
        check_array(self.source_xy_lod, (count, 2), (jnp.float64,))
        check_array(self.indices_xy, (count, 2), (jnp.int32,))
        check_array(self.product_index, (count,), (jnp.int32,))
        if self.intensity.ndim != 3:
            raise ValueError("Mirror images require [query, row, column]")
        check_array(
            self.intensity,
            (count, *self.intensity.shape[1:]),
            (jnp.float32, jnp.float64),
        )


def _positive(values):
    return bool(jnp.all(jnp.isfinite(values) & (values > 0)))


def _metadata_reasons(products, reflection):
    reasons = Reason(0)
    frame, pixels, norm = products.frame, products.pixels, products.normalization
    premises = products.premises
    if (
        not bool(jnp.all((premises.evidence >= 1) & (premises.evidence <= 2)))
        or not all(premises.source_ids)
        or not all(
            (products.identity, products.optical_id, products.spectral_weight_id)
        )
    ):
        reasons |= Reason.METADATA_MISSING
    if not bool(jnp.all(premises.values[:4])):
        reasons |= Reason.FRAME_UNSUPPORTED
    if not bool(premises.values[5]):
        reasons |= Reason.METADATA_CONFLICT
    if not bool(premises.values[4]):
        reasons |= Reason.NORMALIZATION_UNSUPPORTED
    if (
        not _positive(frame.reference_wavelength_nm)
        or not _positive(frame.diameter_m)
        or not frame.diameter_convention
        or frame.coordinate_model != "paraxial_entrance_tilt"
    ):
        reasons |= Reason.FRAME_UNSUPPORTED
    basis = pixels.basis_xy_lod
    norms = jnp.sum(basis * basis, axis=0)
    dot = float(jnp.sum(basis[:, 0] * basis[:, 1]))
    orthogonal = abs(dot) <= 8 * math.ulp(1.0) * math.sqrt(float(jnp.prod(norms)))
    if (
        not _positive(norms)
        or not bool(jnp.all(jnp.isfinite(basis)))
        or not bool(jnp.all(jnp.isfinite(pixels.center_xy_pix)))
        or not orthogonal
        or pixels.measure
        not in ("pixel_integral", "pixel_mean_density", "point_density")
    ):
        reasons |= Reason.FRAME_UNSUPPORTED
    if reflection.kind == "two_axis_pixel_flip_v1":
        expected = jnp.asarray([(n - 1) / 2 for n in pixels.shape_yx[::-1]])
        if (
            not bool(jnp.all(pixels.center_xy_pix == expected))
            or float(basis[0, 1]) != 0
            or float(basis[1, 0]) != 0
        ):
            reasons |= Reason.FRAME_UNSUPPORTED
    if (
        norm.reference not in ("incident_photon", "unocculted_peak")
        or not _positive(norm.photon_conversion)
        or not norm.provenance_id
    ):
        reasons |= Reason.NORMALIZATION_UNSUPPORTED
    uncertainty = norm.reference_uncertainty_fraction
    if not bool(
        jnp.all(
            jnp.isnan(uncertainty) | (jnp.isfinite(uncertainty) & (uncertainty >= 0))
        )
    ):
        reasons |= Reason.NORMALIZATION_UNSUPPORTED
    if (
        not _positive(products.support_extent_xy)
        or not bool(jnp.all(products.support_evidence == 1))
        or not all(products.support_source_ids)
    ):
        reasons |= Reason.UNKNOWN_BANDWIDTH
    if not _positive(products.sampling_min_nm):
        reasons |= Reason.METADATA_MISSING
    if products.kind not in ("monochromatic", "band_mean"):
        reasons |= Reason.METADATA_CONFLICT
    if (
        not _positive(products.wavelength_nm)
        or not _positive(products.limits_nm)
        or bool(jnp.any(products.limits_nm[:, 0] > products.limits_nm[:, 1]))
        or bool(jnp.any(products.wavelength_nm < products.limits_nm[:, 0]))
        or bool(jnp.any(products.wavelength_nm > products.limits_nm[:, 1]))
        or bool(jnp.any(products.sampling_min_nm > products.wavelength_nm))
    ):
        reasons |= Reason.METADATA_CONFLICT
    if products.kind == "monochromatic" and not bool(
        jnp.all(products.limits_nm == products.wavelength_nm[:, None])
    ):
        reasons |= Reason.METADATA_CONFLICT
    campaign = float(products.campaign_sampling_min_nm)
    if not math.isnan(campaign) and (
        not math.isfinite(campaign)
        or campaign <= 0
        or bool(jnp.any(campaign > products.sampling_min_nm))
    ):
        reasons |= Reason.METADATA_CONFLICT
    return reasons


def _sampling_reasons(products, spacing):
    reasons = Reason(0)
    reference = float(products.frame.reference_wavelength_nm)
    campaign = float(products.campaign_sampling_min_nm)
    for minimum, extents in zip(
        products.sampling_min_nm.tolist(),
        products.support_extent_xy.tolist(),
        strict=True,
    ):
        minima = [minimum] if math.isnan(campaign) else [minimum, campaign]
        for value in minima:
            if not math.isfinite(value) or value <= 0:
                continue
            for extent, pitch in zip(extents, spacing, strict=True):
                # Python binary64 operations preserve the declared separate rounding.
                sigma = reference / value
                bandwidth = sigma * extent
                rho = (2.0 * bandwidth) * pitch
                if not rho < 1.0:
                    reasons |= Reason.NYQUIST_FAIL
                if not rho <= 0.76:
                    reasons |= Reason.KERNEL_MARGIN_FAIL
    return reasons


def _offset_evidence(offsets, lattice, indices, original_dtype=None):
    dtype = offsets.dtype if original_dtype is None else jnp.dtype(original_dtype)
    scale = jnp.maximum(
        jnp.max(jnp.abs(offsets), axis=0), jnp.abs(lattice.spacing_xy_lod)
    ).astype(dtype)
    tau = (4 * (jnp.nextafter(scale, jnp.asarray(jnp.inf, dtype)) - scale)).astype(
        jnp.float64
    )
    # Separate eager operations prevent fused multiply-add canonicalization.
    displacement = indices.astype(jnp.float64) * lattice.spacing_xy_lod
    expected = lattice.origin_xy_lod + displacement
    residual = jnp.max(jnp.abs(offsets.astype(jnp.float64) - expected), axis=0)
    valid = (
        bool(jnp.all(jnp.isfinite(offsets)))
        and bool(jnp.all(residual <= tau))
        and bool(jnp.all(tau < 1e-4 * jnp.abs(lattice.spacing_xy_lod)))
    )
    return valid, tuple(tau.tolist()), tuple(residual.tolist())


def _local_peak(image, source, pixels):
    rows, columns = jnp.indices(pixels.shape_yx, dtype=jnp.float64)
    displacement = jnp.stack(
        (columns - pixels.center_xy_pix[0], rows - pixels.center_xy_pix[1])
    )
    coordinates = jnp.einsum("ij,jrc->irc", pixels.basis_xy_lod, displacement)
    selected = jnp.sum((coordinates - source[:, None, None]) ** 2, axis=0) <= 1.0
    return float(jnp.max(jnp.where(selected, image, -jnp.inf)))


def _symmetry_checks(cube, lattice, products, reflection, checks):
    reasons = Reason(0)
    count = cube.shape[0]
    passed = [True] * count
    residuals, axis_residuals, axis_exact = [], [], []
    tolerance = float(reflection.tolerance)
    declared = (
        bool(jnp.all(reflection.axes))
        and int(reflection.evidence) in (1, 2)
        and all(
            (
                reflection.declaration_id,
                reflection.argument_id,
                reflection.check_report_id,
            )
        )
        and math.isfinite(tolerance)
        and 0 < tolerance <= 1e-6
    )
    if not declared:
        return Reason.SYMMETRY_UNSUPPORTED, jnp.zeros(count, dtype=bool), (), (), ()
    if checks is None or checks.product_index.size == 0:
        reasons |= Reason.SYMMETRY_CHECK_MISSING
    else:
        if checks.intensity.shape[1:] != products.pixels.shape_yx:
            raise ValueError("Mirror detector shape differs from samples")
        categories = [set() for _ in range(count)]
        valid_offsets, _, _ = _offset_evidence(
            checks.source_xy_lod, lattice, checks.indices_xy
        )
        valid = (
            valid_offsets
            and bool(jnp.all(jnp.isfinite(checks.intensity) & (checks.intensity >= 0)))
            and bool(jnp.all(jnp.isfinite(checks.source_xy_lod)))
            and bool(checks.provenance_id)
        )
        if not valid:
            reasons |= Reason.SYMMETRY_REFERENCE_INVALID
        for index in range(checks.product_index.size):
            product = int(checks.product_index[index])
            ix, iy = checks.indices_xy[index].tolist()
            if not (
                0 <= product < count
                and abs(ix) < cube.shape[2]
                and abs(iy) < cube.shape[1]
                and (ix < 0 or iy < 0)
            ):
                reasons |= Reason.SYMMETRY_REFERENCE_INVALID
                continue
            categories[product].add((ix < 0, iy < 0))
            reference = checks.intensity[index].astype(jnp.float64)
            predicted = cube[product, abs(iy), abs(ix)].astype(jnp.float64)
            if ix < 0:
                predicted = predicted[:, ::-1]
            if iy < 0:
                predicted = predicted[::-1, :]
            peak = _local_peak(reference, checks.source_xy_lod[index], products.pixels)
            if not math.isfinite(peak) or peak <= 0:
                reasons |= Reason.SYMMETRY_REFERENCE_INVALID
                passed[product] = False
                residuals.append(math.inf)
                continue
            residual = float(jnp.max(jnp.abs(reference - predicted))) / peak
            residuals.append(residual)
            if not residual <= tolerance:
                reasons |= Reason.SYMMETRY_CHECK_FAILED
                passed[product] = False
        if any(
            not {(True, False), (False, True), (True, True)} <= group
            for group in categories
        ):
            reasons |= Reason.SYMMETRY_CHECK_MISSING
    for product in range(count):
        for axis, size in ((0, cube.shape[1]), (1, cube.shape[2])):
            chunk = max(1, min(32, 65536 // math.prod(products.pixels.shape_yx)))
            for start in range(0, size, chunk):
                stop = min(size, start + chunk)
                images = (
                    cube[product, start:stop, 0]
                    if axis == 0
                    else cube[product, 0, start:stop]
                )
                indices = jnp.arange(start, stop, dtype=jnp.float64)
                zeros = jnp.zeros_like(indices)
                source = jnp.stack(
                    (zeros, indices) if axis == 0 else (indices, zeros), axis=1
                )
                source = source * lattice.spacing_xy_lod
                exact, relative = _axis_residuals(images, source, products.pixels, axis)
                values = relative.tolist()
                axis_residuals.extend(values)
                identities = exact.tolist()
                axis_exact.extend(identities)
                if any(
                    not (same or value <= tolerance)
                    for same, value in zip(identities, values, strict=True)
                ):
                    reasons |= Reason.SYMMETRY_CHECK_FAILED
                    passed[product] = False
    return (
        reasons,
        jnp.asarray(passed),
        tuple(residuals),
        tuple(axis_residuals),
        tuple(axis_exact),
    )


@eqx.filter_jit
def _axis_residuals(images, sources, pixels, axis):
    integer = jnp.int32 if images.dtype == jnp.float32 else jnp.int64
    bits = jax.lax.bitcast_convert_type(images, integer)
    # Numeric identity allows both signed zeros but preserves subnormal samples.
    bits = jnp.where((bits & jnp.iinfo(integer).max) == 0, 0, bits)
    flipped_bits = bits[:, :, ::-1] if axis == 0 else bits[:, ::-1, :]
    exact = jnp.all(bits == flipped_bits, axis=(1, 2))
    images = images.astype(jnp.float64)
    rows, columns = jnp.indices(pixels.shape_yx, dtype=jnp.float64)
    displacement = jnp.stack(
        (columns - pixels.center_xy_pix[0], rows - pixels.center_xy_pix[1])
    )
    coordinates = jnp.einsum("ij,jrc->irc", pixels.basis_xy_lod, displacement)
    selected = (
        jnp.sum((coordinates[None] - sources[:, :, None, None]) ** 2, axis=1) <= 1.0
    )
    peaks = jnp.max(jnp.where(selected, images, -jnp.inf), axis=(1, 2))
    flipped = images[:, :, ::-1] if axis == 0 else images[:, ::-1, :]
    differences = jnp.max(jnp.abs(images - flipped), axis=(1, 2))
    positive = peaks > 0
    relative = differences / jnp.where(positive, peaks, 1.0)
    # Exact invariance does not require an in-stamp relative denominator.
    return exact, jnp.where(positive, relative, jnp.nan)


def _digest(*values):
    digest = hashlib.sha256()
    for value in values:
        leaves, tree = jax.tree_util.tree_flatten(value)
        digest.update(str(tree).encode())
        for leaf in leaves:
            if eqx.is_array(leaf):
                digest.update(str((leaf.shape, leaf.dtype)).encode())
                flat = leaf.reshape(-1)
                for start in range(0, flat.size, 65536):
                    digest.update(jax.device_get(flat[start : start + 65536]).tobytes())
            else:
                digest.update(repr(leaf).encode())
    return digest.hexdigest()


_DEFAULT_PROFILE = KernelProfile()


def admit_grid(
    intensity_nodes,
    offsets_xy_lod,
    *,
    declared_lattice,
    indices_xy,
    products,
    reflection,
    domain,
    profile=_DEFAULT_PROFILE,
    checks=None,
    memory,
    mirror_checks=None,
    offset_original_dtype=None,
    provenance=(),
):
    """Admit complete canonical samples, accumulating semantic refusal reasons.

    Args:
        intensity_nodes: Native float32/64 array [P,N,Py,Px].
        offsets_xy_lod: Converted float32/64 source offsets [N,2].
        declared_lattice: Authoritative origin, spacing and shape; never fitted.
        indices_xy: Rebased int32 source indices [N,2], in arbitrary row order.
        products: Physical frame, support, premises and normalization.
        reflection: Explicit full-grid or two-axis reflection declaration.
        domain: Signed served permission region intersected with stencil coverage.
        profile: Fixed raw Kaiser-sinc profile and versioned floor rule.
        checks: Optional independent off-node empirical comparisons.
        memory: Explicit resident, output, scratch and total byte limits.
        mirror_checks: Required independent signed node checks for quadrants.
        offset_original_dtype: Original float32/64 precision before conversion;
            None uses offsets_xy_lod.dtype. Four ULP are evaluated in this dtype
            at the converted coordinate scale, without recasting the offsets.
        provenance: Tuple of (key, text) string pairs retained verbatim in the
            host receipt and hashed into its identity; never traced.

    Returns:
        An Admission with grid=None on semantic refusal. Admission.require()
        raises AdmissionRefused (or a resource-specific subclass) on refusal.
        Structural Python shape/dtype mistakes raise TypeError or ValueError.
    """
    # Local import breaks the intentional admission/reader construction cycle.
    from sinchole.reader import Interpolator, ReadPolicy, Tiling, estimate_memory

    if not isinstance(provenance, tuple) or any(
        not isinstance(pair, tuple)
        or len(pair) != 2
        or not all(isinstance(value, str) for value in pair)
        for pair in provenance
    ):
        raise TypeError("Provenance requires a tuple of (key, text) string pairs")
    if not jax.config.x64_enabled:
        return Admission(
            None, Receipt(provenance=provenance), int(Reason.PRECISION_UNAVAILABLE)
        )
    if not hasattr(intensity_nodes, "shape") or len(intensity_nodes.shape) != 4:
        raise ValueError("Nodes require [product, source, row, column]")
    allocations = estimate_memory(
        intensity_nodes.nbytes,
        products.pixels.shape_yx,
        Tiling(),
        1,
        admission=True,
    )
    if any(
        allocations[key] > getattr(memory, f"max_{key}_bytes")
        for key in ("resident", "output", "scratch", "total")
    ):
        receipt = Receipt(
            allocation_estimate=tuple(allocations.items()), provenance=provenance
        )
        return Admission(None, receipt, int(Reason.MEMORY_LIMIT))
    nodes = jnp.asarray(intensity_nodes)
    offsets = jnp.asarray(offsets_xy_lod)
    indices = jnp.asarray(indices_xy)
    offset_dtype = (
        offsets.dtype
        if offset_original_dtype is None
        else jnp.dtype(offset_original_dtype)
    )
    if offset_dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
        raise TypeError("Original offsets require float32 or float64 precision")
    if nodes.ndim != 4:
        raise ValueError("Nodes require [product, source, row, column]")
    count, rows = nodes.shape[:2]
    check_array(
        nodes,
        (products.wavelength_nm.size, rows, *products.pixels.shape_yx),
        (jnp.float32, jnp.float64),
    )
    check_array(offsets, (rows, 2), (jnp.float32, jnp.float64))
    check_array(indices, (rows, 2), (jnp.int32,))
    check_array(reflection.check_passed, (count,), (jnp.bool_,))
    reasons = _metadata_reasons(products, reflection)
    lattice = declared_lattice
    origin, spacing = lattice.origin_xy_lod.tolist(), lattice.spacing_xy_lod.tolist()
    geometry_valid = all(normal_or_zero(v) for v in origin + spacing) and all(
        h > 0 for h in spacing
    )
    if not geometry_valid:
        reasons |= Reason.GRID_NONUNIFORM
    range_valid = all(1e-6 <= h <= 1e6 for h in spacing) and all(
        v == 0 or 1e-6 <= abs(v) <= 1e6 for v in origin
    )
    if not range_valid:
        reasons |= Reason.LAYOUT_UNSUPPORTED
    geometry_valid = geometry_valid and range_valid
    domain_valid = in_supported_range(domain)
    if not domain_valid:
        reasons |= Reason.METADATA_CONFLICT
    reflected = reflection.kind == "two_axis_pixel_flip_v1"
    if not 0 < float(reflection.tolerance) <= 1e-6:
        reasons |= Reason.SYMMETRY_UNSUPPORTED
    if reflection.kind not in ("none", "two_axis_pixel_flip_v1"):
        reasons |= Reason.LAYOUT_UNSUPPORTED | Reason.SYMMETRY_UNSUPPORTED
    if reflected and any(v != 0 for v in origin):
        reasons |= Reason.GRID_NONUNIFORM
    if not reflected and (bool(jnp.any(reflection.axes)) or mirror_checks is not None):
        reasons |= Reason.SYMMETRY_UNSUPPORTED
    sy, sx = lattice.shape_yx
    # Numerical buffers keep occupancy validation bounded by the delivery size.
    # Never allocate the declared rectangle when its size exceeds delivered rows.
    linear = indices[:, 1].astype(jnp.int64) * sx + indices[:, 0]
    in_range = bool(jnp.all((indices >= 0) & (indices < jnp.array([sx, sy]))))
    if in_range and rows == sx * sy:
        occupancy = jnp.bincount(linear, length=sx * sy)
        duplicate = bool(jnp.any(occupancy > 1))
        complete = bool(jnp.all(occupancy == 1))
    else:
        # Out-of-range pairs need lexicographic keys to avoid linear collisions.
        keys = (indices[:, 1].astype(jnp.int64) << 32) + indices[:, 0]
        ordered = jnp.sort(keys)
        duplicate = bool(jnp.any(ordered[1:] == ordered[:-1]))
        complete = False
    if duplicate:
        reasons |= Reason.GRID_DUPLICATE
    if not complete:
        reasons |= Reason.GRID_INCOMPLETE
    if min(sy, sx) < (17 if reflected else 32):
        reasons |= Reason.INCOMPLETE_STENCIL
    tau, residual = (), ()
    if rows and geometry_valid:
        uniform, tau, residual = _offset_evidence(
            offsets, lattice, indices, offset_dtype
        )
        if not uniform:
            reasons |= Reason.GRID_NONUNIFORM
        reasons |= _sampling_reasons(products, spacing)
    flat_nodes = nodes.reshape(-1)
    for start in range(0, flat_nodes.size, 65536):
        block = flat_nodes[start : start + 65536]
        integer = jnp.int32 if block.dtype == jnp.float32 else jnp.int64
        bits = jax.lax.bitcast_convert_type(block, integer)
        magnitude_mask = jnp.iinfo(integer).max
        negative = (bits < 0) & ((bits & magnitude_mask) != 0)
        if not bool(jnp.all(jnp.isfinite(block) & ~negative)):
            reasons |= Reason.DATA_INVALID
    allocations = estimate_memory(
        nodes.nbytes, products.pixels.shape_yx, Tiling(), 1, admission=True
    )
    if any(
        allocations[key] > getattr(memory, f"max_{key}_bytes")
        for key in ("resident", "output", "scratch", "total")
    ):
        reasons |= Reason.MEMORY_LIMIT
    if (
        geometry_valid
        and domain_valid
        and safe_query(domain, lattice, reflected) is None
    ):
        reasons |= Reason.INCOMPLETE_STENCIL
    receipt = Receipt(
        profile_id=profile.identity,
        execution_profile=f"jax-{jax.__version__}:{jax.default_backend()}:float64",
        offset_dtype=str(offset_dtype),
        provenance=provenance,
        offset_tolerance_xy=tau,
        canonicalization_residual_xy=residual,
        stored_bounds_xy=(
            tuple(origin),
            tuple(
                o + (n - 1) * h
                for o, n, h in zip(origin, (sx, sy), spacing, strict=True)
            ),
        ),
        served_geometry=(
            domain.kind,
            domain.boundary,
            tuple(map(tuple, domain.limits_xy_lod.tolist())),
            tuple(domain.radii_lod.tolist()),
        ),
        premise_evidence=tuple(
            zip(
                products.premises.values.tolist(),
                products.premises.evidence.tolist(),
                products.premises.source_ids,
                strict=True,
            )
        ),
        provenance_ids=(
            products.identity,
            products.optical_id,
            products.spectral_weight_id,
            reflection.declaration_id,
            reflection.argument_id,
            reflection.check_report_id,
        ),
        allocation_estimate=tuple(allocations.items()),
    )
    if reasons:
        return Admission(None, receipt, int(reasons))
    order = jnp.argsort(linear)
    cube = nodes[:, order].reshape(count, sy, sx, *products.pixels.shape_yx)
    if reflected:
        symmetry_reasons, passed, mirror, axes, exact = _symmetry_checks(
            cube, lattice, products, reflection, mirror_checks
        )
        reasons |= symmetry_reasons
        reflection = eqx.tree_at(lambda r: r.check_passed, reflection, passed)
        receipt = replace(
            receipt,
            mirror_local_peak=mirror,
            axis_relative=axes,
            axis_exact=exact,
            provenance_ids=(
                *receipt.provenance_ids,
                mirror_checks.provenance_id if mirror_checks else "",
            ),
        )
    if reasons:
        return Admission(None, receipt, int(reasons))
    identity = _digest(
        cube,
        offsets,
        indices,
        lattice,
        products,
        reflection,
        domain,
        profile,
        checks,
        mirror_checks,
        str(offset_dtype),
        provenance,
    )
    receipt = replace(receipt, identity=identity)
    grid = SampledGrid(cube, lattice, products, reflection, identity)
    if checks is not None:
        if checks.intensity.shape[1:] != products.pixels.shape_yx:
            raise ValueError("Check detector shape differs from samples")
        valid = (
            checks.product_index.size > 0
            and all(
                (checks.provenance_id, checks.reference_method_id, checks.target_id)
            )
            and bool(jnp.all(jnp.isfinite(checks.intensity) & (checks.intensity >= 0)))
            and bool(
                jnp.all(
                    jnp.isnan(checks.core_peak)
                    | (jnp.isfinite(checks.core_peak) & (checks.core_peak > 0))
                )
            )
        )
        if not valid:
            return Admission(None, receipt, int(Reason.DATA_INVALID))
        reader = Interpolator(grid, domain, profile, ReadPolicy(), Tiling(), memory)
        absolute, relative = [], []
        for index in range(checks.product_index.size):
            result = reader.read(
                checks.source_xy_lod[index], checks.product_index[index]
            )
            if not bool(result.accepted):
                return Admission(
                    None, receipt, int(result.reason_bits) | int(Reason.DATA_INVALID)
                )
            error = float(jnp.max(jnp.abs(result.intensity - checks.intensity[index])))
            absolute.append(error)
            relative.append(error / float(checks.core_peak[index]))
        empirical = max(absolute)
        receipt = replace(
            receipt,
            empirical_absolute=tuple(absolute),
            empirical_local_peak=tuple(relative),
            empirical_max_abs=empirical,
            provenance_ids=(
                *receipt.provenance_ids,
                checks.provenance_id,
                checks.reference_method_id,
                checks.target_id,
            ),
        )
        grid = SampledGrid(
            cube, lattice, products, reflection, identity, jnp.asarray(empirical), True
        )
    return Admission(grid, receipt, 0)
