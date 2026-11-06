# sinchole

`sinchole` reads sampled intensity images with a raw, 32-tap-per-axis
Kaiser-sinc kernel in JAX and Equinox. It admits complete signed rectangles
or zero-anchored positive quadrants with declared and independently checked
two-axis detector reflection.

Admission verifies geometry, physical premises, declared bandwidth, sampling
margins and allocation estimates. It does **not** certify optical accuracy.
Sampling reads retain the delivered units and signed excursions, with unknown
error bounds (`+inf`). Independent check residuals are empirical diagnostics.
Bounded-policy reads always refuse in this version.

Install with `pip install sinchole`. Enable JAX x64 before creating arrays or
readers. Samples remain float32 or float64; gathered pixels, weights and
contractions use float64. No weight, DC, flux or photon-unit normalization is
applied by the reader.

## Minimal example

This synthetic constant field supplies explicit physical declarations. Real
products must supply their own provenance and justified support extents.

```python
import jax
import jax.numpy as jnp
import sinchole as sc

jax.config.update("jax_enable_x64", True)

pixels = sc.PixelGrid(
    shape_yx=(2, 2), center_xy_pix=jnp.array([0.5, 0.5]),
    basis_xy_lod=jnp.eye(2) * 0.25, measure="pixel_integral",
)
products = sc.Products(
    frame=sc.Frame(jnp.array(500.), jnp.array(6.), "entrance_diameter"),
    pixels=pixels,
    normalization=sc.Normalization(
        "incident_photon", jnp.array([1.]), jnp.array([jnp.nan]), (), "example-units"
    ),
    kind="monochromatic", wavelength_nm=jnp.array([500.]),
    limits_nm=jnp.array([[500., 500.]]), sampling_min_nm=jnp.array([500.]),
    campaign_sampling_min_nm=jnp.array(jnp.nan),
    support_extent_xy=jnp.ones((1, 2)),
    support_evidence=jnp.array([1], dtype=jnp.int32),
    support_source_ids=("constant-field-support",),
    premises=sc.Premises(
        jnp.ones(6, dtype=bool), jnp.ones(6, dtype=jnp.int32),
        ("analytic-constant-field",) * 6,
    ),
    optical_id="example", spectral_weight_id="mono", identity="example-product",
)
lattice = sc.Lattice(jnp.array([-5., -5.]), jnp.array([0.25, 0.25]), (41, 41))
iy, ix = jnp.indices(lattice.shape_yx, dtype=jnp.int32)
indices = jnp.stack((ix.ravel(), iy.ravel()), axis=1)
offsets = lattice.origin_xy_lod + indices * lattice.spacing_xy_lod
reflection = sc.Reflection(
    "none", jnp.array([False, False]), "", "", jnp.int32(0), "", jnp.array([False])
)
domain = sc.Domain("box", "closed", jnp.array([[-1., -1.], [1., 1.]]), jnp.zeros(2))
memory = sc.MemoryBudget(1_000_000, 1_000_000, 2_000_000, 4_000_000)
admission = sc.admit_grid(
    jnp.ones((1, 41 * 41, 2, 2)), offsets,
    declared_lattice=lattice, indices_xy=indices, products=products,
    reflection=reflection, domain=domain, memory=memory,
)
reader = sc.Interpolator(
    admission.require(), domain, sc.KernelProfile(), sc.ReadPolicy(),
    sc.Tiling(pixel_tile=4), memory,
)
result = sc.require_read(reader.read(jnp.array([0.13, -0.21]), jnp.int32(0)))
assert bool(result.accepted)
assert not bool(result.bound_available)
```

`Admission.require()` raises `AdmissionRefused` on rejection, with the receipt
and accumulated reasons attached. `MemoryLimit` and `PrecisionUnavailable` are
specialized admission exceptions. Compiled queries return per-query statuses
and NaN images on refusal; `require_read()` is an eager convenience that raises
`ReadRefused` with refused indices, statuses and reasons.

`read_batch(reader, positions, product_indices)` accepts float64 `[Q,2]`
positions and int32 `[Q]` product indices. It maps serial chunks and retains
static metadata without batching it. Empty batches perform no interpolation.
`eqx.filter_jit` and outer `jax.vmap` are supported; outer mapped dimensions
require caller memory accounting. Re-admit modified samples before public
reuse so provenance and empirical evidence remain valid. Same-shape array leaf
replacement without readmission does not recompile. A readmitted product has a
content-bound static `receipt_id` and compiles once for that identity; changing
its content or provenance creates a new identity and compilation. Strings are
static metadata.

## Numerical contract

The profile combines `kaiser_w16_beta12_raw_v1` with
`exact_quotient_floor_v1`. Queries flush subnormals to signed zero with integer
bit operations. After one rounded subtraction `d = q - origin`, an IEEE
remainder determines the exact-real quotient floor. The signed stencil is
`floor - 15` through `floor + 16`; kernel phases use ordinary float64 division.
All taps, including zero-weight endpoints, must be present. Permission is
checked at the original signed query, with no expanded boundaries.

The supported metadata range is explicit: each positive lattice spacing and
nonzero origin component has magnitude in `[1e-6, 1e6]` (zero origins are
allowed); box limit magnitudes are at most `1e6`; annulus radii lie in
`[1e-6, 1e6]`, with zero allowed only for the inner radius. Admission refuses
out-of-range lattice metadata with `LAYOUT_UNSUPPORTED` and out-of-range domain
metadata with `METADATA_CONFLICT` because the declared permission region cannot
be represented by this numerical profile.

After query subnormal flushing, nonzero `q - origin` and `fmod` remainders are
normal: cancellation near a nonzero origin and remainders at or above the
minimum spacing have binary resolution at least `2**-73`; below the spacing,
the remainder is the normal difference itself. Permitted quotient magnitudes
are at most `2e12`, below `2**53`, so integer recovery and negative-side floor
correction are exact. Squares of supported radii and boundary-scale coordinates
neither underflow nor overflow. Tiny coordinate squares can round to zero but
cannot change the rounded comparison with these radius limits. Queries far
outside the lattice fail coverage before gathering. The floor algorithm itself
is unchanged and does not promise exact results outside the admitted range.

Safe-point selection tests coverage endpoints, box corners/centers and annulus
corner rays, side intersections and axis crossings using the same membership
and floor rules as reads. The finite candidate list may refuse an unresolved
exotic intersection with `INCOMPLETE_STENCIL`; it never expands the domain.

Quadrants require origin `(0,0)`, at least 17 nodes per axis, an axis-aligned
detector centered at `((nx-1)/2,(ny-1)/2)`, and a `MirrorCheckSet`. Mirror checks
contain signed coordinates, signed int32 node indices, independent images,
product indices and a provenance ID. Each product needs x-negative,
y-negative and both-negative counterparts. The library computes their
local-peak residuals and every stored-axis invariance check. The versioned
relative tolerance is `1e-6`; `Reflection.tolerance` may be stricter, but must
satisfy `0 < tolerance <= 1e-6`. A larger tolerance refuses with
`SYMMETRY_UNSUPPORTED` before comparison. Caller-provided
`check_passed` flags have no authority. Local peaks include pixel centers
within radius 1 in reference angular units. An exactly invariant stored-axis
image passes even if that source's peak region lies outside the detector;
a nonzero discrepancy without a positive local reference refuses. The host
receipt records `axis_exact` independently and uses NaN in `axis_relative`
where no positive local reference exists. An axis image passes if exactly
invariant OR if its positive-reference relative residual is within tolerance.
Full grids
require reflection kind `none` and no mirror checks.

Adapters may pass `offset_original_dtype=jnp.float32` when original float32
coordinates were converted to float64 angular units. The four-ULP allowance
uses `nextafter` in that original dtype at the converted scale; converted values
are preserved, and `Receipt.offset_dtype` records the original precision.
Omitting the keyword uses the supplied offset array's dtype. Adapters may also
pass `provenance=(("key", "full text"), ...)`: ordered pairs, including duplicate
keys, are retained verbatim in the host receipt and hashed into its identity.
This is a preservation channel, not validation of the supplied physical claims.

Memory budgets cover native resident samples, admission staging/reorder,
output leaves and estimated tiled scratch. Admission uses numerical linear
indices and occupancy counts, with no source-sized Python tuple/set storage.
The estimate includes coordinate/index casts, residual arrays, linear indices,
occupancy, sorting/permutation and canonical reorder. Source count is bounded
conservatively from sample bytes assuming one float32 product, independently
of detector pixel count. Hashing and sample validation use bounded blocks. They are allocation estimates, not
measured compiler-workspace or whole-process peak-memory guarantees. The
library does no file I/O, spectral interpolation, fallback interpolation or
certificate evaluation.

## License

MIT. See [LICENSE](LICENSE).
