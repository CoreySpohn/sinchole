# Numerical contract

This page states exactly what a version 0.1 read computes and which inputs
admission accepts. The theory pages explain why, and this page is the
specification a test or a downstream library can rely on.

## Kernel and stencil

The kernel profile is `kaiser_w16_beta12_raw_v1`: the weight of a tap at
signed offset $t$ samples from the query is

$$
w(t)=\operatorname{sinc}(t)\,\frac{I_0\!\left(12\sqrt{1-(t/16)^2}\right)}{I_0(12)}
\quad\text{for } |t|<16,\qquad w(t)=0 \text{ otherwise},
$$

where $\operatorname{sinc}(t)=\sin(\pi t)/(\pi t)$ and $I_0$ is the modified
Bessel function of order zero. The weights are raw: they are not normalized
to sum to one, and no DC, flux or photon-unit normalization is applied.

For a
query $s$ on an axis with origin $o$ and spacing $h$, the stencil is the 32
signed node indices `floor(t) - 15` through `floor(t) + 16` with
$t=(s-o)/h$, and the two axes combine as a separable product of 1024 taps per
pixel. Every tap, including the zero-weight endpoints, must be present. There
is no exact-node shortcut, wrapping, zero filling, shortened stencil or
detector shift. Reads keep signed values, including small negative
excursions.

## Floor rule

The floor rule is `exact_quotient_floor_v1`. Subnormal queries flush to
signed zero with integer bit operations, the difference $d=s-o$ is formed
once, and the IEEE remainder $r=\operatorname{fmod}(d,h)$, which is exact,
determines the floor of the exact real quotient: $n=\operatorname{round}((d-r)/h)$,
minus one when $r\ne0$ and $d<0$. Kernel phases use ordinary float64
division. The reason this matters is on {doc}`stencil-floor`.

## Supported numeric range

Each positive lattice spacing and each nonzero origin component must have
magnitude in $[10^{-6},10^{6}]$ (a zero origin is allowed), box limits must
have magnitude at most $10^6$, and annulus radii must lie in
$[10^{-6},10^{6}]$, with zero allowed only for the inner radius. Admission
refuses out-of-range lattice metadata with `LAYOUT_UNSUPPORTED` and
out-of-range domain metadata with `METADATA_CONFLICT`.

Inside this range the floor rule is exact. After the subnormal flush, a
nonzero difference $s-o$ and a nonzero remainder are normal numbers:
cancellation near a nonzero origin and remainders at or above the minimum
spacing have binary resolution of at least $2^{-73}$, and below the spacing
the remainder is the difference itself. Permitted quotients have magnitude at
most $2\times10^{12}$, below $2^{53}$, so recovering the integer and
correcting negative values are exact. Squares of supported radii and
boundary-scale coordinates neither underflow nor overflow. Queries far
outside the lattice fail coverage before any gathering.

## Permission and coverage

A read is permitted when the original signed query lies in the declared
`Domain` (a box or an origin-centered annulus, closed or lower-closed and
upper-open, never expanded by a tolerance) and every stencil node exists. A
safe point for compiled fallbacks is chosen from coverage endpoints, box
corners and centers, and annulus corner rays, side intersections and axis
crossings, using the same membership and floor rules as reads. An exotic
intersection that the finite candidate list cannot resolve refuses with
`INCOMPLETE_STENCIL` rather than expanding the domain.

## Quadrant grids

A quadrant grid has origin $(0,0)$, at least 17 nodes per axis, an
axis-aligned detector centered at `((nx - 1)/2, (ny - 1)/2)`, the reflection
kind `two_axis_pixel_flip_v1` and a `MirrorCheckSet` with x-negative,
y-negative and both-negative counterparts for each product. Negative taps
read the stored image at the absolute index, with detector columns reversed
for a negative x index and rows reversed for a negative y index.

Admission computes every mirror residual and every stored-axis invariance
check itself, and caller-supplied `check_passed` flags carry no authority.

The versioned relative tolerance is $10^{-6}$ of the local peak (the largest
direct pixel value among pixel centers within radius 1 $\lambda_0/D$ of the
source). `Reflection.tolerance` may be stricter, $0<\text{tolerance}\le10^{-6}$.
A looser tolerance refuses with `SYMMETRY_UNSUPPORTED` before any comparison.
An axis image passes if it is exactly invariant, or if its relative residual
against a positive local reference is within tolerance, and a nonzero discrepancy
without a positive local reference refuses. The receipt records
`axis_exact` separately and stores NaN in `axis_relative` where no positive
reference exists. Full grids require reflection kind `none` and no mirror
checks.

## Sampling admission

For every product and axis, with reference wavelength $\lambda_0$, declared
minimum sampling wavelength $\lambda_{\min}$ (and the optional campaign
minimum), declared support width $L$ in units of $D$ and spacing $h$ in
$\lambda_0/D$, admission computes $\rho=2(\lambda_0/\lambda_{\min})Lh$ in
binary64. $\rho\ge1$ adds `NYQUIST_FAIL` and $\rho>0.76$ adds
`KERNEL_MARGIN_FAIL`. Neither test implies accuracy at any position.

## Precision

Enable JAX 64-bit mode before creating arrays or readers. Samples stay in
their delivered float32 or float64 storage, and gathered pixels, weights and
contractions use float64. Without 64-bit mode, construction refuses with
`PRECISION_UNAVAILABLE`. Adapters that converted float32 coordinates to
float64 angular units pass `offset_original_dtype=jnp.float32`, and the
four-ULP coordinate allowance then uses `nextafter` in that original dtype at
the converted scale.

## Compilation, identity and provenance

`read_batch(reader, positions, product_indices)` takes float64 `[Q, 2]`
positions and int32 `[Q]` product indices and maps serial chunks. Static
metadata stays unbatched, and an empty batch performs no interpolation.
`eqx.filter_jit` and an outer `jax.vmap` are supported, but an outer mapped
dimension needs the caller's own memory accounting.

A readmitted product has
a content-bound static `receipt_id` and compiles once for that identity, while
replacing array leaves of the same shape without readmission does not
recompile. Readmit modified samples before reuse so that provenance and
empirical evidence stay valid. `provenance=(("key", "text"), ...)` pairs are
kept verbatim in the receipt and hashed into its identity, and they are preserved,
not validated.

## Memory accounting

`MemoryBudget` ceilings cover native resident samples, admission staging and
reordering, output leaves and estimated tiled scratch. Admission validates
completeness and duplicates with numerical linear indices and occupancy
counts, with no source-sized Python containers, and its estimate includes
coordinate and index casts, residual arrays, linear indices, occupancy,
sorting and the canonical reorder. These are allocation estimates, not
measured compiler workspace or whole-process peaks. The library does no file
I/O, spectral interpolation, fallback interpolation or certificate
evaluation.
