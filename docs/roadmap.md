# Roadmap

The sinchole library grows by adding readers, products and evidence beside the version
0.1 interfaces rather than by changing them. Each stage below says what it
unlocks for a user and what stays fixed, so code written against version 0.1
keeps working. The order reflects dependencies, not dates.

## Version 0.1: sampled intensity grids (now)

Version 0.1 reads off-axis planet intensity images from a regular grid of
propagated PSFs, either a complete signed rectangle or a positive quadrant
with a declared and checked two-axis mirror symmetry. Admission checks the
geometry, the declared physical premises, the sampling margin and the memory
budget, and every read either returns an image with an unknown error bound or
refuses with a reason. Reads run in JAX on CPU, under `eqx.filter_jit` and
`jax.vmap`, and yippy can use sinchole as an optional backend for its
`Coronagraph`. One starter cell of a public LUVOIR-A APLC design has been
checked end to end ({doc}`validation`).

## Next: release and broader qualification

- **Release hardening.** Publish the first release to PyPI, test the lowest supported
  dependency versions in continuous integration, and keep the yippy backend
  optional.
- **More cells and designs.** Qualify additional cells of the APLC grid and
  other coronagraph families, such as hybrid Lyot and vortex designs, against
  independent propagations. The comparison method on {doc}`validation`
  carries over unchanged.
- **GPU execution.** The reader is fixed-shape JAX with arrays held as
  dynamic leaves, so it should run on a GPU without changes, but that has
  not been tested. Arithmetic, memory and speed need qualification there
  before a GPU result carries the same evidence as a CPU result.

## Then: wavelength and stellar products

- **Wavelength-resolved products.** A spectral reader that combines several
  monochromatic products, using designer-chosen wavelength nodes and
  Chebyshev polynomials in inverse wavelength (a basis that represents
  smooth functions of wavelength with few nodes). Spatial reads stay exactly
  as they are, and the spectral layer calls them.
- **Band means.** A delivered band-mean image stays one exact product. Later
  versions can integrate wavelength-resolved products over a band with an
  explicit quadrature error, but never split a band mean into channels.
- **Stellar products.** On-axis star images need integration over the
  stellar disk and pointing jitter, and sampling in stellar diameter. These
  arrive as separate readers with their own error targets, beside the
  off-axis reader.
- **Other layouts.** Sparse symmetric layouts that store only the useful
  annulus, radial grids and irregular grids each need their own reader and
  error model, behind the same abstract reader interface.

## Later: error certificates and field tables

- **Error certificates.** Version 0.1 reports every error bound as unknown
  and refuses every bounded request. A certificate replaces "unknown" with
  an audited conditional bound: a ledger of every error term (truncation,
  storage precision, propagation convergence, symmetry), each with its
  evidence. Bounded reads then return an image with a stated maximum error,
  or refuse.
- **Complex field tables.** Storing the complex focal-plane field instead of
  intensity allows coherent sums, wavefront-error derivatives and a linked
  stellar wavefront layer. Fields carry a phase convention (a gauge) that
  must be declared and checked, and are converted explicitly before any
  image is produced. Field tables are a separate type, not a subclass of the
  intensity grid, so intensity reads stay unchanged.
- **Scenes and consumers.** Weighted sums of many sources (disks, exozodi,
  multiple planets) with summed error terms, and direct use in
  exposure-time calculators and image simulators.

## What stays fixed

Every stage above adds modules, readers and receipt fields. The version 0.1
names and signatures (`admit_grid`, `Interpolator`, `read_batch`,
`require_read`, the reason bits and statuses) keep their meaning, existing
reads keep their numerical profile, and a missing input still refuses rather
than being filled in.
