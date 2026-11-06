# What each declaration claims

A delivered grid is only readable between its nodes if a list of physical
statements about it is true, and most of those statements cannot be checked
from the images alone. The sinchole library therefore asks for them as explicit
declarations, keeps each with its provenance in the receipt, and refuses
when one is missing rather than assuming it. This page goes through the
declarations in the order `admit_grid` takes them, saying what each claims
and why the reader needs it. The toy coronagraph's versions are in
`docs/toy_coronagraph.py`.

## The samples and their positions

`intensity_nodes`
: The images, a float32 or float64 array of shape `[P, N, Py, Px]`: products
  (for example wavelengths), source nodes, detector rows, detector columns.
  Values must be finite and non-negative. A negative zero is allowed, but a
  negative subnormal is not. Storage precision is kept as delivered.

`offsets_xy_lod`
: The source position of every node, shape `[N, 2]`, $x$ then $y$, in
  reference $\lambda_0/D$. These are checked against the lattice to within
  four units in the last place of their original precision, and they are never
  used to estimate the spacing.

`declared_lattice` (`Lattice`)
: The authoritative origin, spacing and node count per axis. A designer
  states the spacing they propagated at, and sinchole never fits it from rounded
  endpoints, because a fitted spacing a few parts in $10^{12}$ off moves the
  stencil of distant nodes.

`indices_xy`
: The integer node index of every row of `intensity_nodes`, rebased so the
  smallest is zero. Rows may arrive in any order. Admission sorts them,
  rejects duplicates and requires the rectangle to be complete.

## The physical product (`Products`)

`Frame`
: The reference wavelength $\lambda_0$ and diameter $D$ that define the unit
  $\lambda_0/D$, and which diameter convention $D$ refers to.

`PixelGrid`
: Detector shape, center in pixels, pixel basis in $\lambda_0/D$ (square and
  axis-aligned for quadrant grids) and what a pixel value measures:
  `pixel_integral` (photons collected by the pixel), `pixel_mean_density`
  (the mean over the pixel, per unit area) or `point_density` (a sample at
  the pixel center). The reader returns values in the same measure.

`Normalization`
: What one unit of intensity means: the reference (for example incident
  photons at a declared plane), a conversion per product, an optional
  uncertainty and the factors included. Conversions must be the same for
  every source position, since a source-dependent factor breaks the band
  limit ({doc}`../theory/band-limits`).

Wavelengths
: `kind` (`monochromatic` or `band_mean`), the product wavelengths and band
  limits, the shortest wavelength that contributes to each product
  (`sampling_min_nm`) and an optional campaign minimum. The sampling test
  uses the shortest wavelength, because it sets $\sigma_{\max}$ and therefore
  the bandwidth.

Support
: The full illuminated entrance width along $x$ and $y$ in units of $D$
  (`support_extent_xy`), with evidence codes and source identifiers. Together
  with the shortest wavelength this fixes $B=\sigma_{\max}L$ per axis, and
  admission requires $\rho=2Bh\le0.76$ on both axes.

`Premises`
: Six ordered statements the band-limit argument depends on: the source
  enters as a paraxial tilt at the entrance pupil, the optics are fixed and
  linear, the detector registration is fixed, the pixel response is fixed,
  the normalization does not depend on the source and the spectral weights
  do not depend on the source. Each carries an evidence code (designer
  declared or externally established) and a source identifier. A premise
  with unknown evidence refuses admission with `METADATA_MISSING`, and a
  premise declared false refuses with the matching reason. The reader never
  assumes one.

Identities
: `optical_id`, `spectral_weight_id` and `identity` name the optical state,
  the spectral weighting and the product, and are hashed into the receipt.

## Symmetry, permission and memory

`Reflection`
: `none` for a full signed grid, or `two_axis_pixel_flip_v1` for a quadrant,
  with the declaration, the argument that justifies it, its evidence code,
  the check report and an optional stricter tolerance. Admission computes the
  checks itself, and the `check_passed` field is ignored
  ({doc}`../theory/quadrant-symmetry`).

`MirrorCheckSet`
: For a quadrant grid, independent images at negative positions with their
  signed node indices: an $x$-negative, a $y$-negative and a both-negative
  counterpart for each product.

`CheckSet` (optional)
: Independent images at interior positions between nodes. Admission reads the
  grid at each and records the residual relative to the local peak as
  empirical evidence, never as a bound.

`Domain`
: Where reads are permitted: a box or an origin-centered annulus, closed or
  lower-closed and upper-open. A read is served only inside the domain and
  where its whole stencil exists.

`MemoryBudget`
: Ceilings, in bytes, for resident samples, admission staging, outputs and
  read scratch. Admission refuses with `MEMORY_LIMIT` before allocating if an
  estimate exceeds a ceiling.

`provenance` (keyword)
: Optional `(key, text)` pairs, such as the full text of the symmetry
  argument or a normalization definition, kept verbatim in the receipt and
  hashed into its identity.
