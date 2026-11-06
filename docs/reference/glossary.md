# Glossary

Terms as sinchole uses them, each with the page that develops it.

admission
: The one-time check of a grid before any read: geometry, declared physical
  premises, sampling margin, symmetry evidence and memory. It returns an
  `Admission` holding either a usable grid or the reasons for refusal.
  See {doc}`../practice/admission`.

aliasing
: A band-limited signal sampled above the Nyquist limit looks like a slower
  one that passes through the same samples. See
  {doc}`../foundations/sinc-interpolation`.

APLC
: Apodized pupil Lyot coronagraph: an apodizing mask in the pupil, a hard
  occulting disk in the focal plane and a Lyot stop. The LUVOIR-A APLC is the
  public design used on {doc}`../validation`.

band-limited
: Containing no spatial frequency above some limit. A coronagraph's planet
  image, as a function of source position, contains no frequency above
  $B=\sigma_{\max}L$ cycles per $\lambda_0/D$.
  See {doc}`../theory/band-limits`.

bandwidth ($B$)
: The highest spatial frequency in the intensity at a fixed pixel as the
  source moves, in cycles per $\lambda_0/D$.

blend interpolation
: The existing interpolation in yippy of tabulated PSFs, which combines nearby
  delivered images. On the physicaloptix starter delivery its worst error is
  0.19 of the local peak. See {doc}`../validation`.

centrosymmetric
: Unchanged by inversion through the center, $f(-\mathbf{x})=f(\mathbf{x})$.
  A circular occulting disk is centrosymmetric.

domain
: The `Domain` declaration: the region of source positions where reads are
  permitted, a box or an origin-centered annulus.

independent check
: A PSF propagated directly at a position between grid nodes and delivered
  separately, so a read can be compared with it. Checks are evidence about
  the delivered model at those positions, not a bound elsewhere.

Kaiser window
: A bell-shaped taper built from the modified Bessel function $I_0$ with a
  shape parameter $\beta$. The kernel multiplies the sinc by it so the
  weights fall smoothly to zero at 16 samples. See {doc}`../theory/kernel`.

kernel
: The weight function $w(t)$ that turns nearby samples into a value between
  them. The sinchole kernel uses a Kaiser-windowed sinc with half-width 16
  samples and shape parameter $\beta=12$. See {doc}`../theory/kernel`.

lattice
: The regular grid of source positions where PSFs were propagated, given by
  an origin, a spacing per axis and a node count per axis.

local peak
: The largest directly propagated pixel value among pixel centers within
  1 $\lambda_0/D$ of the source. Errors are quoted relative to it, so a
  number such as $10^{-6}$ means one millionth of the planet's own brightest
  pixel.

margin
: The largest oversampling factor the kernel is admitted at, $\rho=0.76$.
  Past it the kernel error rises by orders of magnitude, and admission
  refuses with `KERNEL_MARGIN_FAIL`. See {doc}`../theory/kernel`.

mirror check
: An independent propagation at a negative source position, compared with
  the flipped stored image at the mirrored positive node. Admission requires
  them for quadrant grids. See {doc}`../theory/quadrant-symmetry`.

node
: One lattice position with its delivered PSF.

Nyquist limit
: Sampling at exactly two samples per cycle of the highest frequency
  ($\rho=1$). Below it, samples determine a band-limited function between
  them. Above it, different functions share the same samples.

oversampling factor ($\rho$)
: $\rho=2Bh$, the fraction of the Nyquist limit used. $\rho=0.5$ means four
  samples per cycle of the highest frequency. The sinchole reader admits grids with
  $\rho\le0.76$ on every axis.

premise
: One of six physical statements the band-limit argument needs (for example
  fixed linear optics), declared with its evidence. See
  {doc}`../practice/declarations`.

product
: One delivered image set with its own wavelength (or band) and
  normalization, the leading axis of the sample array.

quadrant grid
: A grid that stores only non-negative source positions, anchored at the
  origin, and reads negative positions through the declared two-axis mirror
  symmetry.

read
: Reconstruction of the PSF at one source position. A read returns an
  image and a status, and a refused read returns NaN images and its reasons.

receipt
: The host-side record of an admission: identity, provenance, served
  region, symmetry and check residuals, allocation estimates. Readmitting
  changed data gives a new receipt identity.

refusal
: A decision not to return a value, recorded as reason bits. The sinchole
  reader refuses rather than filling in a missing input or guessing a premise.
  See {doc}`../practice/admission`.

separable
: A two-dimensional weight that is the product $w(t_x)\,w(t_y)$ of two
  one-dimensional weights, so a read adds $32\times32=1024$ taps. See
  {doc}`../theory/kernel`.

served region
: Source positions where a read is permitted: inside the declared domain and
  with all 32 by 32 stencil nodes present.

spacing rule
: The chain from support width $L$ to bandwidth $B=\sigma_{\max}L$, spacing
  $h\le0.76/(2B)$, node count $\lceil R/h\rceil+16$ per axis and the
  delivered grid. See {doc}`../theory/band-limits`.

starter grid
: The public LUVOIR-A APLC delivery of 27 by 31 nodes at spacing 0.38
  $\lambda_0/D$, serving the cell $3.8\le s_x<4.18$, $5.32\le s_y<5.70$.
  See {doc}`../validation`.

stencil
: The 32 nodes per axis around a query whose samples a read combines, from
  `floor(t) - 15` through `floor(t) + 16` with $t=(s-o)/h$.
  See {doc}`stencil-floor`.

taps
: The individual nodes of a stencil, each with its weight. A two-dimensional
  read adds 1024 of them. See {doc}`../theory/kernel`.

ULP
: Unit in the last place: the spacing between adjacent floating-point
  numbers at a given magnitude, about $2.2\times10^{-16}$ of the value in
  float64 and $1.2\times10^{-7}$ in float32.

unknown bound
: Version 0.1 reports every error bound as `+inf`. Reads in sampling mode
  still return images, and bounded requests refuse.

Next: {doc}`../viz`.
