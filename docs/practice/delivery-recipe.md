---
file_format: mystnb
kernelspec:
  name: python3
---

# How do you produce a delivery that admits the first time?

A designer who follows the physics checklist can still send files that are
refused for a bookkeeping reason, or worse, files that admit and are wrong.
Either outcome costs a propagation campaign, because the images cannot be
patched afterwards. The physics is on {doc}`designers` and the file contract
is on yippy's [schema page][schema]. This page is the order of operations
around them: plan the positions, propagate with a fixed frame and
normalization, choose the symmetry argument, check the files yourself, and
send.

The running example is the public starter request: 27 by 31 nodes at 0.38
$\lambda_0/D$, 16 interior checks, 20 mirror checks and a 128 by 128 frame
of 0.25 $\lambda_0/D$ pixels. The toy coronagraph of the theory pages stands
in wherever a step needs a measurement.

```{code-cell} python
:tags: [remove-cell]
import math

import equinox as eqx
import hwostyle
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import sinchole as sc
from sinchole import viz
import toy_coronagraph as tc

jax.config.update("jax_enable_x64", True)
hwostyle.use("dark")
```

```{admonition} Predict before reading on
Suppose a delivery has a consistent sign error in its detector axes, so that
every image, including the interior checks and the mirror checks, is rotated
by 180 degrees. Will sinchole admit it? Commit to yes or no, then open the
answer under step 2.
```

The prompt returns under step 2, once the detector frame has been fixed.

(recipe-positions)=
## Step 1: Which positions do you propagate?

Run the planner with the reference wavelength, the shortest contributing
wavelength, the illuminated entrance width in units of $D$ and the served
limits. Two limits give a box, one limit gives a circle. The call below
reproduces the starter, and `--check-layout cell` places the 16 interior
checks in the small qualification cell instead of over the radius.

```bash
yippy-sinc plan --lambda0 1e-6 --lambda-min 1e-6 --support 1 \
    --limit 4.18 5.7 --out coords
# spacing 0.38 lambda0/D, rho 0.7600, 27 x 31 nodes (837 training),
# 16 checks, 20 mirror checks
```

Three rules sit behind the output, and each can be checked by hand. The
spacing is the largest $h$ with $\rho=2(\lambda_0/\lambda_{\min})Lh\le0.76$,
rounded down to six decimals, using the larger support width. For $L=1$ at
$\lambda_0$ that is 0.38. The nodes per axis are $\lceil R/h\rceil+16$
because a read near the served limit reaches 16 nodes beyond it, which gives
$\lceil4.18/0.38\rceil+16=27$ and $\lceil5.7/0.38\rceil+16=31$. The planner
writes the training list, the check list and the mirror-check list, and the
lists are used unchanged.

The 16 interior checks are a default count, one in each of 16 equal shells of
the served radius with a seeded angle, off every node so that each is a true
between-node read. Sixteen is enough to see a spread and cheap to propagate,
and the same seed always gives the same list. The 8 mirror bases are the
nodes whose negative-position images are propagated directly.

```{list-table}
:header-rows: 1
:widths: 20 40 40

* - Base $(i_x, i_y)$
  - Role
  - Counterparts
* - (1, 1)
  - near the origin, where the occulter edge sits
  - 3
* - (1, $j_y$), ($j_x$, 1), ($j_x$, $j_y$)
  - beside and at the corner of the served cell
  - 3 each
* - (0, $j_y$), ($j_x$, 0)
  - on an axis
  - 1 each
* - ($n_x-1$, 1), (3, $n_y-1$)
  - the outermost stored column and row
  - 3 each
```

Here $j_x=\lceil R_x/h\rceil-1$ and $j_y=\lceil R_y/h\rceil-1$. A base with both
indices nonzero has three negative counterparts (flip $x$, flip $y$, both)
and a base on an axis has one, so the starter's six general bases and two
axis bases give $6\times3+2\times1=20$ mirror checks.

The two outer-edge bases matter because a read at a negative position near
the served edge uses the mirror images of the outermost stored nodes. A
symmetry that fails only in the far wing, where the stencil of an edge read
lives, would be invisible to checks near the origin. The validator adds a
note, without failing, when no mirror base reaches the largest stored index
on an axis.

(recipe-frame)=
## Step 2: Where does the star sit, and how do you know?

Every image has the same shape, pixel pitch and center, and the star is at
the center of the frame, which is the point $((n_x-1)/2,\,(n_y-1)/2)$ in
zero-based pixel coordinates (the manifest key `detector_center_xy`). For an
even dimension such as 128 that is between the two middle pixels, at 63.5.
For an odd dimension such as 129 it is on the middle pixel, at 64.0. Increasing
column follows positive source $x$ and increasing row follows positive source
$y$. The shape must be square, and `PIXSCALE` is in $\lambda_0/D$ per pixel
([header keywords][schema-fits]).

The frame size is a choice with one soft requirement. The farthest stored
starter node is at radius $\sqrt{9.88^2+11.4^2}=15.08\,\lambda_0/D$ and the
frame half-width is $64\times0.25=16$, so every planet core falls inside the
frame. A smaller frame is allowed, because the parser fixes no minimum, but
sources whose cores fall outside it must still return their in-frame wings.
Choose $N\ge2R_{\rm node}/\text{PIXSCALE}$ with $R_{\rm node}$ the farthest
node radius if downstream use needs every core.

The checks cannot see a global error in the frame, so add a fiducial test
of position. For a source at $(s,0)$ the centroid of the core, taken over
pixels within 1 $\lambda_0/D$ of the expected position, must sit at column
$c+s/\text{PIXSCALE}$ and row $c$, and a source at $(0,s)$ must sit at row
$c+s/\text{PIXSCALE}$. The second source catches a swapped $x$ and $y$. Use an
$s$ larger than the occulter radius so that a core exists to measure.

```{code-cell} python
:tags: [hide-input]
toy = tc.ToyCoronagraph()
center = (toy.n_pix - 1) / 2

def core_centroid(image, source):
    """Centroid of the pixels within 1 lambda0/D of the expected core position."""
    image = np.asarray(image)
    rows, cols = np.indices(image.shape)
    col0 = center + source[0] / toy.pixscale_lod
    row0 = center + source[1] / toy.pixscale_lod
    near = (cols - col0) ** 2 + (rows - row0) ** 2 <= (1.0 / toy.pixscale_lod) ** 2
    weight = image * near
    return (weight * cols).sum() / weight.sum(), (weight * rows).sum() / weight.sum()

s = 3.0
for source in ((s, 0.0), (0.0, s)):
    col, row = core_centroid(toy.psf(jnp.array(source)), source)
    print(f"source {source}: expected column {center + source[0] / toy.pixscale_lod:.2f},"
          f" row {center + source[1] / toy.pixscale_lod:.2f}; measured column {col:.2f},"
          f" row {row:.2f}")
```

On the toy the measured centroids land within 0.1 pixel of the expected
positions. Now the prompt. The cell below rotates every image of the toy
delivery, nodes and mirror checks together, and admits both versions.

```{code-cell} python
:tags: [hide-input]
good = tc.quadrant_delivery(toy)
rotated = dict(good)
rotated["intensity_nodes"] = good["intensity_nodes"][..., ::-1, ::-1]
mirrors = good["mirror_checks"]
rotated["mirror_checks"] = eqx.tree_at(
    lambda m: m.intensity, mirrors, mirrors.intensity[..., ::-1, ::-1])
for name, delivery in (("as propagated", good), ("rotated by 180 degrees", rotated)):
    admission = sc.admit_grid(**delivery)
    print(f"{name}: admitted={admission.admitted}, "
          f"mirror residuals {', '.join(f'{v:.1e}' for v in admission.receipt.mirror_local_peak)}")
node = 8  # training row of the node at (3.04, 0): index = iy * 31 + ix with ix = 8
for name, delivery in (("as propagated", good), ("rotated by 180 degrees", rotated)):
    peak_column = int(np.argmax(np.asarray(delivery["intensity_nodes"][0, node]).max(axis=0)))
    print(f"core of the (3.04, 0) node, {name}: brightest column {peak_column}")
print(f"expected column c + s / PIXSCALE = {center + 3.04 / toy.pixscale_lod:.2f}")
```

```{dropdown} Answer to the prediction
Yes, both versions are admitted, with mirror residuals of order $10^{-14}$ to
$10^{-16}$ in both cases, so no check in the receipt separates them. The
consistent 180 degree rotation commutes with both reversals, so a rotated
mirror check still compares like with like, and the interior reads still agree
with their rotated direct images. Admission tests that the images agree with
each other, and a consistent error agrees with itself. A delivery is correct
when it passes that self-check and also places the star where the sky says. The
fiducial test does: the core of the $(3.04,0)$ node is at column 60, next to
the expected 59.66, as propagated, and at column 35, on the opposite side of
the frame, after the rotation. Run the fiducial test on the delivered files
before sending, as part of step 6.
```

(recipe-pixel)=
## Step 3: What does one pixel hold?

The default is a pixel integral normalized to the photons that enter the
aperture: each stored value is the fraction of incident photons that land on
that pixel (`sample_kind: integral`, `normalization.kind: incident_photon`,
[pixel measure][schema-pixel] and [normalization][schema-norm]). A propagator
that follows the standard Fourier convention gives this directly, because
Parseval's relation makes the intensity over the whole focal plane integrate
to the pupil area in units of $D^2$. The fraction in a pixel is therefore the
intensity integrated over the pixel, in $(\lambda_0/D)^2$, divided by that
pupil area.

Most propagators return point samples at pixel centers. Multiplying a point
sample by the pixel area is a different number from the pixel integral, and
the difference cannot be repaired later, so integrate in the propagator by
supersampling. Propagate on a grid $s$ times finer in each axis and average
each $s\times s$ block, which is the mean over the pixel, then multiply by the
pixel area for the integral. The cell measures how fast this converges for a
planet core on the toy at $(3,0)$, with $s=8$ as the reference.

```{code-cell} python
:tags: [hide-input]
source = jnp.array([3.0, 0.0])
binned = {}
for factor in (1, 2, 4, 8):
    fine = tc.ToyCoronagraph(n_pix=toy.n_pix * factor, pixscale_lod=toy.pixscale_lod / factor)
    image = np.asarray(fine.psf(source))
    binned[factor] = image.reshape(toy.n_pix, factor, toy.n_pix, factor).mean(axis=(1, 3))
for factor in (1, 2, 4):
    gap = np.abs(binned[factor] - binned[8]).max() / binned[8].max()
    print(f"supersampling {factor}: largest pixel difference from s=8, "
          f"as a fraction of the peak: {gap:.1e}")
```

A point sample at the pixel center is off by 1.6 percent of the peak on this
core, and each doubling of $s$ cuts the difference by a factor of about four
to five, reaching $7\times10^{-4}$ at $s=4$. Choose $s$ so that the
difference is well below the $10^{-5}$ level of step 6, and confirm it with
the factor-two refinement that step 6 asks for.

If the propagator reports an unocculted-peak image instead, the other
accepted reference is `unocculted_peak`, with a positive `photon_conversion`
per plane that is the same for every source position. A conversion that
varies with position breaks the band limit
({doc}`../theory/band-limits`). Never normalize an image to its own sum or
peak. Point samples are a sinchole option, but the yippy format of version
0.1 accepts only `integral` and `mean`, so a delivery must be one of those.

The diameter convention also needs a decision before step 1. The unit
$\lambda_0/D$ and the width $L$ are both measured against $D$, so the
declared `diameter_convention` and the declared support must agree. Take a
regular hexagonal aperture of flat-to-flat width $F$. Its vertex-to-vertex
width is $2F/\sqrt3\approx1.155F$. With $D$ the circumscribed width, $L=1$ and
the spacing is 0.38. With $D=F$, inscribed, the illuminated width along the
vertex axis is $L=1.155$ and the spacing falls to
$0.76/(2\times1.155)=0.329$ at $\lambda_0$. The circumscribed choice is
conservative for the spacing, and either is acceptable if stated.

(recipe-band)=
## Step 4: How do you deliver a filter with tails?

Version 0.1 delivers wavelengths as monochromatic planes. The `planes` list
runs from the longest to the shortest wavelength, each plane has its own
cube, offsets, checks and mirror checks, and one `--product` is read at a
time ([planes][schema-planes]). The lattice and the pixel scale are single
values in $\lambda_0/D$ for the whole delivery, so every plane is propagated
at the same angular positions on the same angular frame, and the
$\lambda_0/D$ grid is not rescaled with wavelength. The yippy parser of version
0.1 requires every plane to be monochromatic, so a band-mean plane is not
supported and a band product is assembled from the planes after the read.

The sampling limit is set by the shortest contributing wavelength, because
$\sigma_{\max}=\lambda_0/\lambda_{\min}$ sets the bandwidth. A filter with
tails has no sharp shortest wavelength, so declare it as the wavelength below
which the weighted flux fraction is less than a stated $\epsilon$, and write
$\epsilon$ in the optical state beside the filter curve. The library fixes
no value of $\epsilon$. A component beyond the edge is read at $\rho$ above
0.76, and its error is at most its weight times the worst single-tone kernel
error at that $\rho$, which the cell measures.

```{code-cell} python
:tags: [hide-input]
for rho in (0.76, 0.80, 0.90, 1.00):
    worst = tc.kernel_error_worst(rho)
    print(f"rho = {rho:.2f}: worst single-tone error {worst:.1e}, "
          f"a 1e-3 flux tail contributes {1e-3 * worst:.0e}")
```

Compare that product of weight and error with the $10^{-5}$ interior level
of step 6. The printed products show that a tail carrying $10^{-3}$ of the
flux stays below the level at $\rho=0.8$ and rises above it by $\rho=0.9$,
and at $\rho=1$ the tail contributes its whole weight. The estimate is rough, so let the interior checks
decide. The per-plane `sampling_min_wavelength_m`
must not exceed the plane wavelength, and `campaign_sampling_min_m` records a
single minimum for the whole campaign.

(recipe-symmetry)=
## Step 5: Which symmetry argument does the design support?

The quadrant needs both intensity mirrors for the optical state actually
propagated. The arguments are on {doc}`../theory/quadrant-symmetry`. Two
geometric mirrors of every component give both directly. Conjugation,
$E(-x;-s)=E^*(x;s)$, needs a real pupil, a real stop and a real
centrosymmetric focal mask, and combined with one geometric mirror it gives
both. A complex mask or a wavefront aberration breaks conjugation. The table
applies this to the common families.

```{list-table}
:header-rows: 1
:widths: 20 45 35

* - Design family
  - Symmetry argument
  - Layout
* - APLC, real occulter, pupil with two mirrors
  - Two geometric mirrors (`component_reflections`), and conjugation as a
    second route
  - Quadrant: `two_axis_pixel_flip_v1`, nodes $\lceil R/h\rceil+16$ per axis
* - APLC, real occulter, pupil with one mirror (the public starter)
  - One mirror plus conjugation (`mirror_plus_conjugation`), valid only while
    the optics stay real
  - Quadrant, as above
* - HLC, complex focal mask
  - Conjugation fails. A quadrant needs two geometric mirrors of the pupil and
    a mask phase that is itself mirror symmetric
  - Quadrant if both hold, otherwise full grid
* - Vortex
  - Each single geometric mirror reverses the charge. Two pupil mirrors make
    the $+\ell$ and $-\ell$ intensities equal, but only for a real pupil and a
    real stop, because the equality goes through conjugation, which maps $+\ell$
    to $-\ell$. One pupil mirror maps the product to the opposite charge
  - Quadrant for a two-mirror pupil, full grid otherwise, checked per
    polarization channel
* - Single-mirror pupil with a complex mask, or a segmented or off-axis pupil
    with no mirror
  - None gives both mirrors. With a complex mask, conjugation fails even for
    the point inversion $I(-x;-s)=I(x;s)$, as the both-negative residual in the
    toy cell below shows. Version 0.1 uses neither
  - Full grid
```

The full-grid recipe replaces the quadrant steps. Set `layout` to
`full_rectangle` and `reflection` to null, deliver no mirror checks, and list
signed indices in the training file with the origin at $(0,0)$. A served box
$[-R,R)$ on each axis then needs indices from $-(\lceil R/h\rceil+15)$ to
$+(\lceil R/h\rceil+15)$, which is $2\lceil R/h\rceil+31$ nodes per axis, so
about four times the quadrant count: for the starter region that is
$53\times61=3233$ nodes against 837. The interior checks and everything in
steps 2 to 4 are unchanged.

Two caveats are open. The planner writes quadrant lists only, so a
full-grid training list is written by hand from the rule above. The
full-rectangle path is exercised by the yippy parser tests on a synthetic
delivery and has not been run on a real full-grid delivery. The node count
follows from the 16-node reach of the stencil and the parser's coverage test,
and `yippy-sinc validate` will confirm it.

A note on handedness: a vortex of charge $\ell$ has mask phase
$e^{i\ell\theta}$, and reflecting the system about an axis maps $\theta$ to
$-\theta$, so the reflected optical system carries charge $-\ell$. The
mirror image of a charge $\ell$ product is a charge $-\ell$ product. When the
pupil has both mirrors the two products have the same intensity and the
quadrant holds, and when the pupil has one mirror the geometric mirror maps
one handedness to the other and the check fails. The two circular
polarizations see opposite charges, so polarization channels are checked one
channel at a time, and an unpolarized sum can be symmetric while each channel
is not. Version 0.1 has no polarization axis in the schema, and a separate
directory per channel is the suggested layout, which is unconfirmed.

The claim that a complex mask breaks the quadrant is measured on the toy
coronagraph. The cell gives the toy's focal mask a phase of 0.8 radians that
grows as the square of the radius, a centrosymmetric but complex mask, and
admits a quadrant grid for each case. The toy pupil that is shifted along $x$
keeps one mirror and is the one-mirror case.

```{code-cell} python
:tags: [hide-input]
from dataclasses import dataclass

@dataclass(frozen=True)
class ComplexMaskToy(tc.ToyCoronagraph):
    """The toy with a complex focal mask: radial phase and an optional vortex charge."""

    mask_phase: float = 0.0
    charge: int = 0

    def field(self, source_xy):
        uu, vv = self.pupil_coords
        u = uu[0]
        stop = (uu**2 + vv**2 <= (self.lyot / 2) ** 2).astype(jnp.float64)
        n_occ = 40
        xi = tc._axis(n_occ, 2 * self.occulter_lod / n_occ)
        xx, yy = jnp.meshgrid(xi, xi)
        r2 = (xx**2 + yy**2) / self.occulter_lod**2
        disk = (r2 <= 1).astype(jnp.float64)
        mask = disk * jnp.exp(1j * self.mask_phase * r2)
        if self.charge:
            mask = disk * (1 - jnp.exp(1j * self.charge * jnp.arctan2(yy, xx)))
        du = 1.0 / self.n_pupil
        to_focal = jnp.exp(-2j * jnp.pi * jnp.outer(xi, u)) * du
        to_lyot = jnp.exp(2j * jnp.pi * jnp.outer(u, xi)) * (xi[1] - xi[0])
        to_detector = jnp.exp(-2j * jnp.pi * jnp.outer(self.detector_x, u)) * du
        tilted = self.pupil * jnp.exp(1j * self.phase_ramp(source_xy))
        focal = to_focal @ tilted @ to_focal.T
        lyot_field = tilted - to_lyot @ (mask * focal) @ to_lyot.T
        return to_detector @ (stop * lyot_field) @ to_detector.T

cases = {
    "one mirror, real mask": ComplexMaskToy(pupil_shift=(0.02, 0.0)),
    "one mirror, complex radial mask": ComplexMaskToy(pupil_shift=(0.02, 0.0), mask_phase=0.8),
    "two mirrors, complex radial mask": ComplexMaskToy(mask_phase=0.8),
    "two mirrors, vortex phase (charge 2)": ComplexMaskToy(charge=2),
    "one mirror, vortex phase (charge 2)": ComplexMaskToy(pupil_shift=(0.02, 0.0), charge=2),
}
receipts = {}
for name, case in cases.items():
    admission = sc.admit_grid(**tc.quadrant_delivery(case))
    receipts[name] = admission.receipt
    print(f"{name}: admitted={admission.admitted}, "
          f"reasons={sc.reason_names(admission.reason_bits)}")
    print("   mirror residuals (x-neg, y-neg, both-neg):",
          ", ".join(f"{v:.1e}" for v in admission.receipt.mirror_local_peak))
```

The one-mirror pupil with a real mask passes at rounding level, as the theory
predicts. With the complex mask the same pupil is refused with
`SYMMETRY_CHECK_FAILED`. The $x$-negative and both-negative residuals are
0.11 of the local peak and the $y$-negative residual, the pupil's own mirror,
is still $4\times10^{-16}$, so the failure shows which reversal conjugation
was supplying. The two-mirror pupil with the same complex mask passes,
because two geometric mirrors do not use conjugation.

The toy vortex is an azimuthal phase of charge 2 confined to the occulter
disk, so it is a test of the handedness argument and not a vortex design.
Behind the two-mirror pupil it passes. Behind the one-mirror pupil it is
refused, with the $y$-negative and both-negative residuals at 0.084. The
geometric mirror of that pupil is the one that reverses the charge, and the
$x$-negative check, which conjugation supplies, still passes at
$5\times10^{-16}$.

(recipe-check)=
## Step 6: How do you check the files before sending?

Install the yippy extra and run the validator on the delivery directory.
It runs the yippy integrity checks, then sinchole admission with 64-bit
arithmetic, and it exits with 0 on a pass, 1 on a failure and 2 on a usage
error.

```bash
pip install "yippy[sinc]"
yippy-sinc validate path/to/delivery
yippy-sinc validate path/to/delivery --product mono-000 --json
```

This example needs a delivery on disk.

```python notest
from yippy.sinc_tools import validate

report = validate("path/to/delivery")
print(report.summary())
assert report.passed
```

On the real starter the run passes in about 19 s with $\rho$ of 0.7190 and
0.7463, all 58 axis images exact, and an interior worst and median of
$1.44\times10^{-6}$ and $7.85\times10^{-7}$ of the local peak. The summary
lines mean the following.

```{list-table}
:header-rows: 1
:widths: 22 40 38

* - Line
  - What it says
  - If it is wrong
* - `integrity`
  - hashes, headers, shapes and the complete rectangle agree with the manifest
  - fix the file or the manifest named in the message, then recompute the hash
* - `reasons`
  - the sinchole refusal names, `none` when admitted
  - see the refusals below
* - `rho (x, y)`
  - occupancy per axis against the 0.76 limit
  - recompute the spacing from the shortest wavelength and the widest support
* - `interior checks`
  - worst and median residual of the read against direct propagation, as a
    fraction of the local peak
  - above $10^{-5}$ means a frame, normalization or convergence fault
* - `mirror checks`, `axis images`
  - negative-position images against the flipped stored images, and each
    stored axis image against its own flip, limit $10^{-6}$
  - a frame error or the wrong symmetry argument, see step 5
```

The recommended interior level is $10^{-5}$ of the local peak. The starter's
16 checks read at $1.44\times10^{-6}$ at worst, so the level leaves a margin
of about seven. It is a recommendation from one clean delivery, not a bound
on reads between the checks, and a tighter level of $10^{-6}$ would reject
that clean delivery. The mirror and axis limit of $10^{-6}$ is fixed by the
format.

The refusals that a first delivery meets, and the repair for each:

- `KERNEL_MARGIN_FAIL`: $\rho$ is above 0.76. The shortest wavelength is
  shorter than declared, or the support is wider than declared. Replan.
- `METADATA_MISSING`: a premise, the support evidence or the sampling
  wavelength is unset. Evidence code 2 is not accepted in version 0.1, so
  declare each premise with code 1 and a `source_id`.
- `SYMMETRY_CHECK_FAILED`: a mirror residual is above $10^{-6}$. Test the
  fiducial positions and the frame center, then reconsider the symmetry
  argument and fall back to the full grid.
- `GRID_INCOMPLETE` or a hole in the rectangle: a node is missing. Deliver the
  whole rectangle, including nodes inside the occulter.
- A missing local-peak reference: each interior check needs a positive
  `local_peak_reference`, the largest direct pixel within 1 $\lambda_0/D$ of
  the source.

Numerical convergence and storage precision are reported alongside the
validator output, and are not part of it. Rerun a handful of the interior
check positions at successive factor-two refinements of the propagation
sampling and the pixel integration, and report the change between levels as a
fraction of the finer image's local peak. Compare float32 storage with float64
at identical settings and positions, saving the float64 versions of the
checks. These are comparison levels to report, not gates, and a comparison
that was not run is reported as not run.

The validator does not run the fiducial test of step 2, and it cannot detect
a consistent error. Run that test yourself on the delivered cube.

## What do you send?

Send the directory the validator passed, unchanged: the manifest, the three
position lists, the training cube and offsets, the check and mirror-check
cubes, the mirror report and the refinement and storage comparisons the
request asks for. The layout and every key are on the [schema page][schema].
Two further items belong with the files.

- The stellar and sky products, in the legacy YIP layout, in the directory
  named by `legacy_package_path`. `Coronagraph(path, interpolation="sinc")`
  needs them, so a delivery meant for direct use must include them. The
  starter holds none, and that is why it is read through the router.
- The validator summary, any failed checks reported as failed, and the
  fiducial-test centroids.

## What this page established

A delivery is planned with the 0.76 rule, propagated on a fixed frame with
the star at the center, normalized to incident photons as a pixel integral,
and sent with a symmetry argument that fits the design. The self-check
catches inconsistent files and the fiducial test catches consistent errors,
and the toy showed that a complex mask breaks the quadrant where a real one
does not. The last page shows the measured evidence for the whole chain.

Next: {doc}`../validation`.

[schema]: https://yippy.readthedocs.io/en/latest/sinc-delivery.html
[schema-fits]: https://yippy.readthedocs.io/en/latest/sinc-delivery.html#header-keywords
[schema-pixel]: https://yippy.readthedocs.io/en/latest/sinc-delivery.html#pixel-measure
[schema-norm]: https://yippy.readthedocs.io/en/latest/sinc-delivery.html#normalization
[schema-planes]: https://yippy.readthedocs.io/en/latest/sinc-delivery.html#planes
