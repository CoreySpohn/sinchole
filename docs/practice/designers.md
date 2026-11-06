---
file_format: mystnb
kernelspec:
  name: python3
---

# Delivering a grid: a checklist for designers

This page is for the coronagraph designer who propagates the PSFs. A
delivery that follows it can be admitted by sinchole and read through yippy
without any change on the receiving side, and a delivery that misses a step
is refused with a reason rather than read incorrectly. The physics behind
each step is on the theory pages, and the file layout is on {doc}`yippy`.

```{code-cell} python
:tags: [remove-cell]
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

## The checklist

1. **Choose the spacing from the optics.** With the full illuminated entrance
   widths $L_x$, $L_y$ (in units of $D$) and the shortest contributing
   wavelength $\lambda_{\min}$, use $h_j\le0.76\,\lambda_{\min}/(2\lambda_0L_j)$
   in $\lambda_0/D$, which keeps the oversampling factor $\rho$ at or below
   0.76 ({doc}`../foundations/sinc-interpolation` explains the number and
   {doc}`../theory/band-limits` the formula). For $L=1$ at $\lambda_0$ this
   is 0.38. Declare the
   widths and $\lambda_{\min}$, and do not round the spacing after propagating.
2. **Use the supplied integer indices.** Source positions are
   $(h\,i_x, h\,i_y)$ with integer indices, starting exactly at the origin
   for a quadrant. Keep each image associated with its integers and report
   any discrepancy in actual propagated positions.
3. **Deliver the complete rectangle.** A quadrant serving reads up to, but not
   including, $R$ along an axis needs $\lceil R/h\rceil+16$ nodes, indices 0
   through $\lceil R/h\rceil+15$, including
   positions inside the occulter or outside the useful region, because reads
   near the edge of the served region reach 16 nodes beyond it.
4. **Fix the detector frame.** Same shape, pixel pitch and center for every
   image. For a quadrant, the center at `((nx - 1)/2, (ny - 1)/2)` (between
   the middle pixels, not on one), with increasing column following positive
   source $x$ and increasing row positive source $y$. No recentering, shifts,
   rotations or crops per image.
5. **Keep the units source-independent.** Integrate over each pixel (or
   declare mean density or point samples), normalize to incident photons at a
   declared plane, and never normalize an individual PSF to its own sum or
   peak.
6. **Declare the symmetry and propagate the mirror checks.** State which
   argument makes the intensity two-axis symmetric for the actual optical
   state (see {doc}`../theory/quadrant-symmetry`). Propagate, directly, an
   $x$-negative, a $y$-negative and a both-negative position for each
   product. Do not create them by flipping stored images.
7. **Propagate independent interior checks.** Positions between nodes inside
   the cell that will be qualified, with the same frame, optical state and
   normalization, so reads can be compared with direct images.
8. **Report numerical convergence and storage precision.** Run a few check
   positions at refined propagation settings and report the changes, and compare
   float32 storage with float64 at identical settings.

## What the starter rectangle looks like

The figure shows the starter request for the public LUVOIR-A APLC: 27 by 31
nodes at 0.38 $\lambda_0/D$, the mirrored nodes that exist only by symmetry,
and the small cell $3.8\le s_x<4.18$, $5.32\le s_y<5.70$ that the delivery is
qualified on (the small outlined box near the cross). The shaded square is
the region where every read has a complete 32 by 32 stencil among the stored
and mirrored nodes.

```{code-cell} python
:tags: [hide-input]
starter = sc.Lattice(jnp.zeros(2), jnp.full(2, 0.38), (31, 27))
cell = sc.Domain("box", "lower_closed_upper_open",
                 jnp.array([[3.8, 5.32], [4.18, 5.70]]), jnp.zeros(2))
fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
viz.plot_coverage(starter, domain=cell, quadrant=True, query=(4.0, 5.5), ax=ax)
plt.show()
```

The outlined rectangle is the stencil of one read at $(4.0, 5.5)$. It uses
five mirrored columns at negative $x$ and one mirrored row at negative $y$,
and it ends exactly on the last stored column and row, which is why the
rectangle needs 27 by 31 nodes and no more.

## When symmetry holds, and when it does not

The theory page argues that a real pupil, a real Lyot stop and a real,
centrosymmetric occulter make the intensity unchanged when source and
detector are inverted together, so one geometric mirror is enough for both.
The toy coronagraph tests that directly by decentering its pupil. A shift of
2 percent of $D$ along $x$ leaves the pupil symmetric about the $x$ axis, and a
diagonal shift leaves it symmetric about neither.

```{code-cell} python
:tags: [hide-input]
cases = {"pupil shifted along x": (0.02, 0.0), "pupil shifted diagonally": (0.02, 0.02)}
receipts = {}
for name, shift in cases.items():
    admission = sc.admit_grid(**tc.quadrant_delivery(tc.ToyCoronagraph(pupil_shift=shift)))
    receipts[name] = admission.receipt
    print(f"{name}: admitted={admission.admitted}, "
          f"reasons={sc.reason_names(admission.reason_bits)}")
    print("   mirror residuals (x-neg, y-neg, both-neg):",
          ", ".join(f"{v:.1e}" for v in admission.receipt.mirror_local_peak))
```

```{code-cell} python
:tags: [hide-input]
fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharex=True, constrained_layout=True)
for ax, (name, receipt) in zip(axes, receipts.items()):
    viz.plot_check_residuals(receipt, ax=ax)
    ax.set_title(name)
plt.show()
```

The toy delivery carries no interior checks, so that rail is empty, and two
of the diagonal case's mirror residuals are equal and overlap. The shift along
$x$ passes every check at the level of float64 rounding:
the pupil keeps its $x$-axis mirror and conjugation supplies the other.

The diagonal shift fails the $x$-negative and $y$-negative checks at about 6
percent of the local peak, and admission refuses with
`SYMMETRY_CHECK_FAILED`. Its both-negative check still passes, because
inverting both coordinates is exactly the symmetry conjugation provides, so
the three checks together separate the two arguments. A delivery that fails
them needs a full signed grid, not a looser tolerance: sinchole will not
accept a tolerance above $10^{-6}$.

## Before sending

- The manifest lists every file with its SHA-256 hash, and FITS headers agree
  with it (`PIXSCALE` in $\lambda_0/D$ per pixel, with the unit in its
  comment).
- Images are finite and non-negative, and offsets are float64, $x$ then $y$, in
  the order of the images.
- Stellar and sky products stay separate and unchanged.
- Failed or unperformed checks are reported as such, never as passing.
