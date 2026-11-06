---
file_format: mystnb
kernelspec:
  name: python3
---

# How can one quadrant serve the whole field?

A grid that covers planets in all four quadrants costs four times the
propagations of a grid that covers one. Many coronagraphs are mirror
symmetric, so in principle a designer could deliver one quadrant and let the
reader supply the rest. This page asks what the reader has to do with such a
grid, and what must be true of the optics for the shortcut to be honest.

```{code-cell} python
:tags: [remove-cell]
import hwostyle
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import eyepiece as ep
import sinchole as sc
from sinchole import viz
import toy_coronagraph as tc

jax.config.update("jax_enable_x64", True)
hwostyle.use("dark")
pal = hwostyle.palette
toy = tc.ToyCoronagraph()
delivery = tc.quadrant_delivery(toy)
admission = sc.admit_grid(**delivery)
reader = tc.reader_for(delivery)
lattice = delivery["declared_lattice"]
```

## What does a read near the edge of a quadrant need?

The toy coronagraph of the previous page has a circular pupil, a circular
occulter and a circular Lyot stop, so it is symmetric about both axes. Its
quadrant grid stores 31 by 31 nodes at spacing 0.38 $\lambda_0/D$, all with
$s_x\ge0$ and $s_y\ge0$, starting exactly at the origin.

```{admonition} Predict before reading on
:class: tip
A planet sits at $s_x=+0.2\,\lambda_0/D$, inside the stored quadrant. Does
its read need any data from negative source positions? If so, roughly what
fraction of its stencil?
```

```{dropdown} Answer
Yes, about half. Its position is $t=0.2/0.38=0.53$ samples, so its stencil
runs from node $-15$ to node $16$ along $x$: 15 of its 32 columns of nodes
lie at negative $s_x$, where nothing was stored.
```

If reversing the source's $x$ position reverses the detector columns, and
reversing $y$ reverses the rows, then a stencil node at a negative position
can read the stored image at the mirrored positive node, flipped, and the
read is continuous across the axis as though the full grid had been
delivered.

## A read can cross the axis through mirrored nodes without a seam

The left panel of the animation is the stencil view of the kernel page: the
read's 32 by 32 taps over the node lattice, each marker's area proportional
to the size of its weight and its color for the sign. Stored nodes are the
bright dots and the mirrored nodes that exist only by symmetry are dim. The
animation first holds a read at $s_x=1.5$ and folds every column of taps at a
negative $x$ onto the stored column it reads.

Then the query moves through the axis to $s_x=-1.5$ at fixed $s_y=3$, with
the folded stencil reading stored nodes throughout. The right panel is the
image that the reader returns, and the planet's core moves smoothly through
the axis with no seam.

```{code-cell} python
:tags: [hide-input]
# Medium gate, ground 4 (a sweep whose tracked mark cannot be matched across a
# strip), S3: a sense reversal. The knob is the query x (with the fold held
# fixed after its own opening beat); the tracked mark is the stencil, whose
# columns switch from stored to mirrored nodes as the query crosses the axis,
# and the right panel must show no seam at that crossing, which a strip of
# separate stills cannot show.
sx = np.concatenate([np.full(10, 1.5), np.linspace(1.5, -1.5, 31), np.full(5, -1.5)])
fold = np.concatenate([np.zeros(3), np.linspace(0, 1, 8)[1:], np.ones(36)])
path = np.stack([sx, np.full_like(sx, 3.0)], axis=1)
reads = sc.read_batch(reader, jnp.asarray(path), jnp.zeros(len(sx), dtype=jnp.int32))
images = np.asarray(reads.intensity)


def negative_columns(s):
    floor = np.floor(s / 0.38)
    taps = floor + np.arange(-15, 17)
    return int(np.sum(taps < 0))


fig, (left, right) = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
stencil_view = viz.plot_stencil_2d(path[0], lattice, quadrant=True, max_area=250.0, ax=left)
ep.motion.carried_tag(left, "the stencil of a read")
left.axvline(0, color=ep.style.neutral(0.45), lw=1, ls="--")
left.set_xlim(-9, 9)
left.set_ylim(-5, 12.5)
left.text(11.2, 11.8, "stored nodes", ha="right", color=ep.style.neutral(0.7), fontsize=8)
left.text(-8.7, -4.3, "mirrored nodes", color=ep.style.neutral(0.45), fontsize=8)
psf = viz.plot_read(images[0], pixscale_lod=toy.pixscale_lod, ax=right,
                    vmin=1e-9, vmax=0.5)
right.axvline(0, color=ep.style.neutral(0.45), lw=1, ls="--")
right.set_xlim(-6, 6)
right.set_ylim(-3, 9)


def draw(fig, k):
    stencil_view.update(path[k], fold=fold[k])
    psf.update(images[k])
    left.set_title(rf"$s_x = {sx[k]:+.2f}$: {negative_columns(sx[k])} of 32 columns at $s_x<0$",
                   fontsize=10)


anim = ep.render.animate(fig, draw, range(len(sx)), fps=10)
from IPython.display import HTML

HTML(anim.jshtml(dpi=62))
```

```{code-cell} python
:tags: [hide-input]
fig, axes = plt.subplots(2, 3, figsize=(9, 6), constrained_layout=True)
for j, k in enumerate((10, 25, 41)):
    view = viz.plot_stencil_2d(path[k], lattice, quadrant=True, fold=1.0,
                               max_area=250.0, ax=axes[0, j])
    axes[0, j].axvline(0, color=ep.style.neutral(0.45), lw=1, ls="--")
    axes[0, j].set_xlim(-9, 9)
    axes[0, j].set_ylim(-5, 12.5)
    axes[0, j].set_title(rf"$s_x = {sx[k]:+.2f}$: {negative_columns(sx[k])} of 32 "
                         r"columns at $s_x<0$", fontsize=9)
    viz.plot_read(images[k], pixscale_lod=toy.pixscale_lod, ax=axes[1, j],
                  vmin=1e-9, vmax=0.5, colorbar=False)
    axes[1, j].axvline(0, color=ep.style.neutral(0.45), lw=1, ls="--")
    axes[1, j].set_xlim(-6, 6)
    axes[1, j].set_ylim(-3, 9)
plt.show()
```

Three moments of the sweep: before, on and after the axis, with the folded
stencil above and the read below. The middle read, at $s_x=0$ from a stencil
that is almost half mirrored nodes, is as clean as the other two.

Mechanically, a tap at signed node index $(i,j)$ reads the stored image at
$(|i|,|j|)$, reversing the detector columns when $i<0$ and the rows when
$j<0$. The kernel weights are the same as for a full grid, so a quadrant
read is algebraically identical to a full-grid read of the symmetric
extension. The detector must be centered halfway between its middle pixels,
at `((nx - 1)/2, (ny - 1)/2)`, so that a row or column reversal is an exact
mirror.

## When is a mirrored node a faithful copy of the real one?

The shortcut is only as good as the symmetry. Two geometric mirrors of the
optics suffice: if the pupil, the occulter and the Lyot stop are all
symmetric about the $x$ axis and about the $y$ axis, each source mirror maps
to the matching detector mirror.

One geometric mirror can also suffice,
through a less obvious route. If the entrance amplitude, the Lyot stop and
the focal-plane mask are all real, and the mask is also centrosymmetric
(unchanged by inversion through its center, as a circular disk is), then
complex conjugation of the field gives $E(-\mathbf{x};-\mathbf{s})=E^*(\mathbf{x};\mathbf{s})$,
so the intensity is unchanged by inverting source and detector together.
Combined with a single geometric mirror, that inversion supplies the second
mirror.

The public LUVOIR-A APLC pupil has a column mirror and no row mirror, and
the conjugation argument is what makes this apodized pupil Lyot
coronagraph (APLC) two-axis symmetric in intensity.

Many things break the argument: aberrations or deformable mirror phases,
misalignment, polarization aberrations, and complex focal-plane masks such as
those of hybrid Lyot and vortex coronagraphs. A designer therefore declares
which argument applies to the actual optical state, and sinchole checks it.

## How does the reader know the symmetry is real?

The declaration alone is never trusted. A quadrant delivery carries
independent propagations at negative positions (an $x$-negative, a
$y$-negative and a both-negative counterpart for each product), and
admission compares each with the flipped stored image of its mirrored node.
It also checks that every stored image on an axis is unchanged by the
corresponding flip, since a source on the $x$ axis must produce an image
symmetric under row reversal. The tolerance is $10^{-6}$ of the local peak.
A looser tolerance cannot be requested, and caller-supplied pass flags carry
no weight. For the toy, every residual sits at the level of float64 rounding.

```{code-cell} python
:tags: [hide-input]
receipt = admission.receipt
print("admitted:", admission.admitted)
print("mirror residuals / local peak:", [f"{v:.1e}" for v in receipt.mirror_local_peak])
axis = np.asarray(receipt.axis_relative, dtype=float)
print(f"axis images: {sum(receipt.axis_exact)} exactly invariant, "
      f"largest relative residual of the rest {np.nanmax(axis):.1e}")
```

A finite set of checks cannot prove the symmetry everywhere, but it catches
the common mistakes: a decentered pupil, a mislabeled axis or a flipped
detector convention. {doc}`../practice/designers` decenters the toy's pupil
and shows both sides of the argument: a shift along one axis keeps the
intensity symmetric, and a diagonal shift is refused.

## Try it

1. A planet sits at $s_x=-0.2$, $s_y=1.0$ on the toy grid. Before computing,
   predict how many of its 32 stencil columns lie at negative $s_x$, then
   check with `negative_columns` from the hidden animation cell.
   ```{dropdown} Hint
   $t=-0.2/0.38=-0.53$, so the floor is $-1$ and the columns run from $-16$
   to $15$: 16 negative columns, one more than at $s_x=+0.2$.
   ```
2. Which of these would break the conjugation argument for the toy: a Lyot
   stop shifted along $y$, a small focus aberration added to the pupil, a
   pupil scaled to 0.95 $D$?
   ```{dropdown} Hint
   The focus aberration, because it makes the pupil field complex. The
   shifted stop stays real and keeps a mirror axis, and a scaled pupil keeps
   both mirrors (it changes the bandwidth instead, see
   {doc}`band-limits`).
   ```

## What this page established

A quadrant grid reads negative positions by flipping stored images, which
is exact for optics with the declared two-axis symmetry, and admission
verifies that symmetry against independent propagations to $10^{-6}$ of the
local peak before any read. With bandwidth, spacing and symmetry settled, the
remaining question is what the numerical contract promises about the taps.

Next: {doc}`../reference/numerical-contract`.
