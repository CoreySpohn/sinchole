---
file_format: mystnb
kernelspec:
  name: python3
---

# What limits how fast a planet image changes?

A grid of PSFs can only be read between its nodes if the image changes
slowly enough with source position that the nodes pin it down. If a planet
image could change arbitrarily fast as the planet moved, no grid would be
fine enough and every interpolation would be a guess. This page asks what
sets the speed limit for a coronagraph, and turns the answer into the node
spacing a designer should use.

```{dropdown} Symbols on this page
$\mathbf{s}$: source position in $\lambda_0/D$. $\mathbf{x}$: detector
position. $\mathbf{u}$: entrance-pupil coordinate in units of $D$.
$\sigma=\lambda_0/\lambda$. $P$: the illuminated pupil. $L$: its full width in
units of $D$. $g$: everything the fixed optics do. $E$: the detector field.
$B$: the intensity bandwidth in cycles per $\lambda_0/D$. $h$, $\rho$: node
spacing and oversampling factor, as on the foundations page.
```

```{code-cell} python
:tags: [remove-cell]
import hwostyle
import jax
import matplotlib.pyplot as plt
import numpy as np

import eyepiece as ep
from sinchole import viz
from toy_coronagraph import ToyCoronagraph

jax.config.update("jax_enable_x64", True)
hwostyle.use("dark")
pal = hwostyle.palette
toy = ToyCoronagraph()
x_pix = np.asarray(toy.detector_x)
col = int(np.argmin(np.abs(x_pix - 3.1)))
row = int(np.argmin(np.abs(x_pix - 0.6)))
```

## How fast does one pixel of a toy coronagraph change?

The examples use a toy Lyot coronagraph computed on the page: a circular
entrance pupil one diameter $D$ wide, an opaque occulting disk of radius
2.5 $\lambda_0/D$ in the first focal plane, a Lyot stop 0.8 $D$ wide, and a
detector of 96 by 96 pixels, each 0.25 $\lambda_0/D$ across. It is small
enough to propagate hundreds of PSFs in a fraction of a second, and it has
the property that matters, which is that the image changes shape, not only
position, as the planet moves past the occulter. One detector pixel, at
$(3.1, 0.6)\,\lambda_0/D$, serves as the probe.

As the planet slides along $x$, the probe pixel's brightness traces out a
curve against the planet's position $s_x$. That curve has wiggles, and their
frequency is measured in cycles per $\lambda_0/D$ of planet motion, the
quantity the foundations page called the bandwidth.

```{admonition} Predict before reading on
:class: tip
The entrance pupil is one diameter wide. What is the highest frequency in
the probe pixel's curve: 0.5, 1 or 2 cycles per $\lambda_0/D$ of planet
motion?
```

```{dropdown} Answer
1 cycle per $\lambda_0/D$, set by the one-diameter width of the entrance
pupil. The occulter, the Lyot stop and the detector change how bright the
pixel gets, but they cannot make it vary faster.
```

A source tilt multiplies the entrance pupil by a phase ramp, so every field
downstream in fixed linear optics is a sum of pupil contributions with
frequencies set by their pupil coordinate, and contains no source frequency
above half the pupil width. The intensity, a field times its conjugate,
contains none above the full pupil width.

## The image changes in kind, the pixel changes smoothly

In the animation the planet crosses the field from left to right. On the
left the detector image changes character: a bright core outside the
occulter, a faint leak while it hides behind the occulting disk, then the
core again. On the right the probe pixel's intensity is drawn as the planet
moves. However abruptly the image seems to change, the pixel's curve stays
smooth, with wiggles no shorter than one $\lambda_0/D$.

```{code-cell} python
:tags: [hide-input]
s_path = np.concatenate([np.full(3, -6.0), np.linspace(-6.0, 6.0, 37), np.full(6, 6.0)])
path = np.stack([s_path, np.full_like(s_path, 0.5)], axis=1)
images = np.asarray(toy.psf(path))
trace = images[:, row, col]

fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True,
                                  gridspec_kw={"width_ratios": [1, 1.25]})
image = viz.plot_read(images[0], pixscale_lod=toy.pixscale_lod, ax=left,
                      vmin=1e-9, vmax=0.5)
left.add_patch(plt.Circle((0, 0), toy.occulter_lod, fill=False,
                          color=ep.style.neutral(0.5), lw=1, ls="--"))
left.plot(x_pix[col], x_pix[row], "s", mfc="none", mec=pal.yellow, ms=7, mew=1.5)
(planet,) = left.plot([], [], "+", color=pal.pink, ms=10, mew=1.5)
left.set_xlim(-8, 8)
left.set_ylim(-8, 8)
right.set_xlim(-6, 6)
right.set_ylim(1e-9, 1)
right.set_yscale("log")
right.set_xlabel(r"planet position $s_x$ ($\lambda_0/D$)")
right.set_ylabel("probe pixel intensity")
s_fine = np.linspace(-6, 6, 961)
fine = np.asarray(toy.psf(np.stack([s_fine, np.full_like(s_fine, 0.5)], 1)))[:, row, col]
(curve,) = right.plot([], [], color=pal.yellow, lw=2)
(dot,) = right.plot([], [], "o", color=pal.yellow, ms=6)
right.axvspan(-toy.occulter_lod, toy.occulter_lod, color=ep.style.neutral(0.15), lw=0)
right.text(0, 2e-9, "behind occulter", ha="center", color=ep.style.neutral(0.55), fontsize=9)


def draw(fig, k):
    image.update(images[k])
    planet.set_data([path[k, 0]], [path[k, 1]])
    shown = s_fine <= s_path[k]
    curve.set_data(s_fine[shown], fine[shown])
    dot.set_data([s_path[k]], [trace[k]])


anim = ep.render.animate(fig, draw, range(len(s_path)), fps=10)
from IPython.display import HTML

HTML(anim.jshtml(dpi=72))
```

```{code-cell} python
:tags: [hide-input]
fig = plt.figure(figsize=(9, 4.6), constrained_layout=True)
grid = fig.add_gridspec(2, 3, height_ratios=[1, 0.9])
for j, k in enumerate((8, 22, 36)):
    ax = fig.add_subplot(grid[0, j])
    viz.plot_read(images[k], pixscale_lod=toy.pixscale_lod, ax=ax, vmin=1e-9, vmax=0.5,
                  colorbar=False)
    ax.add_patch(plt.Circle((0, 0), toy.occulter_lod, fill=False,
                            color=ep.style.neutral(0.5), lw=1, ls="--"))
    ax.plot(x_pix[col], x_pix[row], "s", mfc="none", mec=pal.yellow, ms=6, mew=1.2)
    ax.plot(path[k, 0], path[k, 1], "+", color=pal.pink, ms=9, mew=1.5)
    ax.set_title(rf"$s_x = {path[k, 0]:+.1f}$")
ax = fig.add_subplot(grid[1, :])
ax.semilogy(s_fine, fine, color=pal.yellow, lw=2)
ax.axvspan(-toy.occulter_lod, toy.occulter_lod, color=ep.style.neutral(0.15), lw=0)
ax.set_xlabel(r"planet position $s_x$ ($\lambda_0/D$)")
ax.set_ylabel("probe pixel")
plt.show()
```

The still holds the same content: three moments of the image above, with
the planet (pink cross) before, behind and past the occulter, and the whole
curve of the probe pixel (yellow square) below. The curve spans about ten
orders of magnitude, and its sharp dips are zeros of the intensity drawn on
a logarithmic axis, not fast features. Between them nothing varies faster
than about once per $\lambda_0/D$.

## Why the pupil sets the limit

A distant source at angular position $\mathbf{s}$ (in units of $\lambda_0/D$)
arrives at the entrance pupil as a tilted plane wave, which multiplies the
pupil field by the phase ramp $e^{2\pi i\sigma\mathbf{u}\cdot\mathbf{s}}$,
where $\mathbf{u}$ is the pupil coordinate in units of $D$ and
$\sigma=\lambda_0/\lambda$ accounts for the wavelength. The strip below shows
that ramp for three source positions: the farther the source, the more phase
cycles fit across the pupil.

```{code-cell} python
:tags: [hide-input]
fig, axes = plt.subplots(1, 3, figsize=(8, 2.6), constrained_layout=True)
uu, vv = toy.pupil_coords
for ax, sx in zip(axes, (0.5, 2.0, 4.0)):
    phase = np.angle(np.exp(1j * np.asarray(toy.phase_ramp((sx, 0.0)))))
    phase = np.where(np.asarray(toy.pupil) > 0, phase, np.nan)
    ax.imshow(phase, cmap=hwostyle.cmaps.phase, origin="lower", extent=(-0.5, 0.5, -0.5, 0.5),
              interpolation="nearest")
    ax.set_title(rf"$s_x = {sx}\,\lambda_0/D$")
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
```

Every optic after the entrance pupil (the occulting mask, the Lyot stop, the
detector's pixel response) is fixed and acts linearly on that field, so the
field at detector point $\mathbf{x}$ is

$$
E(\mathbf{x};\mathbf{s})=\int_P g(\mathbf{x},\mathbf{u})\,
e^{2\pi i\sigma\mathbf{u}\cdot\mathbf{s}}\,d\mathbf{u},
$$

where $P$ is the illuminated entrance pupil and $g$ collects everything the
optics do, independent of $\mathbf{s}$. Read as a function of $\mathbf{s}$,
this is a Fourier integral whose frequencies are $\sigma\mathbf{u}$, and
$\mathbf{u}$ ranges only over the pupil. The field therefore contains no
source frequency beyond $\sigma L/2$, where $L$ is the pupil's full width in
units of $D$. The intensity $|E|^2=E\,E^*$ multiplies two such integrals, so
its frequencies are differences $\sigma(\mathbf{u}-\mathbf{u}')$ of pupil
points, and its bandwidth along each axis is

$$
B=\sigma_{\max}L .
$$

The occulter and the Lyot stop do not appear in $B$: they act after the tilt
and never multiply the field by anything that depends on $\mathbf{s}$. A
Lyot stop smaller than the pupil changes the brightness, not the bandwidth.

## A near miss: one source-dependent step breaks the band

The spectrum of the probe pixel's curve makes the band visible. On the left,
the curve of the animation: the spectrum drops by more than five orders of
magnitude at $|f|=1$ cycle per $\lambda_0/D$ and keeps falling beyond it,
where what remains is leakage of the analysis window. On the right, the same curve after one innocent-looking step, dividing
each PSF by its own peak value. That normalization depends on the source
position, so it is not fixed linear optics, and the spectrum spills far past
the band.

```{code-cell} python
:tags: [hide-input]
s_long = np.arange(-512, 512) * 0.05
stack = np.asarray(toy.psf(np.stack([s_long, np.full_like(s_long, 0.5)], 1)))
linear_trace = stack[:, row, col]
peak_normalized = linear_trace / stack.max(axis=(1, 2))
fig, (a, b) = plt.subplots(1, 2, figsize=(9, 2.8), sharey=True, constrained_layout=True)
viz.plot_band_spectrum(linear_trace, 0.05, bandwidth=1.0, ax=a)
viz.plot_band_spectrum(peak_normalized, 0.05, bandwidth=1.0, ax=b)
a.set_title("fixed linear optics")
b.set_title("each PSF divided by its own peak")
for ax in (a, b):
    ax.set_xlim(-4, 4)
plt.show()
```

This is why sinchole refuses products that apply a source-dependent step,
such as per-PSF normalization, recentering, moving image stamps or optics
that change with field position. Different products (for example different
wavelengths) may carry different fixed normalization factors. A factor that
varies with the source may not.

## How does a bandwidth become a node spacing?

The foundations page found that the 32-tap Kaiser-sinc kernel is accurate
to about $10^{-6}$ when the oversampling factor $\rho=2Bh$ stays at or below
0.76. For a monochromatic product at $\lambda_0$ ($\sigma_{\max}=1$) and a
pupil one diameter wide ($L=1$), the bandwidth is 1 cycle per $\lambda_0/D$,
the Nyquist spacing is 0.5, and the kernel allows $0.76/2=0.38\,\lambda_0/D$.
A product whose shortest contributing wavelength is $0.9\lambda_0$ has
$\sigma_{\max}=1.11$ and needs $0.76/2.22=0.342$. In general, with
$B=\sigma_{\max}L$, the largest admissible spacing along an axis is

$$
h\le\frac{0.76}{2\,\sigma_{\max}L}.
$$ A pupil support wider than the
diameter used for $\lambda_0/D$ (an oversized secondary support, for
instance) needs a proportionally finer grid, which is why the delivery
declares the full illuminated widths along $x$ and $y$.

The number of nodes follows from the spacing. A quadrant grid that serves
reads up to, but not including, $R$ along an axis needs
$\lceil R/h\rceil+16$ nodes, indices 0 through $\lceil R/h\rceil+15$, because
each read reaches 16 nodes past the start of its own cell. The physicaloptix starter delivery on {doc}`../validation` serves the
cell $3.8\le s_x<4.18$, $5.32\le s_y<5.70$ at $h=0.38$, and $27\times31=837$
nodes are the smallest quadrant rectangle whose stencils cover it.

## When does the bandwidth bound stop holding?

The bandwidth bound holds for a source that enters as a paraxial tilt at the
entrance pupil and passes through optics that do not change with source
position: fixed masks, stops and deformable mirror settings, a fixed pixel
response, a fixed mixture of polarizations summed incoherently, and spectral
weights that do not depend on the source. Pupil shear, field-dependent
aberrations or source-dependent normalization break it. Within it, the bound
is exact, but it says nothing about how accurately the delivered images
represent the real telescope, and sinchole interpolates the model the arrays
define.

## Try it

1. A design's illuminated pupil is 1.1 $D$ wide along $x$ and 1.0 $D$ along
   $y$. Predict the largest admissible spacing along each axis for a
   monochromatic product, then check it with the formula.
   ```{dropdown} Hint
   $h_x\le0.76/2.2=0.345$ and $h_y\le0.38$. The axes may use different
   spacings, and admission checks each separately.
   ```
2. In the executed near-miss cell, replace the per-PSF peak normalization by
   multiplication with a constant factor of 2. Predict the spectrum before
   running it.
   ```{dropdown} Hint
   A constant does not depend on the source, so the spectrum keeps its shape
   and stays inside the band, and only its normalization changes, and the plot
   normalizes it away.
   ```
3. Reaching back to the foundations page: if a delivery accidentally used
   $h=0.45$ for $L=1$ at $\lambda_0$, what is $\rho$, and what error should a
   read expect?
   ```{dropdown} Hint
   $\rho=2\times1\times0.45=0.9$, past the cliff, where the measured kernel
   error is about $3\times10^{-2}$. Admission refuses it with
   `KERNEL_MARGIN_FAIL` ({doc}`../practice/admission` shows exactly this
   case).
   ```

## What this page established

A coronagraph's planet intensity at any fixed pixel is band-limited in source
position with bandwidth $B=\sigma_{\max}L$ cycles per $\lambda_0/D$, so a
grid at $h\le0.76/(2B)$ (0.38 $\lambda_0/D$ for a monochromatic product and a
one-diameter pupil) can be read between its nodes to about $10^{-6}$. The
next page halves the cost twice by storing one quadrant and reading the other
three through symmetry.
