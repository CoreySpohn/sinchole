---
file_format: mystnb
kernelspec:
  name: python3
---

# A first read

This tutorial admits a grid of PSFs, reads it between its nodes and checks
the reads against direct propagation. It uses the toy Lyot coronagraph from
the theory pages, whose helper module (`toy_coronagraph.py`, in the `docs`
folder of the repository) builds a complete quadrant grid: 31 by 31 PSFs at
spacing 0.38 $\lambda_0/D$, mirror checks, and the declarations sinchole
requires. {doc}`declarations` explains each declaration, and here they are
taken as given.

```{code-cell} python
:tags: [remove-cell]
import eyepiece as ep
import hwostyle
import matplotlib.pyplot as plt
import numpy as np

hwostyle.use("dark")
pal = hwostyle.palette
```

## Admit the grid

Enable JAX 64-bit mode first, because sinchole refuses to build a reader without it.
Admission is a one-time check of the whole grid, and it either returns a
usable grid or explains why not.

```{code-cell} python
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import sinchole as sc
import toy_coronagraph as tc

toy = tc.ToyCoronagraph()
delivery = tc.quadrant_delivery(toy)
admission = sc.admit_grid(**delivery)
print("admitted:", admission.admitted)
print("reasons:", sc.reason_names(admission.reason_bits))
print("receipt:", admission.receipt.identity[:16], "...")
```

The receipt records what was admitted: its identity is a hash of the data,
the declarations and their provenance, so readmitting changed data gives a
new identity.

## Build a reader and read

An `Interpolator` binds the admitted grid to a permission region (here the
delivery's box of half-width 5 $\lambda_0/D$), the kernel profile, a read
policy, a tiling and a memory budget.

```{code-cell} python
reader = sc.Interpolator(
    admission.require(),
    delivery["domain"],
    sc.KernelProfile(),
    sc.ReadPolicy(),
    sc.Tiling(pixel_tile=1024),
    delivery["memory"],
)
result = reader.read(jnp.array([4.0, 0.3]), jnp.int32(0))
print("accepted:", bool(result.accepted), " shape:", result.intensity.shape)
print("error bound:", float(result.abs_error_bound))
```

The read returns the image together with its status. The error bound is
`inf` because version 0.1 does not certify accuracy, and a read in sampling mode
still returns its image, and the comparison below is how to judge it.

## Compare with direct propagation

The toy can propagate the planet directly at the same position, which gives
the truth to compare against. The residual is shown relative to the local
peak, the brightest direct pixel within 1 $\lambda_0/D$ of the planet.

```{code-cell} python
:tags: [hide-input]
from sinchole import viz

query = np.array([4.0, 0.3])
truth = np.asarray(toy.psf(query))
fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
viz.plot_read(result, pixels=delivery["products"].pixels, ax=a,
              vmin=1e-9, vmax=0.5)
res = viz.plot_residual(result, truth, source_xy_lod=query,
                        pixels=delivery["products"].pixels, scale="magnitude", ax=b)
for ax in (a, b):
    ax.set_xlim(-1, 8)
    ax.set_ylim(-4.5, 4.5)
a.set_title("sinchole read at (4.0, 0.3)")
b.set_title("|read - direct| / local peak")
plt.show()
```

The residual is a few parts in $10^7$ of the planet's peak, with no
structure tied to the planet's core: it is the kernel's truncation error, the
floor measured on {doc}`../foundations/sinc-interpolation`.

## Many reads at once

`read_batch` maps a batch of positions through the reader in fixed-size
chunks, and the reader works inside `eqx.filter_jit`.

```{code-cell} python
import equinox as eqx

rng = np.random.default_rng(1)
radius = rng.uniform(0.5, 4.8, 60)
angle = rng.uniform(0, 2 * np.pi, 60)
positions = jnp.asarray(np.stack([radius * np.cos(angle), radius * np.sin(angle)], 1))
batch = eqx.filter_jit(sc.read_batch)(reader, positions, jnp.zeros(60, dtype=jnp.int32))
print("all accepted:", bool(batch.accepted.all()), " images:", batch.intensity.shape)
```

An error can be measured against two references. One is the planet's own
peak, the local peak used so far. The other is each pixel's brightest value
over all stored source positions, the most light that pixel ever receives as
the planet moves.

```{admonition} Predict before reading on
:class: tip
Outside the occulter, reads match direct propagation to about $10^{-6}$ of
the planet's own peak or better. For a planet behind the occulting disk, at
about 1.7 $\lambda_0/D$, will the error relative to the planet's own peak be
smaller, about the same, or larger?
```

```{dropdown} Answer
Larger, up to about $2\times10^{-4}$ at the positions sampled here. The
absolute error in a pixel follows how bright that pixel gets for the
planet positions in its stencil, not the occulted planet's own faint image,
whose peak is near $10^{-3}$ of the unocculted peak. Relative to each
pixel's brightest value the error stays below $10^{-6}$ at every sampled
position.
```

```{code-cell} python
:tags: [hide-input]
truths = np.asarray(toy.psf(positions))
error = np.abs(np.asarray(batch.intensity) - truths)
own_peak = truths.max(axis=(1, 2))
stored = np.asarray(delivery["intensity_nodes"][0])
brightest = stored.max(axis=0)
brightest = np.maximum.reduce([brightest, brightest[:, ::-1], brightest[::-1, :],
                               brightest[::-1, ::-1]])
relative_own = error.max(axis=(1, 2)) / own_peak
relative_pixel = (error / brightest).max(axis=(1, 2))

fig, ax = plt.subplots(figsize=(7, 3), constrained_layout=True)
ax.semilogy(radius, relative_own, "o", color=pal.pink, ms=4)
ax.semilogy(radius, relative_pixel, "s", color=pal.cyan, ms=4)
ax.axvline(toy.occulter_lod, color=ep.style.neutral(0.4), lw=1, ls="--")
ax.text(0.55, relative_own.max() * 1.6, "relative to the planet's own peak",
        color=pal.pink, ha="left", va="bottom", fontsize=9)
ax.text(4.8, relative_pixel.max() * 1.8, "relative to each pixel's brightest value",
        color=pal.cyan, ha="right", va="bottom", fontsize=9)
ax.set_ylim(relative_pixel.min() * 0.5, relative_own.max() * 5)
ax.text(toy.occulter_lod + 0.05, relative_pixel.max() * 30, "occulter edge", ha="left",
        va="center", fontsize=9, color=ep.style.neutral(0.6))
ax.set_xlabel(r"planet radius ($\lambda_0/D$)")
ax.set_ylabel("max |read - direct|")
plt.show()
print(f"worst relative to own peak: {relative_own.max():.1e}; "
      f"worst relative to pixel brightest: {relative_pixel.max():.1e}")
```

The two series measure the same errors against two references. Relative to
each pixel's brightest value (cyan) the error is flat and below $10^{-6}$ at
every sampled radius, the size the foundations page measured for the kernel. Relative to the planet's
own peak (pink) it rises behind the occulter, because the denominator falls.
For science that cares about an occulted planet's photometry, the absolute
error is what matters, and it is the same small number everywhere.

## Refusals are values, not surprises

A read outside the permission region is refused. Compiled reads return a
refused status and a NaN image instead of raising, so a batch never stops
halfway, and `require_read` is the eager helper that raises with the refused
indices and reasons.

```{code-cell} python
outside = reader.read(jnp.array([6.5, 0.0]), jnp.int32(0))
print("accepted:", bool(outside.accepted), " status:", sc.Status(int(outside.status)).name)
print("reasons:", sc.reason_names(int(outside.reason_bits)))
try:
    sc.require_read(outside)
except sc.ReadRefused as err:
    print("ReadRefused:", type(err).__name__)
```

## What this tutorial established

You admitted a grid, read it at arbitrary positions individually, in batches
and under JIT, compared reads with direct propagation, and saw a refusal.
At the 60 sampled positions, the error stays below $10^{-6}$ of each pixel's
brightest value. {doc}`declarations` explains what each
declaration in the toy delivery claims, and {doc}`admission` lists what
admission refuses and how to fix it.
