---
file_format: mystnb
kernelspec:
  name: python3
---

# sinchole

**Read a coronagraph's planet PSF at any position between the grid points a
designer delivered, to about one part in a million, or get a clear refusal
saying why not.**

Image simulators, yield codes and exposure-time calculators need the PSF of
a planet wherever the planet happens to be, but a coronagraph designer can
only deliver a finite grid of propagated images. Something has to fill the
gaps, and whatever fills them puts its error straight into planet
photometry. The sinchole library fills them with a windowed-sinc
reconstruction, the interpolation that sampling theory makes exact for a
band-limited signal, checks before every use that the grid meets the
conditions that theory needs, and refuses when it does not. It
is written in [JAX](https://github.com/jax-ml/jax) and
[Equinox](https://github.com/patrick-kidger/equinox).

The sinchole reader takes grids delivered for it: a regular grid of PSFs at a spacing
chosen from the optics, with independent checks and declarations
({doc}`practice/designers`). Existing YIP packages keep working with yippy's
blend interpolation and are not read by sinchole.

```{figure} _static/validation/worst_check.png
:alt: Four panels around a check position at about (4, 5.4) lambda0/D in the public LUVOIR-A APLC. Direct propagation and the sinchole read look identical. The sinchole residual is blank at a scale of one part in a million. The blend residual shows a dipole of a few percent.
:width: 100%

A planet between grid nodes of the public LUVOIR-A APLC, propagated
directly and read from the delivered grid. The circle marks the
1 $\lambda_0/D$ radius used for the local peak. Look at the two residual
panels, each on its own color scale,
the sinchole read is within $1.4\times10^{-6}$ of the local peak (the brightest direct
pixel within 1 $\lambda_0/D$ of the planet), while blending neighboring
images, yippy's existing interpolation, is off by 5 percent here and 19
percent at worst. Details on {doc}`validation`.
```

:::{admonition} Who can use this today
:class: note

- **Monochromatic, off-axis products only.** Version 0.1 reads planet PSFs
  at one wavelength. Stellar and sky products stay with yippy, and band-mean
  products are declared but not yet demonstrated.
- **One real delivery exists.** The evidence above is one design (the public
  LUVOIR-A APLC) read at 16 independent interior positions and 20 mirror
  positions. The one-part-in-a-million figure is what those checks measured,
  not a bound: every read reports its error bound as unknown.
- **A delivery costs propagations.** The starter cell above is 837 PSFs
  (about 57 MB, close to the 52 MB legacy off-axis file for the same
  design), at a spacing set by the optics ({doc}`theory/band-limits`)
  rather than by convenience.
- **Install from git for now.** A PyPI release is pending.
:::

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} You need PSFs at arbitrary planet positions
Read a delivered grid through yippy with `interpolation="sinc"`, or
directly with sinchole. Start with {doc}`practice/first-read`, then
{doc}`practice/yippy`.
:::

:::{grid-item-card} You design coronagraphs and deliver grids
What spacing, which positions, which checks and which declarations a
readable delivery needs: {doc}`practice/designers`.
:::

:::{grid-item-card} You want to know why it works
Sampling theory in general, then why a coronagraph PSF is band-limited and
how symmetry halves the grid twice: {doc}`foundations/sinc-interpolation`,
then {doc}`theory/kernel`.
:::

:::{grid-item-card} You want to know what it does not do
Version 0.1 does not certify accuracy: error bounds are reported as
unknown, and empirical checks are evidence, not bounds. See
{doc}`practice/admission` and the {doc}`roadmap`.
:::
::::

## What version 0.1 does

```{mermaid}
flowchart LR
    D["Designer grid<br/>PSFs, offsets, checks,<br/>declarations"] --> Y["yippy<br/>files, hashes, units"]
    Y --> A["admit_grid<br/>geometry, premises,<br/>sampling, symmetry"]
    A -->|admitted| I["Interpolator<br/>32 x 32 Kaiser-sinc reads"]
    A -->|refused| R["reasons"]
    I --> P["planet PSF<br/>at any served position"]
    I -->|outside| R
```

From a physical support width to the grid a designer delivers, one rule
chains five quantities. Every delivery in these pages follows it.

```{code-cell} python
:tags: [remove-input]
import hwostyle
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import sinchole as sc
from sinchole import viz

jax.config.update("jax_enable_x64", True)
hwostyle.use("dark")
starter = sc.Lattice(jnp.zeros(2), jnp.full(2, 0.38), (31, 27))
fig, ax = plt.subplots(figsize=(10, 2.5), constrained_layout=True)
viz.plot_spacing_rule(lattice=starter, support_width=1.0, sigma_max=1.0, ax=ax)
plt.show()
```

The strip reads left to right: support width, bandwidth, the largest spacing
the kernel allows, the node count and the quadrant grid of the starter
delivery. {doc}`theory/band-limits` derives each step.

- Reads off-axis planet intensity products from a complete signed rectangle
  or a positive quadrant with declared, independently checked two-axis
  mirror symmetry.
- Admits a grid only if its declared bandwidth puts it at or below 76
  percent of the Nyquist spacing, the limit up to which the 32-tap kernel is
  accurate to about $10^{-6}$.
- Returns images in the delivered units with float64 arithmetic, under
  `eqx.filter_jit` and `jax.vmap`, at about 42 ms per 128 by 128 PSF on a
  laptop CPU.
- Refuses with named reasons whenever a declaration is missing, a check
  fails or a read falls outside the served region.

## Install

The first release is pending. Until it is on PyPI, install from source:

```bash
pip install "sinchole[viz] @ git+https://github.com/CoreySpohn/sinchole"
```

After the release, `pip install sinchole` installs the library,
`pip install "sinchole[viz]"` adds the plotting module `sinchole.viz`, and
`pip install "yippy[sinc]"` installs yippy with sinchole as its sinc backend.

Enable JAX 64-bit mode (`jax.config.update("jax_enable_x64", True)`) before
creating arrays or readers.

```{toctree}
:maxdepth: 1
:caption: Getting started
:hidden:

practice/first-read
```

```{toctree}
:maxdepth: 1
:caption: For users
:hidden:

practice/yippy
practice/admission
```

```{toctree}
:maxdepth: 1
:caption: For designers
:hidden:

practice/designers
practice/declarations
practice/delivery-recipe
validation
```

```{toctree}
:maxdepth: 1
:caption: Understanding
:hidden:

foundations/sinc-interpolation
theory/kernel
theory/band-limits
theory/quadrant-symmetry
```

```{toctree}
:maxdepth: 1
:caption: Reference
:hidden:

reference/numerical-contract
reference/stencil-floor
reference/performance
reference/glossary
viz
roadmap
autoapi/index
```
