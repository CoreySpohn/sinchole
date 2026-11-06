---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  render_markdown_format: myst
---

# sinchole

```{code-cell} python
:tags: [remove-cell]
import csv
import sys

sys.path.insert(0, "scripts")
from IPython.display import Markdown
from myst_nb import glue
from validation_tables import load, static_dir

import toy_coronagraph as tc

v = load("validation.json")
q = load("qualification_extended.json")
bm = load("benchmark.json")
with open(static_dir() / "qualification_extended.csv") as handle:
    rows = list(csv.DictReader(handle))
starter_rows = [r for r in rows if r["group"] == "starter"]
worst_row = max(starter_rows, key=lambda r: float(r["sinc_local_peak"]))


def tex(value):
    """Format a number as a LaTeX power of ten."""
    mantissa, exponent = f"{value:.1e}".split("e")
    return Markdown(rf"${mantissa}\times10^{{{int(exponent)}}}$")


glue("n_reads", str(q["all"]["n"]), display=False)
glue("n_nodes", str(v["training_nodes"]), display=False)
glue("n_checks", str(v["independent_checks"]), display=False)
glue("n_mirror", str(v["mirror_checks"]), display=False)
glue("n_all_checks", str(v["independent_checks"] + v["mirror_checks"]), display=False)
glue("sinc_here", tex(float(worst_row["sinc_local_peak"])), display=False)
glue("blend_here", f"{100 * float(worst_row['blend_local_peak']):.0f}", display=False)
glue("blend_worst", f"{100 * v['blend_worst_error_local_peak']:.0f}", display=False)
glue("pixel_all", tex(q["all"]["sinc"]["pixel_brightest"]["worst"]), display=False)
glue("floor_tone", tex(tc.kernel_error_worst(0.76)), display=False)
glue("ms_f32", f"{1e3 * bm['float32']['single_steady_median_seconds']:.1f}", display=False)
glue("ms_f64", f"{1e3 * bm['float64']['single_steady_median_seconds']:.0f}", display=False)
glue("ms_batch", f"{1e3 * bm['float32']['batch_10000']['seconds_per_read']:.0f}", display=False)
glue("mb_nodes", f"{bm['float32']['resident_mib'] * 1.048576:.0f}", display=False)
```

**Read a coronagraph's planet PSF at any position between the grid points a
designer delivered, to about one part in a million of the planet's local peak
(measured at {glue:text}`n_reads` positions in one cell of one design, with no
bound), or get a clear refusal saying why not.**

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
({doc}`practice/designers`). The starter delivery used on these pages is the
first real grid built this way, the public LUVOIR-A APLC propagated with
physicaloptix on {glue:text}`n_nodes` nodes. Existing YIP packages (yippy
input products, the FITS and CSV files that describe a coronagraph's off-axis
PSFs) keep working with yippy's blend interpolation, a weighted sum of the
four nearest delivered images each shifted to the query, and are not read by
sinchole.

```{figure} _static/validation/worst_check.png
:alt: Four panels around a check position at about (4, 5.4) lambda0/D in the public LUVOIR-A APLC. Direct propagation and the sinchole read look identical. The sinchole residual is blank at a scale of one part in a million. The blend residual shows a dipole of a few percent.
:width: 100%

A planet between grid nodes of the public LUVOIR-A APLC, propagated
directly and read from the delivered grid. The circle marks the
1 $\lambda_0/D$ radius used for the local peak. Look at the two residual
panels, each on its own color scale. At this position the sinchole read is
within {glue}`sinc_here` of the local peak (the brightest direct pixel
within 1 $\lambda_0/D$ of the planet), while blending neighboring images,
yippy's existing interpolation, is off by {glue:text}`blend_here` percent here
and {glue:text}`blend_worst` percent at worst of the {glue:text}`n_checks`
starter checks. Details on {doc}`validation`.
```

:::{admonition} Who can use this today
:class: note

- **Monochromatic, off-axis products only.** Version 0.1 reads planet PSFs
  at one wavelength. Stellar and sky products stay with yippy, and band-mean
  products are declared but not yet demonstrated.
- **One real delivery exists.** The evidence above is one design (the public
  LUVOIR-A APLC) read at {glue:text}`n_checks` independent interior positions
  and {glue:text}`n_mirror` mirror positions, plus {glue:text}`n_reads`
  qualification reads. The one-part-in-a-million figure is what those checks
  measured, not a bound: every read reports its error bound as unknown.
- **The checks cover one cell.** All of them lie in one cell near the inner
  working angle. Measured per pixel against each pixel's brightest stored
  value, the worst error is about {glue}`pixel_all`.
- **A delivery costs propagations.** The starter is {glue:text}`n_nodes` PSFs
  (about {glue:text}`mb_nodes` MB) plus {glue:text}`n_all_checks` checks, close
  to the 52 MB legacy off-axis file for the same design, at a spacing set by
  the optics ({doc}`theory/band-limits`) rather than by convenience.
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
how symmetry halves the grid twice: {doc}`foundations/sampling`,
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
the kernel allows, the node count and the quadrant grid of the starter plan.
{doc}`theory/band-limits` derives each step and explains each panel. The plan
used a support width $L=1$ (so $\rho=0.76$). The starter's declaration is 0.946
by 0.982 $D$, which gives $\rho=0.719$ along $x$ and $0.746$ along $y$. The
label $27\times31$ counts nodes along $x$ and then along $y$, and the code's
lattice shape is ordered $(n_y, n_x)=(31, 27)$.

- Reads off-axis planet intensity products from a complete signed rectangle
  or a positive quadrant with declared, independently checked two-axis
  mirror symmetry.
- Admits a grid only if its declared bandwidth puts it at or below 76
  percent of the Nyquist spacing, the limit up to which the 32-tap kernel's
  worst single tone is measured at about {glue}`floor_tone`.
- Returns images in the delivered units with float64 arithmetic, under
  `eqx.filter_jit` and `jax.vmap`, at {glue:text}`ms_f32` ms per 128 by 128 PSF
  with float32 storage ({glue:text}`ms_f64` ms with float64 storage) in steady
  state, and {glue:text}`ms_batch` ms per read in a batch of $10^4$, on a loaded
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

foundations/sampling
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
