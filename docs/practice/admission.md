---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  render_markdown_format: myst
---

# What does a refusal protect you from?

A grid that is too coarse does not fail loudly. Its reads still return
images, and each one is wrong by an amount nobody checks. Admission exists to
catch that before the first read, and every read checks its own position
too. Both report what went wrong as named reason bits. This page asks how
the library decides from the spacing alone that a delivery is unusable, what
it says when it refuses, and why version 0.1 reports every error bound as
unknown.

```{code-cell} python
:tags: [remove-cell]
import hwostyle
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import eyepiece as ep

jax.config.update("jax_enable_x64", True)
hwostyle.use("dark")
pal = hwostyle.palette

import sinchole as sc
import toy_coronagraph as tc
from IPython.display import Markdown
from myst_nb import glue


def tex_str(value):
    """Format a number as a LaTeX power of ten."""
    mantissa, exponent = f"{value:.1e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def tex(value):
    """Return the power of ten as a Markdown display object."""
    return Markdown(tex_str(value))
```

## Does a slightly coarser grid cost a slightly larger error?

The toy grid at spacing 0.38 $\lambda_0/D$ sits exactly at the kernel margin,
$\rho=2Bh=0.76$ with $B=1$ for the toy's one-diameter pupil. Suppose a
designer saves propagations by coarsening it to 0.45, which is $\rho=0.90$.

```{admonition} Predict before reading on
:class: tip
By what factor does the kernel's worst-tone error grow when the spacing goes
from 0.38 to 0.45: about 2, about 100 or about 60,000?
```

The animation sweeps the spacing from $\rho=0.5$ to the Nyquist limit at 1.
The curve is the kernel's measured worst-tone error from the kernel page, drawn
as the spacing grows. The verdict is the answer of `admit_grid`, run on a
real toy delivery at each spacing.

```{code-cell} python
:tags: [hide-input]
# Medium gate, ground 4 (a sweep whose tracked mark cannot be matched across a
# strip), S1 and S3: one knob, the spacing, with its value on screen. The
# tracked mark is the head on the error curve and the verdict text, and the
# mid-sweep event is the flip at 0.76, where the error leaves the 1e-6 floor
# and admission changes its answer, then the second flip at 1 where a second
# reason joins. A strip would need a frame at 0.76 and one just above it to
# show a threshold, which the still below provides for the static record.
rho_frames = np.unique(np.concatenate([np.arange(0.5, 1.0001, 0.0125), [0.76]]).round(4))
spacings = rho_frames / 2
errors = np.array([tc.kernel_error_worst(r) for r in rho_frames])
verdicts = []
for h in spacings:
    delivery_h = tc.quadrant_delivery(spacing=float(h), nodes=17, domain_half=1.0)
    outcome = sc.admit_grid(**delivery_h)
    verdicts.append(sc.reason_names(outcome.reason_bits) if not outcome.admitted else ())
hold = lambda a, n: np.concatenate([np.full(n, a[0]), a, np.full(n + 3, a[-1])])
order = hold(np.arange(len(rho_frames)), 3)

fig, ax = plt.subplots(figsize=(7.5, 3.9), constrained_layout=True)
ax.set_yscale("log")
ax.set_xlim(0.5, 1.03)
ax.set_ylim(1e-8, 3)
ax.set_xlabel(r"oversampling factor $\rho = 2Bh$ (spacing $h = \rho / 2$)")
ax.set_ylabel("kernel worst-tone error / amplitude")
ep.plot.ref_line(ax, 0.76, "margin 0.76", orient="v", where="end")
ep.plot.ref_line(ax, 1.0, "Nyquist 1", orient="v", where="end")
ep.motion.carried_tag(ax, r"kernel error against $\rho$", corner="upper left")
(curve,) = ax.plot([], [], color=pal.cyan, lw=2)
(head,) = ax.plot([], [], "o", color=pal.cyan, ms=7)
verdict = ax.text(0.03, 0.56, "", transform=ax.transAxes, va="top", fontsize=13)
spacing_text = ax.text(0.03, 0.43, "", transform=ax.transAxes, va="top", fontsize=10,
                       color=ep.style.neutral(0.7))


def draw(fig, k):
    k = int(k)
    curve.set_data(rho_frames[: k + 1], errors[: k + 1])
    head.set_data([rho_frames[k]], [errors[k]])
    names = verdicts[k]
    verdict.set_text("admitted" if not names else " + ".join(names))
    verdict.set_color(ep.style.role("text") if not names else ep.style.role("alert"))
    spacing_text.set_text(rf"spacing $h$ = {spacings[k]:.4f} $\lambda_0/D$, error {errors[k]:.1e}")


anim = ep.render.animate(fig, draw, order, fps=10)
from IPython.display import HTML

HTML(anim.jshtml(dpi=80))
```

```{code-cell} python
:tags: [hide-input]
fig, ax = plt.subplots(figsize=(7.5, 3.9), constrained_layout=True)
ax.semilogy(rho_frames, errors, color=pal.cyan, lw=2)
ax.set_xlim(0.5, 1.03)
ax.set_ylim(1e-8, 3)
ax.set_xlabel(r"oversampling factor $\rho = 2Bh$ (spacing $h = \rho / 2$)")
ax.set_ylabel("kernel worst-tone error / amplitude")
ep.plot.ref_line(ax, 0.76, "margin 0.76", orient="v", where="end")
ep.plot.ref_line(ax, 1.0, "Nyquist 1", orient="v", where="end")
before, after = errors[rho_frames == 0.76][0], errors[rho_frames == 0.9][0]
ax.annotate(tex_str(before), (0.76, before), (0.6, before * 0.2), color=pal.cyan,
            arrowprops={"arrowstyle": "-", "color": pal.cyan, "lw": 0.8})
ax.annotate(tex_str(after) + " at 0.90", (0.9, after), (0.8, after * 25), color=pal.cyan,
            arrowprops={"arrowstyle": "-", "color": pal.cyan, "lw": 0.8})
plt.show()
print(f"error growth from rho 0.76 to 0.90: {after / before:.0f}x")
growth = errors[rho_frames == 0.9][0] / errors[rho_frames == 0.76][0]
glue("a_before", tex(before), display=False)
glue("a_after", tex(after), display=False)
glue("a_growth", tex(growth), display=False)
```

```{dropdown} Answer
About 60,000. At 0.38 the worst-tone error is {glue}`a_before` of the
amplitude, and at 0.45 it is {glue}`a_after`, a factor near
{glue}`a_growth`. The still
marks the cliff: admission flips from admitted to `KERNEL_MARGIN_FAIL` at
$\rho=0.76$, just past where the error leaves its floor, and adds
`NYQUIST_FAIL` at $\rho=1$.
```

The rule is a cliff, so there is no gentle middle in which a slightly
coarse grid is slightly wrong. This is why admission compares the spacing
with the margin before any read, using the declared support width, and
refuses with a reason that names the fix.

## Which refusals can a delivery expect, and what fixes each?

| Reason | What it means | What to do |
|---|---|---|
| `KERNEL_MARGIN_FAIL` | $\rho=2Bh>0.76$ on some axis: the grid is too coarse for the kernel's accuracy | Propagate a finer grid, or check the declared support width and shortest wavelength |
| `NYQUIST_FAIL` | $\rho\ge1$: the grid is coarser than the Nyquist limit and cannot determine the image between nodes | As above, and this always comes with `KERNEL_MARGIN_FAIL` |
| `UNKNOWN_BANDWIDTH` | The support width or its evidence is missing | Declare the full illuminated entrance widths with a source |
| `METADATA_MISSING` | A premise, identity or provenance is unknown or empty | Supply the declaration, since sinchole never assumes a premise |
| `GRID_INCOMPLETE`, `GRID_DUPLICATE` | Missing or repeated node indices in the rectangle | Deliver the complete rectangle, once each |
| `GRID_NONUNIFORM` | Offsets disagree with the declared lattice beyond four ULPs | Check the coordinate conversion, or pass `offset_original_dtype` |
| `SYMMETRY_CHECK_FAILED` | A mirror or axis residual exceeds $10^{-6}$ of its local peak | The optics are not mirror symmetric at that level, so deliver a full signed grid |
| `SYMMETRY_CHECK_MISSING` | A quadrant grid lacks one of the three negative counterparts | Propagate the missing mirror checks |
| `OUTSIDE_DOMAIN`, `INCOMPLETE_STENCIL` | (read) The query is outside the permission region, or its stencil runs off the grid. `INCOMPLETE_STENCIL` is also raised at admission: fewer than 17 nodes per quadrant axis, fewer than 32 per full-grid axis, or a failed safe-point search over the declared domain | Read inside the served region, or deliver a larger rectangle |
| `FRAME_UNSUPPORTED`, `NORMALIZATION_UNSUPPORTED`, `METADATA_CONFLICT` | A premise is declared false (premises 1 to 4, 5 and 6 in that order), or the frame, pixel, normalization or wavelength declarations disagree. Through yippy, a false premise raises a parse error before admission | Correct the declaration, or deliver a model for which the premise holds |
| `PRECISION_UNAVAILABLE` | JAX 64-bit mode is off | `jax.config.update("jax_enable_x64", True)` before building arrays |
| `MEMORY_LIMIT` | An allocation estimate exceeds the `MemoryBudget` | Raise the ceilings or tile reads more finely |
| `BOUND_UNKNOWN` | (read) No error bound exists. Version 0.1 sets it on every read, accepted or not, and a bounded policy refuses because of it | Nothing to fix in a sampling read. Bounded reads need a future error model |
| `EMPIRICAL_ONLY` | (read) The only accuracy evidence is the finite set of empirical checks | Informational. It is set whenever checks exist |

A refusal can carry several reasons at once, except the two early returns
above, and `sc.reason_names(bits)` lists them.

## A refusal, end to end

The toy grid at spacing 0.38 has $\rho=0.76$, exactly at the kernel margin.
Coarsening it to 0.45 gives $\rho=0.9$, which the sweep above shows is past
the cliff, and admission refuses before any read.

```{code-cell} python
coarse = tc.quadrant_delivery(spacing=0.45, nodes=27)
refused = sc.admit_grid(**coarse)
print("admitted:", refused.admitted)
print("reasons:", sc.reason_names(refused.reason_bits))
try:
    refused.require()
except sc.AdmissionRefused as err:
    print("AdmissionRefused:", err, "| receipt attached:", err.receipt is not None)
```

## What does admission check, and in what order?

Admission checks in a fixed order, and two failures end the run before
anything else is looked at. Everything after them is either accumulated into
one report or skipped once an earlier group has failed.

1. **Precision.** If JAX 64-bit mode is off, `PRECISION_UNAVAILABLE` is
   returned alone.
2. **Memory.** An allocation estimate from the declared array sizes is compared
   with the `MemoryBudget`. A breach returns `MEMORY_LIMIT` alone, before any
   other check and before anything large is allocated.
3. **Semantic checks, accumulated together.** Structure (array shapes and
   dtypes), metadata and the six premises with their evidence, geometry (a
   supported layout, a complete rectangle without duplicates, offsets within
   four ULPs of the declared lattice, values in the supported numeric range
   with `LAYOUT_UNSUPPORTED` and `GRID_NONUNIFORM` for violations), node
   data (finite and non-negative), bandwidth ($\rho=2(\lambda_0/\lambda_{\min})Lh$
   below 1 for `NYQUIST_FAIL` and at most 0.76 for `KERNEL_MARGIN_FAIL`,
   using the shortest contributing wavelength and, if given, the campaign
   minimum), the stencil minimum (`INCOMPLETE_STENCIL`) and the memory
   estimate again with the full set of arrays. All reasons found here are
   reported together.
4. **Symmetry.** For quadrant grids, the declared reflection, a tolerance no
   looser than $10^{-6}$, every mirror residual and every axis-invariance
   check, computed from the images. These run only if no earlier reason was
   found.
5. **Empirical interior checks.** If a `CheckSet` is supplied, each check is
   read and its residual is recorded in the receipt. The residual never gates
   admission: a large one is evidence, not a refusal. Only a malformed
   check set (non-finite values, missing identifiers) or a check position the
   reader refuses returns `DATA_INVALID`.

Because the symmetry checks wait for everything earlier, a delivery with a
semantic problem never learns whether it would also have failed a mirror
check. The residual denominator also differs by check type. The mirror and
axis checks recompute their local peak from the delivered images, but the
interior check divides by the `core_peak` the delivery supplied, which the
library does not recompute.

## What does the receipt record about a decision?

The `Receipt` is a small, immutable host-side record of the admission. Its
`identity` hashes the data, the declarations and the provenance text, so it
changes whenever any of them does. It records the stored and served regions,
the premise evidence, the canonicalization residual of the offsets, the
allocation estimate, and the empirical evidence: interior check residuals
(`empirical_local_peak`, `empirical_absolute`), mirror residuals
(`mirror_local_peak`) and axis invariance (`axis_exact`, `axis_relative`).
The function `sinchole.viz.plot_check_residuals` draws the evidence fields.

## Why does version 0.1 report every error bound as unknown?

A read under the default `ReadPolicy` is a sampling read: it returns the
image whenever the grid was admitted and the query is served, and reports
`abs_error_bound = inf` and `bound_available = False`. That is not a claim
that the error is large. It is a refusal to claim anything the library
cannot support.

The interior and mirror checks in the receipt are evidence
about the delivered model at specific positions, and the comparisons on
{doc}`first-read` and {doc}`../validation` show what that evidence looks
like, but neither is a bound at other positions. A bounded `ReadPolicy`
requires an error model that can certify every term of the error budget,
which version 0.1 does not have, so every bounded request refuses with
`BOUND_UNKNOWN`. The {doc}`../roadmap` describes the certificates that will
change this.

## Every reason bit

```{code-cell} python
:tags: [remove-input]
from IPython.display import Markdown

rows = ["| Bit | Reason |", "|---|---|"]
for reason in sc.Reason:
    rows.append(f"| {reason.value.bit_length() - 1} | `{reason.name}` |")
Markdown("\n".join(rows))
```

## What this page established

Admission refuses a delivery whose spacing puts $\rho$ above 0.76 because the
kernel's worst-tone error rises about 60,000 times across that cliff, and it names every
reason it finds. Version 0.1 reads without a certified bound, so the checks
in the receipt are evidence and never a bound. A delivery that passes is
the designer's job to build, which the next page walks through.

Next: {doc}`designers`.
