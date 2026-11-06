---
file_format: mystnb
kernelspec:
  name: python3
---

# Admission and refusals

The sinchole library would rather refuse than return a number it cannot stand behind.
Every grid passes through admission once, and every read checks its own
position, and both report what went wrong as named reason bits. This page
lists the common refusals with what to do about each, shows what admission
checks and in what order, and explains why version 0.1 reports every error
bound as unknown.

```{code-cell} python
:tags: [remove-cell]
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import sinchole as sc
import toy_coronagraph as tc
```

## Common refusals and what to do

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
| `OUTSIDE_DOMAIN`, `INCOMPLETE_STENCIL` | (read) The query is outside the permission region, or its stencil runs off the grid | Read inside the served region, or deliver a larger rectangle |
| `PRECISION_UNAVAILABLE` | JAX 64-bit mode is off | `jax.config.update("jax_enable_x64", True)` before building arrays |
| `MEMORY_LIMIT` | An allocation estimate exceeds the `MemoryBudget` | Raise the ceilings or tile reads more finely |
| `BOUND_UNKNOWN` | (read) No error bound exists. Version 0.1 sets it on every read, accepted or not, and a bounded policy refuses because of it | Nothing to fix in a sampling read. Bounded reads need a future error model |
| `EMPIRICAL_ONLY` | (read) The only accuracy evidence is the finite set of empirical checks | Informational. It is set whenever checks exist |

A refusal can carry several reasons at once, and `sc.reason_names(bits)`
lists them.

## A refusal, end to end

The toy grid at spacing 0.38 has $\rho=0.76$, exactly at the kernel margin.
Coarsening it to 0.45 gives $\rho=0.9$, which the foundations page showed is
past the cliff, and admission refuses before any read.

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

## What admission checks, in order

Admission accumulates reasons rather than stopping at the first, so one
report lists everything wrong with a delivery. The checks fall into seven
groups.

1. **Structure.** Array shapes and dtypes, finite non-negative samples,
   declarations present.
2. **Geometry.** A supported layout (full signed rectangle, or quadrant with
   origin $(0,0)$ and at least 17 nodes per axis), a complete rectangle
   without duplicates, offsets within four ULPs of the declared lattice, and
   lattice and domain values inside the supported numeric range
   ({doc}`../reference/numerical-contract`).
3. **Physical premises.** All six premises true with known evidence, a
   supported frame, pixel measure and normalization.
4. **Bandwidth.** For every product and axis, $\rho=2(\lambda_0/\lambda_{\min})Lh$
   below 1 (`NYQUIST_FAIL`) and at most 0.76 (`KERNEL_MARGIN_FAIL`), using
   the shortest contributing wavelength and, if given, the campaign minimum.
5. **Symmetry.** For quadrants, the declared reflection, a tolerance no
   looser than $10^{-6}$, every mirror residual and every axis-invariance
   check computed from the images.
6. **Empirical checks.** Interior independent checks, if supplied, are read
   and their residuals recorded relative to the local peak.
7. **Memory.** Allocation estimates for resident samples, staging, outputs
   and scratch against the `MemoryBudget`, before anything large is
   allocated.

## What the receipt records

The `Receipt` is a small, immutable host-side record of the admission. Its
`identity` hashes the data, the declarations and the provenance text, so it
changes whenever any of them does. It records the stored and served regions,
the premise evidence, the canonicalization residual of the offsets, the
allocation estimate, and the empirical evidence: interior check residuals
(`empirical_local_peak`, `empirical_absolute`), mirror residuals
(`mirror_local_peak`) and axis invariance (`axis_exact`, `axis_relative`).
The function `sinchole.viz.plot_check_residuals` draws the evidence fields.

## Sampling reads and unknown bounds

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

## All reason bits

```{code-cell} python
:tags: [remove-input]
from IPython.display import Markdown

rows = ["| Bit | Reason |", "|---|---|"]
for reason in sc.Reason:
    rows.append(f"| {reason.value.bit_length() - 1} | `{reason.name}` |")
Markdown("\n".join(rows))
```
