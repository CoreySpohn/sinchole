---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  render_markdown_format: myst
---

# How we know: the physicaloptix starter delivery

The theory pages predict that a grid at $h=0.38\,\lambda_0/D$ read with the
32-tap Kaiser-sinc kernel reproduces a coronagraph's planet image to about
one part in a million. A prediction from a toy model is not evidence about a
real design, so sinchole version 0.1 was tested against a delivery made the
way a designer would make one: independent propagations of a real
coronagraph, compared with reads at positions the grid never saw. The
question for this page is how far the million-fold claim carries, and which
denominator it is measured against.

```{code-cell} python
:tags: [remove-cell]
import csv
import sys

sys.path.insert(0, "scripts")
from IPython.display import Markdown, display
from validation_tables import bin_tables, load, sci, static_dir

v = load("validation.json")
q = load("qualification_extended.json")
d = load("floor_diagnosis.json")
b = load("benchmark.json")
with open(static_dir() / "qualification_extended.csv") as handle:
    rows = list(csv.DictReader(handle))
radii = [float(r["r_lod"]) for r in rows]
```

## The delivery

The design is the public LUVOIR-A apodized pupil Lyot coronagraph (APLC),
propagated with the open-source [physicaloptix](https://physicaloptix.readthedocs.io)
package at a single wavelength. The delivery follows {doc}`practice/designers`:

```{code-cell} python
:tags: [remove-input]
display(Markdown(
    f"- {v['training_nodes']} training PSFs on a 27 by 31 positive quadrant at spacing "
    "0.38 $\\lambda_0/D$, 128 by 128 pixels of 0.25 $\\lambda_0/D$, stored in float32.\n"
    f"- {v['independent_checks']} independent interior checks inside the cell "
    "$3.8\\le s_x<4.18$, $5.32\\le s_y<5.70$, propagated directly in float64. This is the last "
    "cell whose 32-tap stencil fits inside the stored quadrant, so it is the only cell the "
    "version 0.1 grid serves, and it straddles the design inner working angle of "
    f"{q['design_iwa_lod']} $\\lambda_0/D$.\n"
    f"- {v['mirror_checks']} independent mirror checks at negative positions. A stored base with "
    "both indices nonzero has three signed counterparts and a base on an axis has one, so "
    f"{v['mirror_bases_both_indices_nonzero']} bases times 3 plus "
    f"{v['mirror_bases_on_an_axis']} axis bases times 1 gives "
    f"{v['mirror_checks_at_those_bases'] + v['mirror_checks_at_axis_bases']} checks."
))
```

The check positions were chosen before any read, and the comparison metric
and the target of $10^{-5}$ were fixed in advance. The `yippy-sinc validate`
command on the starter delivery reports PASS, with sampling margins $\rho$ of
0.7190 and 0.7463 on the two axes against the limit of 0.76, and all 58 stored
axis images exactly invariant under reflection.

## What the 16 starter checks showed

```{code-cell} python
:tags: [remove-input]
display(Markdown(
    "| Quantity | Result | Estimate or reference |\n|---|---|---|\n"
    f"| Admission | admitted, no reasons, {b['admission_first_call_seconds']:.0f} s on first call | none |\n"
    f"| Mirror checks, {v['mirror_checks']} positions | residual exactly {sci(v['mirror_residual_max'])} | "
    "exact symmetry of a real apodizer and Lyot stop |\n"
    f"| Axis invariance, all stored axis images | residual exactly {sci(v['axis_residual_max'])} | same |\n"
    f"| Worst of {v['independent_checks']} interior checks, error over local peak | "
    f"{sci(v['worst_check_error_local_peak'])} | kernel floor about "
    f"{sci(d['band']['kernel_tone_error_max_in_band'])} per tone, target $10^{{-5}}$ |\n"
    f"| Same worst check, error over each pixel's brightest stored value | "
    f"{sci(v['worst_check_error_pixel_brightest'])} | see the diagnosis below |\n"
    f"| yippy blend, worst of the same checks, error over local peak | "
    f"{sci(v['blend_worst_error_local_peak'])} | one planet core misplaced by a fraction of a node |"
))
```

The numbers come from `docs/scripts/validation_figures.py`, which reads the
delivery from its manifest, admits it with sinchole and writes
`_static/validation/validation.json` beside the figures below.

```{figure} _static/validation/worst_check.png
:alt: Four panels around the worst interior check near 4 and 5.4 lambda0 over D. The direct propagation and the sinchole read look identical. The sinchole residual panel is blank at a color scale of one part in a million of the local peak. The blend residual shows a dipole across the planet core.
:width: 100%

Compare the two residual panels, each on its own color scale. The sinchole
residual has no structure at the planet core, while the blend residual is a
dipole there, because blend interpolation cannot place a core between nodes.
```

```{figure} _static/validation/check_residuals.png
:alt: Three rails of residuals on a logarithmic axis. Independent checks lie between 3e-8 and 1.5e-6, left of the 1e-5 target. All mirror checks and all axis checks are exactly zero.
:width: 85%

Find where the independent checks sit against the $10^{-5}$ target on the top
rail. All of them fall below it, and every mirror and axis check is exactly
zero.
```

## Does the error hold between the checks?

Sixteen positions cannot speak for a continuum, so the evidence was widened
before any claim was made. A seeded set of positions was drawn inside the
served cell and read by sinchole and by yippy's blend interpolation, and each
read was compared with a direct float64 propagation.

```{code-cell} python
:tags: [remove-input]
g = q["groups"]
display(Markdown(
    f"- {g['starter']} starter checks, {g['behind']} positions behind the inner working angle "
    f"(source radius below {q['design_iwa_lod']}), {g['near']} positions from the inner working "
    f"angle to 0.2 above it, and {g['random']} uniform positions over the cell drawn with seed "
    f"{q['seed']}. That makes {q['all']['n']} reads.\n"
    f"- The cell reaches source radii from {min(radii):.2f} to {max(radii):.2f} "
    "$\\lambda_0/D$, so the \"behind\" group is the part of the cell just inside the design "
    "inner working angle and not the deep interior of the occulter.\n"
    f"- Photometric aperture radius {q['aperture_radius_lod']} $\\lambda_0/D$."
))
```

A prediction before the tables: will the blend's median error over the local
peak, the same metric used for sinchole, be near $10^{-3}$, near $10^{-1}$,
or near 1?

````{dropdown} Answer
It is a few percent to ten percent, the middle choice. The blend median over
the local peak is given in the first table below. The blend gets the planet
core's brightness roughly right and its position wrong, so the error is a
fraction of the core and not the whole of it.
````

Three metrics are reported for every read. Each one uses all 16384 pixels of
the image:

- Local peak: the largest absolute pixel difference, over the brightest
  direct pixel within 1 $\lambda_0/D$ of the source.
- Pixel brightest: the largest over pixels of the absolute difference divided
  by that pixel's brightest stored value, taken over the quadrant and its
  three reflections. A pixel that is dim at every stored source position can
  have a large ratio for a tiny absolute error.
- Core flux: the relative error of the summed pixels whose centers lie within
  0.7 $\lambda_0/D$ of the true source, which is the number a yield code
  actually uses.

The blend is computed exactly as yippy's `OffJAX` does it in the version that
produced these numbers:

```{code-cell} python
:tags: [remove-input]
display(Markdown(
    f"- Version: yippy {q['yippy_version']}. Its routine reproduces the delivery's own stored "
    f"blend reads to {sci(q['blend_matches_delivery_reads_max_abs'])} in absolute value.\n"
    "- Take the four delivered images nearest the query by polar arc-length distance.\n"
    "- Weight each by $1/d^2$, with $d$ that distance, and normalize the weights to sum to one.\n"
    "- Shift each image to the query position with a Fourier phase ramp.\n"
    "- Sum the four shifted, weighted images."
))
```

### Sinchole reads

```{code-cell} python
:tags: [remove-input]
display(Markdown(bin_tables(q, "sinc")))
```

### Blend interpolation, same positions and units

```{code-cell} python
:tags: [remove-input]
display(Markdown(bin_tables(q, "blend")))
```

```{figure} _static/validation/qualification_radius.png
:alt: Three panels of relative read error against source radius, one per metric. The sinchole points sit in a thin horizontal band from about 1e-7 to 1e-5 in every panel, and the blend points sit several orders of magnitude higher, with a spread of about a decade.
:width: 100%

Look at the vertical gap between the two clouds in each panel. The cyan
sinchole reads stay near their floor across the inner working angle line, and
the pink blend reads sit far above, so the separation does not depend on
which metric or which radius is chosen.
```

```{code-cell} python
:tags: [remove-input]
s, a = q["all"]["sinc"], q["all"]["blend"]
display(Markdown(
    f"Over all {q['all']['n']} reads the sinchole worst case over the local peak is "
    f"{sci(s['local_peak']['worst'])} and over each pixel's brightest value "
    f"{sci(s['pixel_brightest']['worst'])}, while the worst core flux error is "
    f"{sci(s['photometric']['worst'])}. The blend's worst values are "
    f"{sci(a['local_peak']['worst'])}, {sci(a['pixel_brightest']['worst'])} and "
    f"{sci(a['photometric']['worst'])}."
))
```

## Where does the pixel-relative error come from?

Measured against each pixel's brightest stored value, the worst error is
several times the local-peak error. Three candidates could explain it, and
each was measured separately.

```{code-cell} python
:tags: [remove-input]
st, bd = d["storage"], d["band"]
lines = bd["lines"]
oob = max(p["out_of_band_rms_fraction"] for ln in lines.values() for p in ln["pixels"])
top = max(p["highest_frequency_above_1e8_of_peak"] for ln in lines.values() for p in ln["pixels"])
lv = {r["position"]: r for r in d["propagation_levels"]}
cell_lv = [r for r in d["propagation_levels"] if r["position"].startswith(("worst", "node 1"))]
kx = d["kernel_on_exact_function"]["sweeps"]["x"]
ky = d["kernel_on_exact_function"]["sweeps"]["y"]
display(Markdown(
    "| Candidate | Measurement | Verdict |\n|---|---|---|\n"
    f"| Storage precision | Reading with float64 nodes changes the worst pixel error from "
    f"{sci(st['worst_pixel_relative_float32_nodes'], 4)} to "
    f"{sci(st['worst_pixel_relative_float64_nodes'], 4)}. float32 roundoff over the brightest value is "
    f"at most {sci(st['node_quantization_over_brightest_max'])}. | not the cause |\n"
    f"| A band wider than the declared half band of {bd['declared_half_band']:.0f} | "
    f"Out-of-band energy at the worst pixels is at most {sci(oob)} of the response rms, and "
    f"the highest frequency above $10^{{-8}}$ of the peak is {top:.3f} of the declared edge. | "
    "not the cause: the overshoot is about one percent and carries almost no energy |\n"
    f"| Propagation noise | Levels of propagator resolution differ by "
    f"{sci(min(r['level0_vs_level2_over_local_peak'] for r in cell_lv))} to "
    f"{sci(max(r['level0_vs_level2_over_local_peak'] for r in d['propagation_levels']))} of the "
    "local peak, a smooth model change far above the error being explained, and the same error "
    "appears when the kernel reads a finely sampled exact propagation. | not the cause |\n"
    f"| Kernel floor on a response that peaks outside the stored lattice | Error over the "
    f"pixel's own peak response is {sci(ky['worst_error_over_pixel_peak_in_cell'])} along $y$ and "
    f"{sci(kx['worst_error_over_pixel_peak_in_cell'])} along $x$, at every position, while error "
    f"over the brightest stored value grows toward the cell edge. | the cause |"
))
```

The first row says storing the nodes in float64 leaves the error unchanged, so
it is not stored rounding. The third row needs one sentence: a different
resolution of the same propagator moves every pixel smoothly by far more than
$10^{-6}$, so noise at that level cannot be what the kernel is reading.

```{figure} _static/validation/floor_diagnosis.png
:alt: Two panels of one-axis read error against source position, along x on the left and along y on the right. The cyan curve, error over each pixel's brightest stored value, is flat near 4e-7 and then climbs past 2e-6 inside the shaded served cell. The pink curve, error over each pixel's peak response, stays flat near 4e-7 everywhere.
:width: 100%

Follow the cyan curve into the shaded served cell and then look at the pink
curve beneath it. The cyan error climbs only where the pixel's stored values
become tiny compared with the response it would have at its own peak. The pink
error, which uses the pixel's peak response as denominator, does not move.
```

The mechanism is one sentence long. A pixel far from the planet sees almost
nothing from sources inside the stored quadrant, so its brightest stored value
is small, but its response peaks at a source position outside that quadrant.
The kernel's error scales with the response beyond the stencil edge, which is
large, and not with the stored values, which are small. At the worst
pixel the stored brightest is a small fraction of the peak response.

```{code-cell} python
:tags: [remove-input]
wp = d["location"]["worst_pixels"][0]
display(Markdown(
    f"Along $x$ the peak response is {kx['pixel_peak_over_brightest_at_worst_pixel']:.0f} times the "
    f"stored brightest value at the worst pixel, and along $y$ it is "
    f"{ky['pixel_peak_over_brightest_at_worst_pixel']:.0f} times. The worst error sits at pixel "
    f"({wp['ix']}, {wp['iy']}), {wp['r_from_source_lod']:.1f} $\\lambda_0/D$ from the source, where the "
    f"stored brightest is {wp['brightest_over_local_peak']:.2f} of the local peak."
))
```

## Reading the numbers

```{code-cell} python
:tags: [remove-input]
display(Markdown(
    "| Denominator | Worst error over all reads | What it means |\n|---|---|---|\n"
    f"| Local peak | {sci(s['local_peak']['worst'])} | one part in a million of the planet core, "
    f"estimate from the kernel floor {sci(d['band']['kernel_tone_error_max_in_band'])} per tone |\n"
    f"| Pixel's brightest stored value | {sci(s['pixel_brightest']['worst'])} | six parts in a million, "
    "the same kernel error on pixels the quadrant barely lights |\n"
    f"| Pixel's own peak response | {sci(ky['worst_error_over_pixel_peak_in_cell'])} | "
    "the kernel floor itself, flat in position |\n"
    f"| Core flux in a 0.7 aperture | {sci(s['photometric']['worst'])} | the quantity a yield code uses |"
))
```

So the statement "about one part in a million" holds when the denominator is
the local peak, the brightest pixel near the planet. It does not hold per
pixel against the stored brightest value, where the worst is about six parts
in a million, and the reason is now known. The foundations page's toy value of
$7\times10^{-7}$ for the kernel itself, the per-pixel floor measured here
against peak response, and the local-peak figure all agree to within a factor
of a few, which is the agreement the theory predicted.

The mirror checks are exactly zero because the physicaloptix propagation is
exactly symmetric under the declared reflections, which is what the
conjugation argument of {doc}`theory/quadrant-symmetry` predicts for a real
apodizer, a real Lyot stop and a hard occulting disk. Through yippy
(`Coronagraph(path, interpolation="sinc")`), the same checks pass the same
target.

## What this does not establish

- One cell, one wavelength, one design. The served cell is the only part of
  the quadrant where the 32-tap stencil fits, so every read here sits within
  about half a node of the inner working angle. Reads deep behind the occulter
  and far outside it are not yet served and so are not qualified.
- Positions are seeded draws, not a proof over the continuum. The 200 random
  positions bound the typical error and the worst case over those draws is
  stated above, and the true worst case can only be larger.
- The truth is a direct float64 propagation by the same code that made the
  nodes. It checks that sinchole reproduces the model defined by the delivered
  arrays, and it does not test that model against a real telescope.
- Propagator resolution differs from level to level by far more than the read
  error, so the grid's own discretization of the optics, not the read, limits
  accuracy against the continuous design.
- Speed and memory are measured on one CPU only. See {doc}`reference/performance`.
  There is no GPU measurement ("GPU untested").
- Designs with complex focal-plane masks need their own symmetry evidence and
  checks.

Next: the {doc}`roadmap` lists the qualification still to come.
