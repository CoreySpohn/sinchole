# How we know: the physicaloptix starter delivery

The theory pages predict that a grid at $h=0.38\,\lambda_0/D$ read with the
32-tap Kaiser-sinc kernel reproduces a coronagraph's planet image to about
$10^{-6}$. A prediction from a toy model is not evidence about a real design,
so sinchole version 0.1 was tested against a delivery made the way a
designer would make one: independent propagations of a real coronagraph,
compared with reads at positions the grid never saw. This page reports that
test and what it does and does not establish.

## The delivery

The design is the public LUVOIR-A apodized pupil Lyot coronagraph (APLC),
propagated with the open-source [physicaloptix](https://physicaloptix.readthedocs.io)
package at a single wavelength. The delivery follows {doc}`practice/designers`:

- 837 training PSFs on a 27 by 31 positive quadrant at spacing
  0.38 $\lambda_0/D$, 128 by 128 pixels of 0.25 $\lambda_0/D$, stored in
  float32.
- 16 independent interior checks inside the cell $3.8\le s_x<4.18$,
  $5.32\le s_y<5.70$, a cell that crosses the inner working angle, propagated
  directly in float64.
- 20 independent mirror checks at negative positions, including three signed
  counterparts at each of two stored bases near the outer edges of the
  rectangle.

The check positions were chosen before any read, and the comparison metric
(the largest absolute pixel difference divided by the local peak, the
brightest direct pixel within 1 $\lambda_0/D$ of the source) and the target of
$10^{-5}$ were fixed in advance.

## Results

| Quantity | Result |
|---|---|
| Admission | admitted, no reasons, 7 s on a laptop CPU |
| Mirror checks, 20 positions | residual exactly 0 |
| Axis invariance, all stored axis images | exactly invariant |
| Worst of 16 interior checks, max error / local peak | $1.44\times10^{-6}$ (target $10^{-5}$) |
| Aperture flux error, worst check | $1.5\times10^{-8}$ |
| Read time, batched, CPU | about 42 ms per 128 by 128 PSF |
| yippy blend interpolation on the same positions | 0.19 |

The numbers come from `docs/scripts/validation_figures.py`, which reads the
delivery from its manifest, admits it with sinchole and writes
`_static/validation/validation.json` beside the figures below.

```{figure} _static/validation/worst_check.png
:alt: Four panels around the worst interior check at about (4, 5.4) lambda0/D. The direct propagation and the sinchole read look identical. The sinchole residual panel is blank at a color scale of one part in a million of the local peak. The blend residual shows a dipole of a few percent across the planet core.
:width: 100%

The worst of the 16 interior checks. Look at the two residual panels, each
on its own color scale: sinchole's residual peaks at $1.4\times10^{-6}$ of
the local peak with no structure at the planet core, while blend
interpolation misplaces the core and leaves a dipole of 5 percent at this
position (19 percent at its worst check).
```

```{figure} _static/validation/check_residuals.png
:alt: Three rails of residuals on a logarithmic axis. Independent checks lie between 3e-8 and 1.5e-6, left of the 1e-5 target. All 20 mirror checks and all axis checks are exactly zero.
:width: 85%

Every piece of evidence in the receipt on one axis. All 16 interior checks
fall below the target, the worst by a factor of nearly seven, and every
mirror and axis check is exactly zero.
```

## Reading the numbers

The foundations page measured the kernel's own error at $\rho=0.76$ as about
$7\times10^{-7}$ of a signal's total amplitude, and on the toy coronagraph
the read error stayed below $10^{-6}$ of each pixel's brightest stored value.

The worst interior error here, $1.44\times10^{-6}$ of the local peak, is of
that order. Measured instead against each pixel's brightest stored value,
the worst error on this delivery is $6.1\times10^{-6}$, several times the
toy's, so some dim pixels carry errors above the kernel floor. Their source
(propagation noise in the delivered images, a support wider than declared,
or storage precision) has not been separated for this delivery, and the
comparison is reported here rather than explained away.

The mirror checks are exactly zero because the physicaloptix propagation is
exactly symmetric under the declared reflections, which is what the
conjugation argument of {doc}`theory/quadrant-symmetry` predicts for a real
apodizer, a real Lyot stop and a hard occulting disk. Blend interpolation, by
contrast, combines neighboring delivered images, and a planet core that moves
between nodes cannot be built from cores that sit at the nodes.

Through yippy (`Coronagraph(path, interpolation="sinc")`), the same 16 checks
pass the same target, and yippy's own reads agree with sinchole's.

## What this does not establish

This is one cell, one wavelength and one design, run on CPU. It checks that
sinchole reproduces the model defined by the delivered arrays at 16
positions, and it does not establish the accuracy of that model against a real
telescope, the error at positions that were not checked, or behavior on GPU.
Other designs (with complex focal-plane masks, for instance) need their own
symmetry evidence and checks. The {doc}`roadmap` lists the qualification
still to come.
