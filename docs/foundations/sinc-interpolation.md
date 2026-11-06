---
file_format: mystnb
kernelspec:
  name: python3
---

# When do samples determine what lies between them?

Every interpolation scheme makes a guess about what a function does between
its samples. Linear interpolation guesses straight lines, and splines guess
smooth curves, but neither guess is forced by the samples themselves. For one
class of functions the guess is not needed at all: the samples fix every
value in between. This page asks which functions those are, how the exact
reconstruction works, and what it costs to use it with a finite number of
samples, which is the foundation sinchole builds on.

```{dropdown} Symbols on this page
$t$: position in units of the sample spacing. $\rho$: oversampling factor,
twice the highest frequency times the spacing (Nyquist at 1). $W$: kernel
half-width in samples (16). $\beta$: Kaiser window shape parameter (12).
$I_0$: modified Bessel function of order zero. $A$, $N$, $\Delta\omega$:
window rejection in decibels, tap count and transition width, used only in
the estimate.
```

```{code-cell} python
:tags: [remove-cell]
import hwostyle
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import eyepiece as ep
from sinchole import viz
from sinchole.kernel import weights

jax.config.update("jax_enable_x64", True)
hwostyle.use("dark")
pal = hwostyle.palette
```

## What does it mean for a signal to have no fast wiggles?

Take a signal built from twelve cosines with random amplitudes and phases,
none of which completes more than 0.38 cycles per unit of $t$, and sample it
at the integers. The fastest component therefore takes at least 2.6 samples
per cycle. Nothing in the samples alone says the signal is slow, so the
statement that it contains no frequency above 0.38 cycles per sample is an
assumption about the signal, and it is the assumption everything on this page
rests on.

```{code-cell} python
:tags: [hide-input]
rng = np.random.default_rng(3)
freqs = rng.uniform(-0.38, 0.38, 12)
amps = rng.normal(size=12)
phases = rng.uniform(0, 2 * np.pi, 12)


def signal(t):
    """The band-limited test signal, evaluated anywhere."""
    t = np.asarray(t, dtype=float)[..., None]
    return np.sum(amps * np.cos(2 * np.pi * freqs * t + phases), axis=-1)


nodes = np.arange(-40, 41)
samples = signal(nodes)
t_fine = np.linspace(-18, 20, 1500)

fig, ax = plt.subplots(figsize=(7, 2.6), constrained_layout=True)
ax.plot(t_fine, signal(t_fine), color=ep.style.neutral(0.45), lw=1.5)
ax.plot(nodes, samples, "o", color=pal.cyan, ms=4)
ax.set_xlim(-18, 20)
ax.set_xlabel(r"$t$ (samples)")
ax.set_ylabel("signal")
ax.text(19.5, samples.max(), "samples", color=pal.cyan, ha="right", va="top")
ax.text(-17.5, signal(-17.5) + 0.3, "signal", color=ep.style.neutral(0.6))
plt.show()
```

The samples (dots) are all a reconstruction gets to see, and the gray curve is
the truth it should recover.

```{admonition} Predict before reading on
:class: tip
In the exact reconstruction at $t=0.3$, how much does the sample at $t=10$,
ten samples away, contribute? Pick one: nothing, about 3 percent of its
value, or about 30 percent of its value.
```

```{dropdown} Answer
About 3 percent. The exact weight of a sample at distance $\Delta t$ is
$\operatorname{sinc}(\Delta t)=\sin(\pi\Delta t)/(\pi\Delta t)$, and at
$\Delta t=9.7$ its magnitude is $0.027$. The influence of a sample decays only
as one over its distance, so in the exact formula every sample matters a
little. (The practical kernel below tapers that weight to $0.0026$, which is
the point of a window.)
```

A signal sampled at more than two samples per cycle of its fastest component
is fixed between its samples, and every sample contributes to every value,
with an influence that falls off only as one over the distance. That slow
falloff is why a practical reconstruction needs a window, and choosing the
window is what sets how close to that sampling limit a grid can go.

## Each sample spreads as a sinc

A practical reconstruction at a query point combines only a fixed set of
nearby samples, its stencil, and each sample in the stencil (a tap) gets a
weight. The animation follows one query point as it slides across three
sample cells. The bottom panel shows the 32 weights that sinchole's kernel assigns
to the nearest samples, and the top panel, zoomed in around the query, draws
the reconstruction (their weighted sum) as the query moves.

Watch the moment the query lands on a sample: all
weights but one drop to zero, the reconstruction passes exactly through that
sample, and the set of 32 samples in use shifts by one.

```{code-cell} python
:tags: [hide-input]
queries = np.concatenate([np.zeros(4), np.linspace(0.0, 3.0, 46), np.full(6, 3.0)])
fig, (top, bottom) = plt.subplots(
    2, 1, figsize=(7, 4.4), constrained_layout=True,
    gridspec_kw={"height_ratios": [1.2, 1]},
)
top.plot(t_fine, signal(t_fine), color=ep.style.neutral(0.35), lw=1.2)
top.plot(nodes, samples, "o", color=pal.cyan, ms=3.5)
(trace,) = top.plot([], [], color=pal.pink, lw=2.2)
(head,) = top.plot([], [], "o", color=pal.pink, ms=7)
top_line = top.axvline(0.0, color=ep.style.neutral(0.7), lw=1)
top.set_ylabel("signal")
top.set_xlim(-4, 7)
top.set_xlabel(r"$t$ (samples), zoomed")
stencil = viz.plot_stencil(0.0, origin=0.0, spacing=1.0, ax=bottom)
bottom.set_xlim(-18, 20)
readout = bottom.text(0.99, 0.95, "", transform=bottom.transAxes, ha="right", va="top")


def reconstruct(q):
    taps = np.floor(q) + np.arange(-15, 17)
    return float(np.sum(np.asarray(weights(jnp.asarray(q - taps))) * signal(taps)))


t_drawn = np.linspace(0.0, 3.0, 301)
v_drawn = np.array([reconstruct(q) for q in t_drawn])


def draw(fig, q):
    stencil.update(q)
    shown = t_drawn <= q
    trace.set_data(t_drawn[shown], v_drawn[shown])
    head.set_data([q], [reconstruct(q)])
    top_line.set_xdata([q, q])
    readout.set_text(rf"query $t$ = {q:.2f}")


anim = ep.render.animate(fig, draw, queries, fps=12)
from IPython.display import HTML

HTML(anim.jshtml(dpi=80))
```

```{code-cell} python
:tags: [hide-input]
fig, ax = plt.subplots(figsize=(7, 2.2), constrained_layout=True)
still = viz.plot_stencil(0.3, origin=0.0, spacing=1.0, ax=ax)
ax.set_xlim(-18, 20)
plt.show()
```

The still shows the weights at $t=0.3$: most of the weight on the two
nearest samples, then alternating lobes on either side that the window
tapers smoothly to zero by the ends of the 32-sample stencil.

## The ideal formula adds one sinc per sample

The animation is a picture of the Whittaker-Shannon interpolation formula. A
signal $f(t)$ with no frequency above half a cycle per sample is recovered
from its samples by

$$
f(t)=\sum_{n=-\infty}^{\infty} f(n)\,\operatorname{sinc}(t-n),
\qquad \operatorname{sinc}(t)=\frac{\sin \pi t}{\pi t}.
$$

Every sample contributes a sinc centered on itself, scaled by its value, and
the signal is their sum. Because each sinc is zero at every integer except
its own, the sum passes through every sample exactly, and because each sinc
contains only frequencies below half a cycle per sample, so does the sum.
The result is the sampling theorem, treated for optics in Goodman's
*Introduction to Fourier Optics*, section 2.4.1, and the condition that makes
it exact has a name: sampling at more than twice the highest frequency, the
Nyquist condition.

The sinchole library states that condition with one number, the oversampling factor
$\rho$, which is twice the highest frequency times the sample spacing. In the
units of this page the spacing is one, so $\rho=2\times0.38=0.76$ for the
test signal: the sample spacing is 76 percent of the largest spacing the
Nyquist condition allows, or equivalently 2.6 samples per cycle of the
fastest component. The Nyquist limit itself is $\rho=1$, two samples per
cycle.

## Past the Nyquist limit, the samples lie

The condition is not a formality. Below, a single cosine is sampled at three
spacings, and the ideal formula is applied to the samples each time. At
$\rho=0.5$ and $\rho=0.9$ the reconstruction follows the cosine. At
$\rho=1.4$ it confidently draws a different, slower cosine that passes
through exactly the same samples. Nothing in the samples reveals the
mistake, which is why sinchole never guesses the bandwidth of a delivered
grid and requires it to be declared instead.

```{code-cell} python
:tags: [hide-input]
fig, axes = plt.subplots(1, 3, figsize=(9, 2.4), sharey=True, constrained_layout=True)
t_dense = np.linspace(0, 12, 800)
for ax, rho in zip(axes, (0.5, 0.9, 1.4)):
    f0 = rho / 2
    n = np.arange(-200, 213)
    rec = np.sum(np.cos(2 * np.pi * f0 * n) * np.sinc(t_dense[:, None] - n), axis=1)
    ax.plot(t_dense, np.cos(2 * np.pi * f0 * t_dense), color=ep.style.neutral(0.45), lw=1.2)
    ax.plot(t_dense, rec, color=pal.pink, lw=1.8)
    ax.plot(n[(n >= 0) & (n <= 12)], np.cos(2 * np.pi * f0 * n[(n >= 0) & (n <= 12)]), "o",
            color=pal.cyan, ms=4)
    ax.set_title(rf"$\rho = {rho}$")
    ax.set_xlabel(r"$t$ (samples)")
axes[0].set_ylabel("signal")
axes[2].text(11.8, -1.25, "reconstruction", color=pal.pink, ha="right", fontsize=9)
plt.show()
```

At $\rho=1.4$ the pink reconstruction and the gray truth share every sample
and nothing else.

## What goes wrong when the ideal formula is cut at 32 samples?

The ideal formula needs every sample, and a delivered grid is finite. The
obvious repair keeps only the 32 samples nearest the query and drops the
rest.

```{admonition} Predict before reading on
:class: tip
Keep the 32 nearest samples of the ideal formula and drop the rest. For the
test signal, is the worst error near $10^{-2}$, $10^{-4}$ or $10^{-7}$ of the
signal's total amplitude?
```

```{dropdown} Answer
Near $10^{-2}$: the measured worst case is 0.9 to 1.5 percent of the summed
cosine amplitudes for every $\rho$ up to 0.76, rising to a few percent near
the Nyquist limit. The dropped samples begin 16 samples away, where a sinc
weight has magnitude up to $1/(16\pi)\approx0.02$, and their alternating
contributions do not cancel at the cut. The error is set mostly by that end
term, not by how slowly the signal varies.
```

The cure is to taper the sinc smoothly to zero at the edges of the stencil
instead of cutting it. The sinchole kernel multiplies the sinc by a Kaiser window, a
bell-shaped taper built from the modified Bessel function $I_0$ with a shape
parameter $\beta$, and uses half-width $W=16$ samples and $\beta=12$:

$$
w(t)=\operatorname{sinc}(t)\,\frac{I_0\!\left(\beta\sqrt{1-(t/W)^2}\right)}{I_0(\beta)}
\quad\text{for } |t|<W .
$$

The window leaves the central lobes nearly untouched and drives the far
weights smoothly to zero, so nothing jumps at the edge of the stencil.

```{code-cell} python
:tags: [hide-input]
def worst_error(kernel, rho, trials=40):
    """Worst reconstruction error over random band-limited signals."""
    worst = 0.0
    for seed in range(trials):
        r = np.random.default_rng(seed)
        f = r.uniform(-1, 1, 12) * rho / 2
        a = r.normal(size=12)
        p = r.uniform(0, 2 * np.pi, 12)
        sig = lambda t: np.sum(a * np.cos(2 * np.pi * f * np.asarray(t)[..., None] + p), -1)
        q = r.uniform(-0.5, 0.5, 50)
        taps = np.floor(q)[:, None] + np.arange(-15, 17)
        est = np.sum(kernel(q[:, None] - taps) * sig(taps), axis=1)
        worst = max(worst, np.max(np.abs(est - sig(q))) / np.abs(a).sum())
    return worst


kaiser = lambda t: np.asarray(weights(jnp.asarray(t)))
truncated = lambda t: np.where(np.abs(t) < 16, np.sinc(t), 0.0)
rhos = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.76, 0.8, 0.85, 0.9, 0.95])
err_kaiser = np.array([worst_error(kaiser, r) for r in rhos])
err_trunc = np.array([worst_error(truncated, r) for r in rhos])

fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.1), constrained_layout=True)
viz.plot_kernel(ax=left)
left.annotate("Kaiser-sinc", (2.5, 0.13), (6, 0.55), color=pal.cyan, fontsize=9,
              arrowprops={"arrowstyle": "-", "color": pal.cyan, "lw": 0.8})
left.annotate("ideal sinc", (13.5, 0.03), (8, -0.17), color=ep.style.neutral(0.65),
              fontsize=9, arrowprops={"arrowstyle": "-", "color": ep.style.neutral(0.5),
                                      "lw": 0.8})
right.semilogy(rhos, err_trunc, "o-", color=ep.style.neutral(0.55), lw=1.5)
right.semilogy(rhos, err_kaiser, "o-", color=pal.cyan, lw=2)
right.axvline(0.76, color=ep.style.neutral(0.35), lw=1, ls="--")
right.axvline(1.0, color=ep.style.neutral(0.35), lw=1)
right.text(0.765, 3e-1, "0.76", color=ep.style.neutral(0.6), fontsize=9)
right.text(0.985, 3e-1, "Nyquist", color=ep.style.neutral(0.6), fontsize=9, ha="right")
right.text(0.32, 2.5e-2, "truncated sinc", color=ep.style.neutral(0.65), fontsize=9)
right.text(0.32, 1.5e-6, "Kaiser-sinc", color=pal.cyan, fontsize=9)
right.set_xlabel(r"oversampling factor $\rho$")
right.set_ylabel("worst error / total amplitude")
right.set_ylim(1e-8, 1)
plt.show()
```

```{code-cell} python
:tags: [remove-cell]
glue_floor = err_kaiser[rhos <= 0.76].max()
glue_08 = err_kaiser[rhos == 0.8][0]
glue_09 = err_kaiser[rhos == 0.9][0]
print(f"{glue_floor:.1e} {glue_08:.1e} {glue_09:.1e}")
```

The right panel is the reason for sinchole's sampling rule. The windowed
kernel (cyan) holds the error below about $7\times10^{-7}$ of the signal's
total amplitude for every $\rho$ up to 0.76, then falls off a cliff:
$2\times10^{-4}$ at $\rho=0.8$ and $3\times10^{-2}$ at $\rho=0.9$. The
truncated sinc (gray) stays between about 1 and 4 percent across the whole
range. The sinchole reader therefore
admits a grid only when $\rho\le0.76$ on every axis, and treats
$\rho\ge1$ as a separate, harder failure.

A back-of-envelope estimate predicts both numbers: Kaiser's window design
rules give a floor near $1.3\times10^{-6}$ for $\beta=12$ and, for a 32-tap
kernel, an accurate response up to about 0.38 cycles per sample, which is
$\rho\approx0.75$.

```{dropdown} The estimate, worked out
Kaiser's first rule relates the shape parameter to the rejection the window
achieves, $\beta=0.1102(A-8.7)$, so $\beta=12$ gives $A\approx118$ dB, an
amplitude ratio near $1.3\times10^{-6}$: the size of the floor. His length
rule, $N-1=(A-8)/(2.285\,\Delta\omega)$, gives the width of the transition
from passing to rejecting frequencies for an $N=32$ tap kernel:
$\Delta\omega\approx1.55$ radians per sample, or 0.25 cycles per sample. The
flat part of the response therefore ends near $0.5-0.12=0.38$ cycles per
sample, $\rho\approx0.75$, where the measured cliff begins.

Kaiser derived both rules empirically for finite impulse response filter
design (Kaiser 1974, and see also Oppenheim and Schafer, *Discrete-Time Signal
Processing*, section 7.6). They apply here because interpolation at a
fractional offset is filtering with a shifted kernel: the passband ripple
$\delta\approx10^{-A/20}$ is both the error on in-band frequencies and the
leakage of the spectral images at $1-f$ that sampling creates. The rules are
approximations, and the measured curve above is the evidence.
```

## Two dimensions cost 1024 samples

A PSF grid is sampled in two source coordinates. The sinchole reader uses the product
of two one-dimensional kernels, $w(t_x)\,w(t_y)$, so every pixel of a read
combines a 32 by 32 block of 1024 delivered images, and the accuracy rule
applies separately along each axis. The cost of a read is therefore set by
1024 multiply-adds per detector pixel, independent of how large the grid is.

## A band of 1 cycle per unit sets a spacing of 0.38

Suppose a function of source position contains no frequency above $B=1$
cycle per unit, the case the next page derives for a coronagraph with an
entrance pupil one diameter wide. The Nyquist spacing is $1/(2B)=0.5$, and
the kernel's accuracy rule $2Bh\le0.76$ gives $h\le0.38$. A grid at that
spacing, read with the Kaiser-sinc kernel, reproduces such a function to
about one part in $10^6$ of its amplitude, while a grid at the Nyquist
spacing of 0.5 would sit past the cliff, with errors near 20 percent. For
the public LUVOIR-A coronagraph used on {doc}`../validation`, that spacing
is exactly the one the delivered grid uses.

The guarantee assumes the function really is band-limited, that the grid is
regular and complete, and that the query is at least 15 samples above the
first node and more than 15 samples below the last, so that its whole
stencil exists. Break the first and the error is no longer controlled, and if you break the
others then sinchole refuses the read.

## Try it

1. Change $\beta$ from 12 to 8 in the `weights` call (copy the formula above
   into a function). Predict first, using Kaiser's rule: how high does the
   floor rise, and does the cliff move left or right?
   ```{dropdown} Hint
   $A=8/0.1102+8.7\approx81$ dB, an amplitude near $9\times10^{-5}$. The
   transition narrows, so the cliff moves right.
   ```
2. Repeat the error curve with 16 taps instead of 32. Before running it,
   use the length rule to predict where the cliff starts.
   ```{dropdown} Hint
   Halving $N$ doubles the transition width, to about 0.5 cycles per sample,
   so the flat response ends near $0.5-0.25=0.25$ cycles per sample:
   $\rho\approx0.5$.
   ```
3. A source has frequencies up to $B=1.2$ cycles per unit. What spacing does
   the 0.76 rule require, and how many more grid points does a square grid of
   fixed size need than at $B=1$?
   ```{dropdown} Hint
   $h\le0.76/(2\times1.2)=0.317$. The node count per axis scales as $1/h$,
   so a square grid needs $1.2^2=1.44$ times as many nodes.
   ```

## What this page established

Samples of a band-limited function fix it between them, a Kaiser-windowed
sinc of 32 taps reconstructs it to about $10^{-6}$ when $\rho\le0.76$, and
the error grows by orders of magnitude just past that. The next page asks
the question this leaves open for a coronagraph: what is the highest
frequency in a planet image as the planet moves, and so what spacing does a
grid need?
