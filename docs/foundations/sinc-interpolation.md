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
reconstruction works, and what goes wrong when it is used with a finite
number of samples. It is the general sampling theory that the rest of this
documentation builds on.

:::{admonition} Definitions
:class: note

- **Sample spacing.** The distance between neighboring samples. This page
  measures position $t$ in units of it, so the spacing is 1.
- **Bandwidth.** The highest frequency $f_{\max}$ the signal contains, in
  cycles per sample.
- **Nyquist limit.** Two samples per cycle of the highest frequency, which is
  $f_{\max}=1/2$.
- **Oversampling factor.** $\rho=2f_{\max}$, the fraction of the Nyquist limit
  the signal uses. Nyquist is $\rho=1$.
- **Weight, tap and stencil.** A reconstruction at one query point adds up
  nearby samples. Each sample used is a tap, its multiplier is its weight,
  and the set of taps is the stencil.
- **Alias.** A slower signal that passes through exactly the same samples as
  a faster one.
:::

```{code-cell} python
:tags: [remove-cell]
import hwostyle
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import eyepiece as ep
import toy_coronagraph as tc
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
little.
```

A signal sampled at more than two samples per cycle of its fastest component
is fixed between its samples, and every sample contributes to every value,
with an influence that falls off only as one over the distance. That slow
falloff is why a practical reconstruction needs a stencil of finite width,
and why the weights on its edges decide how good the result is.

## Each sample spreads as a sinc

A practical reconstruction at a query point combines only a fixed set of
nearby samples, its stencil, and each sample in the stencil (a tap) gets a
weight. The animation follows one query point as it slides across three
sample cells. The bottom panel shows the 32 weights of a practical kernel
(the next page opens it) on the nearest samples, and the top panel, zoomed
in around the query, draws the reconstruction, their weighted sum, as the
query moves.

Watch the moment the query lands on a sample: all
weights but one drop to zero, the reconstruction passes exactly through that
sample, and the set of 32 samples in use shifts by one.

```{code-cell} python
:tags: [hide-input]
# Medium gate, ground 4 (a sweep whose tracked mark cannot be matched across a
# strip): the knob is the query position, value on screen; the tracked mark is
# the weight pattern, and the mid-sweep event is the node crossing, where all
# weights collapse to a single 1 and the stencil jumps by one index.
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
nearest samples, then alternating lobes on either side that taper smoothly
to zero by the ends of the 32-sample stencil.

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

The oversampling factor $\rho$ states that condition with one number. For the
test signal $\rho=2\times0.38=0.76$: the sample spacing is 76 percent of the
largest spacing the Nyquist condition allows, or equivalently 2.6 samples per
cycle of the fastest component. The Nyquist limit itself is $\rho=1$, two
samples per cycle.

## Past the Nyquist limit, the samples lie

The condition is not a formality. Below, a single cosine is sampled at three
spacings, and the ideal formula is applied to the samples each time. At
$\rho=0.5$ and $\rho=0.9$ the reconstruction follows the cosine. At
$\rho=1.4$ it confidently draws a different, slower cosine that passes
through exactly the same samples. Nothing in the samples reveals the
mistake, which is why the bandwidth of a sampled signal has to be known
rather than guessed from the samples.

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

The ideal formula needs every sample, and any real data set is finite. The
obvious repair keeps only the 32 samples nearest the query and drops the
rest.

```{admonition} Predict before reading on
:class: tip
Keep the 32 nearest samples of the ideal formula and drop the rest. For the
test signal, is the worst error near $10^{-2}$, $10^{-4}$ or $10^{-7}$ of the
signal's total amplitude?
```

```{code-cell} python
:tags: [hide-input]
def truncated(t):
    """The ideal sinc with every weight beyond 16 samples dropped."""
    return np.where(np.abs(t) < 16, np.sinc(t), 0.0)


for rho in (0.3, 0.5, 0.76, 0.9):
    worst = tc.kernel_error_worst(rho, kernel=truncated)
    print(f"rho = {rho}: worst single-tone error {worst:.1e} of the amplitude")
```

```{dropdown} Answer
Near $10^{-2}$: the worst single cosine, over every frequency up to the band
edge and every query offset in a cell, is off by 2 to 5 percent of its
amplitude for every $\rho$ from 0.3 to 0.76 (see the printed values), and by
nearly 10 percent at $\rho=0.9$. The dropped samples begin 16 samples away, where a sinc
weight has magnitude up to $1/(16\pi)\approx0.02$, and their alternating
contributions do not cancel at the cut. The error is set mostly by that end
term, not by how slowly the signal varies.
```

The cut leaves an error of a few percent however slow the signal is, so
the weights at the edge of the stencil are the problem. The cure is to taper
the sinc smoothly to zero there instead of cutting it. Which taper, and how
close to the Nyquist limit it lets a grid go, is the subject of
{doc}`../theory/kernel`.

## Try it

1. Complete the calculation. A signal has a highest frequency of 0.3 cycles
   per sample. What is $\rho$, and how many samples per cycle does its
   fastest component get?
   ```{dropdown} Hint
   $\rho=2\times0.3=0.6$, and the fastest component has a period of
   $1/0.3=3.3$ samples.
   ```
2. At $t=0.3$, what is the exact weight of the sample at $n=-20$? Predict
   from the one-over-distance falloff before using the formula.
   ```{dropdown} Hint
   The distance is $20.3$, so the falloff alone gives $1/(20.3\pi)=0.016$,
   and the sine factor $\sin(0.3\pi)=0.81$ makes it $0.013$.
   ```
3. Predict, then change the tuple `(0.5, 0.9, 1.4)` in the aliasing cell to
   `(0.5, 0.9, 1.7)`. Which slower cosine does the reconstruction draw at
   $\rho=1.7$?
   ```{dropdown} Hint
   Samples at the integers cannot tell a frequency $f$ from $f-1$. A
   cosine at $f=0.85$ looks like one at $0.15$ cycles per sample, a period of
   about 6.7 samples.
   ```

## What this page established

Samples of a band-limited function fix it between them when it is sampled at
$\rho<1$, the exact reconstruction needs every sample, and cutting it to 32
samples leaves a worst-case error of a few percent. The next page tapers the cut and
measures how far toward the Nyquist limit that lets a grid go.

Next: {doc}`../theory/kernel`.
