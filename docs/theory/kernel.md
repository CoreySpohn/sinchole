---
file_format: mystnb
kernelspec:
  name: python3
---

# How close to the Nyquist limit can a 32-tap kernel go?

Cutting the ideal sinc at 32 samples leaves an error near one percent, and a
planet photometry budget cannot spend that much on filling the gaps of a
grid. The sinchole library promises about one part in a million. This page
asks what the kernel does to earn that promise, how far toward the Nyquist
limit a grid may go before the promise breaks, and what one read costs when
the grid has two axes.

:::{admonition} Definitions
:class: note

- **Kaiser-sinc kernel.** The sinc multiplied by a Kaiser window, with
  half-width $W=16$ samples and shape parameter $\beta=12$. It is zero beyond
  16 samples, so a read uses 32 taps per axis.
- **Kaiser window.** A bell-shaped taper built from the modified Bessel
  function $I_0$. The shape parameter $\beta$ sets how fast it falls.
- **Separable.** A two-dimensional weight is the product $w(t_x)\,w(t_y)$ of
  two one-dimensional weights.
- **Margin.** The largest oversampling factor the kernel is admitted at,
  $\rho=0.76$.
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
from sinchole.kernel import stencil, weights

jax.config.update("jax_enable_x64", True)
hwostyle.use("dark")
pal = hwostyle.palette
```

## Does tapering the sinc remove the cut error?

Tapering multiplies the sinc by a smooth window that falls to zero at the
edge of the stencil, so nothing jumps where the formula is cut. The same
test as on the foundations page applies: random band-limited signals,
sampled at the integers and rebuilt at random offsets, with the worst error
measured as a fraction of the signal's total amplitude.

```{admonition} Predict before reading on
:class: tip
With the Kaiser taper in place and the signal at $\rho=0.5$, is the worst
error near $10^{-3}$, $10^{-5}$ or $10^{-7}$ of the amplitude?
```

```{dropdown} Answer
Near $10^{-7}$: about $4\times10^{-7}$, four orders of magnitude below the
cut sinc at the same $\rho$. The window leaves the central lobes nearly
untouched and drives the far weights smoothly to zero.
```

The kernel multiplies the sinc by the window,

$$
w(t)=\operatorname{sinc}(t)\,\frac{I_0\!\left(\beta\sqrt{1-(t/W)^2}\right)}{I_0(\beta)}
\quad\text{for } |t|<W,
$$

and is zero otherwise. In words: the window is one at the center and falls
to $1/I_0(\beta)$ at the edge, so the weights near the query are almost the
sinc's own and the weights near $t=\pm16$ are nearly zero.

## The error holds below 7e-7 until a cliff at the margin

The left panel draws the kernel against the ideal sinc. The right panel
measures the worst error against $\rho$ for the cut sinc and for the kernel.

```{code-cell} python
:tags: [hide-input]
def truncated(t):
    """The ideal sinc with every weight beyond 16 samples dropped."""
    return np.where(np.abs(t) < 16, np.sinc(t), 0.0)


rhos = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.76, 0.8, 0.85, 0.9, 0.95])
err_kaiser = np.array([tc.kernel_error(r) for r in rhos])
err_trunc = np.array([tc.kernel_error(r, kernel=truncated) for r in rhos])

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

The kernel (cyan) holds the error below about $7\times10^{-7}$ of the
signal's total amplitude for every $\rho$ up to 0.76, then falls off a cliff:
$2\times10^{-4}$ at $\rho=0.8$ and $3\times10^{-2}$ at $\rho=0.9$. The cut
sinc (gray) stays between about 1 and 4 percent across the whole range. The
sinchole reader therefore admits a grid only when $\rho\le0.76$ on every
axis, and treats $\rho\ge1$ as a separate, harder failure. This is the
margin, and every later page uses it.

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

## A two-dimensional read adds 1024 weighted samples

A grid of PSFs is sampled in two source coordinates, and the reader uses the
product of two one-dimensional kernels, $w(t_x)\,w(t_y)$. A read at one
detector pixel therefore adds a 32 by 32 block of $1024$ delivered values, each
multiplied by its weight, and the accuracy rule applies separately along each
axis. The cost of a read is 1024 multiply-adds per detector pixel,
independent of how large the grid is.

```{admonition} Predict before reading on
:class: tip
The 1024 taps are added in order of decreasing weight. How many are needed
before the partial sum stays within one part in a thousand of the final read:
about 30, about 300 or about 1000?
```

The animation counts the taps one at a time for a planet at
$(1.3, 0.7)\,\lambda_0/D$ and reads the probe pixel of the toy coronagraph
(the next page introduces it). On the left, each tap is a marker at its node,
with area proportional to the size of its weight and color for its sign. Taps
not yet counted are faint dots. On the right, the running sum of tap weight
times node value converges onto the final read.

```{code-cell} python
:tags: [hide-input]
# Medium gate, ground 4 (a sweep whose tracked mark cannot be matched across a
# strip), S1: one knob, the count k of taps added, with its value on screen.
# The tracked mark is the front of counted taps spreading outward from the
# query while the partial sum settles; 1024 states cannot be matched across a
# strip, and the claim is the rate of settling, which a still at one k cannot
# carry. The still below is the end state, k = 1024.
toy = tc.ToyCoronagraph()
x_pix = np.asarray(toy.detector_x)
col = int(np.argmin(np.abs(x_pix - 3.1)))
row = int(np.argmin(np.abs(x_pix - 0.6)))
query = np.array([1.3, 0.7])
grid = ((-6.08, -6.08), (0.38, 0.38), (40, 40))

tx, wx = stencil(jnp.asarray([query[0]]), grid[0][0], grid[1][0])
ty, wy = stencil(jnp.asarray([query[1]]), grid[0][1], grid[1][1])
node_x = grid[0][0] + np.asarray(tx[0]) * grid[1][0]
node_y = grid[0][1] + np.asarray(ty[0]) * grid[1][1]
tap_xy = np.stack(np.meshgrid(node_x, node_y), axis=-1).reshape(-1, 2)
tap_weight = np.outer(np.asarray(wy[0]), np.asarray(wx[0])).ravel()
tap_value = np.asarray(toy.psf(tap_xy))[:, row, col]

by_size = np.argsort(-np.abs(tap_weight), kind="stable")
partial = np.cumsum((tap_weight * tap_value)[by_size])
final = partial[-1]
off = np.abs(partial - final) > 1e-3 * abs(final)
settled = int(np.nonzero(off)[0].max()) + 2
counts = np.unique(np.round(np.geomspace(1, 1024, 36)).astype(int))
frames = np.concatenate([[0, 0], counts, np.full(6, 1024)])

fig, (left, right) = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True,
                                  gridspec_kw={"width_ratios": [1, 1.1]})
stencil_view = viz.plot_stencil_2d(query, grid, order="weight", n_taps=0,
                                   max_area=450.0, ax=left)
left.set_xlim(-5.5, 8.3)
left.set_ylim(-5.5, 8.3)
conv = ep.plot.convergence((tap_weight * tap_value)[by_size], running="sum",
                           refs=final, ref_labels=["read value"],
                           show_samples=False, ax=right)
right.set_xscale("log")
right.set_xlim(1, 1024 * 3)
right.set_xlabel(r"taps counted $k$, largest weight first")
right.set_ylabel("partial sum (probe pixel)")
count_text = right.text(0.03, 0.93, "", transform=right.transAxes, va="top")
left.set_title("1024 taps, area is weight", fontsize=10)


def draw(fig, k):
    k = int(k)
    stencil_view.update(n_taps=k)
    conv.update(max(k, 1) if k else 0)
    count_text.set_text(rf"$k$ = {k}" + (rf", sum / read = {partial[k - 1] / final:.4f}" if k else ""))


anim = ep.render.animate(fig, draw, frames, fps=10)
from IPython.display import HTML

HTML(anim.jshtml(dpi=72))
```

```{code-cell} python
:tags: [hide-input]
fig, (left, right) = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True,
                                  gridspec_kw={"width_ratios": [1, 1.1]})
viz.plot_stencil_2d(query, grid, order="weight", n_taps=1024, max_area=450.0, ax=left)
left.set_xlim(-5.5, 8.3)
left.set_ylim(-5.5, 8.3)
ep.plot.convergence((tap_weight * tap_value)[by_size], running="sum", refs=final,
                    ref_labels=["read value"], show_samples=False, ax=right)
right.set_xscale("log")
right.set_xlim(1, 1024 * 3)
right.set_xlabel(r"taps counted $k$, largest weight first")
right.set_ylabel("partial sum (probe pixel)")
plt.show()
print(f"the sum is within 1e-3 of the read from k = {settled} on; "
      f"largest weight {np.abs(tap_weight).max():.2f}, 1024th {np.abs(tap_weight).min():.1e}")
```

```{dropdown} Answer
About 300: in this run the sum stays within $10^{-3}$ of the read from
$k=337$ on, and it is nowhere near that before $k=30$. The heaviest taps are
the central lobe and its first sidelobes, and their signs alternate, so the
partial sum overshoots the read by a factor of several and swings through
zero before the outer taps pull it in.
```

The still is the end state, $k=1024$. Taps are ordered by decreasing weight
so that the convergence is the lesson. The largest weight is 0.70 and the
smallest is $4\times10^{-12}$, yet the large taps alone do not give the
answer, because positive and negative lobes nearly cancel and the outer taps
supply the correction. The marker areas are drawn to scale, so only the
largest weights are visible, and the cancellation is the reason the kernel
needs all 32 taps per axis.

## A band of 1 cycle per unit sets a spacing of 0.38

Suppose a function of source position contains no frequency above $B=1$
cycle per unit, the case the next page derives for a coronagraph with an
entrance pupil one diameter wide. The Nyquist spacing is $1/(2B)=0.5$, and
the margin rule $2Bh\le0.76$ gives $h\le0.38$. A grid at that spacing, read
with the kernel, reproduces such a function to about one part in $10^6$ of
its amplitude, while a grid at the Nyquist spacing of 0.5 would sit past the
cliff, with errors near 20 percent.

The guarantee assumes the function really is band-limited, that the grid is
regular and complete, and that the query is at least 15 samples above the
first node and more than 15 samples below the last, so that its whole
stencil exists. Break the first and the error is no longer controlled. If you
break the others, the library refuses the read.

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
4. Predict, then change `query` in the animation cell to $(1.3, 0.0)$, a
   planet exactly on a row of nodes. How many taps carry a nonzero weight?
   ```{dropdown} Hint
   The weight $w(t_y)$ is zero at every integer but one, so one row of 32
   taps carries the read and the other rows weigh about $10^{-16}$.
   ```

## What this page established

A 32-tap Kaiser-sinc kernel reconstructs a band-limited function to about
$10^{-6}$ when $\rho\le0.76$, and the error grows by orders of magnitude just
past that margin. A read on a two-dimensional grid adds 1024 taps whose signs alternate,
so it settles only after a few hundred of them. The next page asks the question this leaves
open for a coronagraph: what is the highest frequency in a planet image as
the planet moves, and so what spacing does a grid need?

Next: {doc}`band-limits`.
