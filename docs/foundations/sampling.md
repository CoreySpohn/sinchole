---
file_format: mystnb
kernelspec:
  name: python3
---

# When do samples pin down a signal?

sinchole computes coronagraph images for planet positions that lie between
the positions where images were precomputed. Seen from one detector pixel,
that is interpolation: the pixel's brightness is a function of the planet's
position, the precomputed images are samples of that function on a grid, and
every value a user asks for lies between samples. Everything sinchole reports
therefore rests on one question: what has to be true, and what has to be
done, for the values between samples to come out exactly right?

:::{admonition} The question of these two pages
:class: important

What does perfect interpolation need? This page finds the first two
requirements, which are about the signal and the samples. The
{doc}`next page <sinc-interpolation>` finds the third, which is about how the
samples are combined, and then asks what is lost when it cannot be met in
full.
:::

```{code-cell} python
:tags: [remove-cell]
import tempfile
from pathlib import Path

import hwostyle
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Video

import eyepiece as ep

hwostyle.use("dark")
pal = hwostyle.palette
FAINT = ep.style.neutral(0.25)
SCENERY = ep.style.neutral(0.45)

_clips = iter(range(100))


def clip(anim):
    """Embed an animation as an mp4 with the browser's own player controls."""
    path = Path(tempfile.gettempdir()) / f"sinchole-docs-sampling-{next(_clips)}.mp4"
    anim.video(path, dpi=200)
    return Video(str(path), embed=True, html_attributes="controls loop muted playsinline")

F_MAX = 0.38
rng = np.random.default_rng(3)
freqs = rng.uniform(-F_MAX, F_MAX, 12)
amps = rng.normal(size=12)
phases = rng.uniform(0, 2 * np.pi, 12)


def component(k, t):
    """One cosine of the test signal."""
    return amps[k] * np.cos(2 * np.pi * freqs[k] * np.asarray(t, dtype=float) + phases[k])


def signal(t):
    """The test signal: the sum of its twelve cosines."""
    return sum(component(k, t) for k in range(12))
```

## Can the samples alone decide what lies between them?

Start with twelve samples of a smooth signal, taken one unit apart. The
figure shows three different curves that pass through exactly the same
twelve points: a slow, smooth curve, straight lines from point to point, and
a curve that swings up and down once between every pair of samples.

```{code-cell} python
:tags: [hide-input]
n12 = np.arange(0, 12)
t_fine = np.linspace(0, 11, 1200)
curves = {
    "a slow curve": signal(t_fine),
    "straight lines": np.interp(t_fine, n12, signal(n12)),
    "a fast curve": signal(t_fine) + 1.2 * np.sin(2 * np.pi * t_fine),
}
fig, axes = plt.subplots(1, 3, figsize=(9, 2.4), sharey=True, constrained_layout=True)
for ax, (name, curve), color in zip(axes, curves.items(), (pal.pink, pal.yellow, pal.green)):
    ax.plot(t_fine, curve, color=color, lw=1.6)
    ax.plot(n12, signal(n12), "o", color=pal.cyan, ms=4.5)
    ax.set_title(name, color=color)
    ax.set_xlabel(r"position $t$ (samples)")
axes[0].set_ylabel("value")
plt.show()
```

No. All three curves fit the twelve samples perfectly, and so do infinitely
many others, so the samples cannot tell us which curve is the real one. To
pick one we need to know something about the signal itself. What separates
the three curves is how rapidly each one oscillates: the slow curve rises
and falls over several samples, while the fast curve rises and falls between
every pair of samples. The rest of this page makes "how rapidly a signal can
oscillate" precise and shows that knowing it, together with close enough
samples, picks out a single curve.

## A cosine repeats at one steady rate

The simplest signal that oscillates at one definite rate is a cosine. Its
**frequency** counts how many full cycles it completes per unit of position.
Positions on these pages are measured in units of the sample spacing, so
frequency comes out in cycles per sample.

```{code-cell} python
:tags: [hide-input]
fig, axes = plt.subplots(1, 3, figsize=(9, 2.5), sharey=True, constrained_layout=True)
t_c = np.linspace(0, 12, 1200)
n_c = np.arange(0, 13)
for ax, f in zip(axes, (0.1, 0.25, 0.4)):
    ax.plot(t_c, np.cos(2 * np.pi * f * t_c), color=pal.cyan, lw=1.6)
    ax.plot(n_c, np.cos(2 * np.pi * f * n_c), "o", color=pal.cyan, ms=3.5)
    ax.set_title(f"{f} cycles per sample", fontsize=11)
    ax.set_xlabel(r"position $t$ (samples)")
period = 1 / 0.25
axes[1].annotate("", xy=(period, 1.25), xytext=(0, 1.25),
                 arrowprops=dict(arrowstyle="<->", color=SCENERY, lw=1))
axes[1].text(period + 0.3, 1.25, "one cycle = 4 samples", color=SCENERY, va="center",
             fontsize=9)
axes[0].set_ylabel("value")
axes[0].set_ylim(-1.2, 1.5)
plt.show()
```

The middle cosine completes one cycle every 4 samples, so its frequency is
$1/4=0.25$ cycles per sample. The left one needs 10 samples per cycle (0.1
cycles per sample), the right one only 2.5 (0.4 cycles per sample). A
frequency and its number of samples per cycle are reciprocals.

"Fast" on these pages means a high frequency: many turns per sample. It does
not mean steep. A tall cosine at 0.1 cycles per sample can have a steeper
slope than a small one at 0.4, but its shape still only changes over a
distance of several samples, and that distance, not the slope, is what
decides whether samples can capture it.

## Adding cosines builds other shapes

A cosine alone is a very particular shape, but adding several cosines with
different frequencies, sizes and starting points (**phases**) builds new
shapes. The animation builds the slow curve of the first figure this way,
one cosine at a time, starting from the slowest. The top panel shows the
cosine being added at each step (yellow), and the panel below shows the sum
of every cosine added so far (pink), with the target curve faint behind it.
Both use the same vertical scale, so the sizes compare directly. On the
right, each cosine appears as a stem whose position is its frequency and
whose height is its size: the step's cosine in yellow, earlier ones in cyan.

```{code-cell} python
:tags: [hide-input]
# Medium gate, ground 4 (a sweep whose tracked mark cannot be matched across a
# strip), S1: one knob, the number of cosines added, value on screen. The
# tracked mark is the running sum taking shape, linked to the stem that each
# added cosine lights; twelve paired states cannot be matched across a strip.
by_freq = np.argsort(np.abs(freqs))
t_b = np.linspace(0, 11, 700)
partials = np.cumsum([component(k, t_b) for k in by_freq], axis=0)
Y_LIM = (signal(t_b).min() - 0.8, signal(t_b).max() + 1.0)


def buildup_axes():
    fig, axd = plt.subplot_mosaic(
        [["new", "spec"], ["sum", "spec"]], figsize=(9, 3.9), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.5, 1]},
    )
    for name in ("new", "sum"):
        ax = axd[name]
        ax.axhline(0, color=FAINT, lw=0.8)
        ax.set_xlim(0, 11)
        ax.set_ylim(*Y_LIM)
        ax.set_ylabel("value")
    axd["new"].tick_params(labelbottom=False)
    axd["sum"].plot(t_b, signal(t_b), color=FAINT, lw=3)
    axd["sum"].set_xlabel(r"position $t$ (samples)")
    axd["sum"].set_title("running sum", color=pal.pink, fontsize=11)
    spec = axd["spec"]
    spec.vlines(np.abs(freqs), 0, np.abs(amps), color=FAINT, lw=2)
    spec.set_xlim(0, 0.5)
    spec.set_ylim(0, 2.3)
    spec.set_xlabel("frequency (cycles per sample)")
    spec.set_ylabel("size")
    spec.set_title("the recipe (sizes only)", fontsize=11)
    return fig, axd


fig, axd = buildup_axes()
(newest,) = axd["new"].plot([], [], color=pal.yellow, lw=1.8)
new_title = axd["new"].set_title("", color=pal.yellow, fontsize=11)
(running,) = axd["sum"].plot([], [], color=pal.pink, lw=2)
earlier = axd["spec"].vlines([], [], [], color=pal.cyan, lw=2.4)
current = axd["spec"].vlines([], [], [], color=pal.yellow, lw=3)


def stem(j):
    return [(abs(freqs[j]), 0), (abs(freqs[j]), abs(amps[j]))]


def draw(fig, k):
    j = by_freq[k - 1]
    newest.set_data(t_b, component(j, t_b))
    new_title.set_text(f"cosine {k} of 12 being added: {abs(freqs[j]):.2f} cycles per sample")
    running.set_data(t_b, partials[k - 1])
    earlier.set_segments([stem(i) for i in by_freq[: k - 1]])
    current.set_segments([stem(j)])


frames = [1, 1] + [k for k in range(1, 13) for _ in range(2)] + [12] * 3
anim = ep.render.animate(fig, draw, frames, fps=2)
clip(anim)
```

```{code-cell} python
:tags: [hide-input]
fig, axd = buildup_axes()
for k in range(12):
    axd["new"].plot(t_b, component(k, t_b), color=pal.cyan, lw=1, alpha=0.7)
axd["new"].set_title("all twelve cosines", color=pal.cyan, fontsize=11)
axd["sum"].plot(t_b, partials[-1], color=pal.pink, lw=2)
axd["sum"].set_title("their sum", color=pal.pink, fontsize=11)
axd["spec"].vlines(np.abs(freqs), 0, np.abs(amps), color=pal.cyan, lw=2.4)
plt.show()
```

With all twelve cosines in, the pink sum lies on the slow curve. The still
shows the twelve cosines together on top, in the same cyan as their stems,
and their sum below. The list of frequencies and sizes is the curve's **spectrum**.
The stems show sizes only: each cosine also has a phase, which a complete
recipe lists too but which these pages never need.

:::{admonition} Known result: Fourier decomposition
:class: seealso

Any signal of finite energy can be built as a sum of cosines in this way,
using a continuous range of frequencies when the signal does not repeat, and
the recipe of sizes and phases is unique. This is Fourier analysis
(Goodman, *Introduction to Fourier Optics*, section 2.1). These pages use it
without proof.
:::

## A band limit caps how rapidly the signal can oscillate

Because the signal is a sum of cosines, it can have no feature narrower than
its fastest cosine allows. The right panel below is the slow curve's
spectrum again, with a line at 0.38 cycles per sample. None of its cosines
lies above the line, because the test signal was built that way. A frequency
above which a signal is known to contain nothing is its **bandwidth**
$f_{\max}$, and a signal with nothing above its bandwidth is
**band-limited**. The bandwidth is a bound, so the fastest cosine actually
present, here near 0.31, may sit below it.

```{code-cell} python
:tags: [hide-input]
fig, (left, right) = plt.subplots(
    1, 2, figsize=(9, 2.8), constrained_layout=True, gridspec_kw={"width_ratios": [1.2, 1]}
)
t_e = np.linspace(0, 11, 1100)
left.axhline(0, color=FAINT, lw=0.8)
left.plot(t_e, 1.2 * np.sin(2 * np.pi * t_e), color=pal.green, lw=1.5)
left.plot(n12, np.zeros_like(n12, dtype=float), "o", color=pal.cyan, ms=4.5)
left.set_xlim(0, 11)
left.set_xlabel(r"position $t$ (samples)")
left.set_ylabel("value")
left.set_title("the fast curve minus the slow curve", fontsize=11)

right.vlines(np.abs(freqs), 0, np.abs(amps), color=pal.cyan, lw=2.2)
right.vlines([1.0], 0, [1.2], color=pal.green, lw=2.2)
right.axvline(F_MAX, color=pal.pink, lw=1.4)
right.text(F_MAX + 0.02, 2.05, r"$f_{\max}=0.38$", color=pal.pink, fontsize=10)
right.text(0.97, 1.3, "fast curve's\nextra cosine", color=pal.green, ha="right", fontsize=9)
right.set_xlim(0, 1.1)
right.set_ylim(0, 2.3)
right.set_xlabel("frequency (cycles per sample)")
right.set_ylabel("size")
right.set_title("spectrum", fontsize=11)
plt.show()
```

The left panel is the fast curve minus the slow curve. It is a single wave
at 1 cycle per sample, and every sample (cyan) lands on one of its zero
crossings, which is why both curves fit the same samples. On the right that
extra wave (green) sits far above the bandwidth, so knowing that the signal
is band-limited at 0.38 rules the fast curve out.

The first requirement is to **know that the signal is band-limited, and know
its bandwidth**. This knowledge has to come from where the signal comes from,
because the samples cannot show it: the extra wave is zero at every one of
them. For a coronagraph, the bandwidth follows from the size of the telescope
pupil, which {doc}`../theory/band-limits` derives.

:::{admonition} Why 0.38?
:class: note

Any bandwidth below 0.5 would serve these two pages. They use 0.38 because
it is the bandwidth that sinchole's own interpolation is designed for, as
{doc}`../theory/kernel` explains, so the numbers here carry over. A cosine at
0.38 cycles per sample completes one cycle every $1/0.38=2.6$ samples.
:::

## What happens when the samples are too far apart?

Knowing the bandwidth is not the end of the story. To see why, look at the
samples of a single cosine at 0.7 cycles per sample, a cosine that completes
a cycle every 1.4 samples, with the cosine itself hidden.

```{code-cell} python
:tags: [hide-input]
n_a = np.arange(0, 11)
fig, ax = plt.subplots(figsize=(7.5, 2.2), constrained_layout=True)
ax.axhline(0, color=FAINT, lw=0.8)
ax.plot(n_a, np.cos(2 * np.pi * 0.7 * n_a), "o", color=pal.cyan, ms=6)
ax.set_xlim(-0.3, 10.3)
ax.set_ylim(-1.3, 1.3)
ax.set_xlabel(r"position $t$ (samples)")
ax.set_ylabel("value")
ax.set_title("samples of a cosine at 0.7 cycles per sample", fontsize=11)
plt.show()
```

```{admonition} Predict before reading on
:class: tip
Join the dots by eye with the smoothest curve you can. Does it look like a
cosine faster than 0.7 cycles per sample, the same, or slower? Roughly how
many samples does your curve take per cycle?
```

In the animation the cosine (gray) speeds up from 0.05 to 0.95 cycles per
sample while the samples (cyan) stay one unit apart. The pink curve is the
slowest cosine that passes through every sample. Watch the moment the
frequency passes 0.5: the pink curve leaves the gray one and starts to slow
down while the true cosine keeps speeding up.

```{code-cell} python
:tags: [hide-input]
# Medium gate, ground 4, S3: one knob, the frequency f, value on screen. The
# tracked mark is the slowest cosine through the samples; the mid-sweep event
# is f = 0.5, where it separates from the true cosine and its frequency
# reverses direction, a sense reversal a strip cannot carry. Still below.
sweep = np.concatenate([np.full(2, 0.05), np.linspace(0.05, 0.95, 37), np.full(4, 0.95)])
t_a = np.linspace(0, 10, 1200)
fig, ax = plt.subplots(figsize=(7.5, 2.8), constrained_layout=True)
(true_line,) = ax.plot([], [], color=SCENERY, lw=1.1)
(alias_line,) = ax.plot([], [], color=pal.pink, lw=2)
(dots,) = ax.plot([], [], "o", color=pal.cyan, ms=5)
ax.set_xlim(0, 10)
ax.set_ylim(-1.3, 1.7)
ax.set_xlabel(r"position $t$ (samples)")
ax.set_ylabel("value")
readout = ax.text(0.01, 0.97, "", transform=ax.transAxes, va="top", fontsize=10)


def draw(fig, f):
    seen = min(f, 1 - f)
    true_line.set_data(t_a, np.cos(2 * np.pi * f * t_a))
    alias_line.set_data(t_a, np.cos(2 * np.pi * seen * t_a))
    dots.set_data(n_a, np.cos(2 * np.pi * f * n_a))
    readout.set_text(f"true cosine: {f:.2f} cycles/sample    samples look like: {seen:.2f}")


anim = ep.render.animate(fig, draw, sweep, fps=6)
clip(anim)
```

```{code-cell} python
:tags: [hide-input]
fig, axes = plt.subplots(1, 4, figsize=(10, 2.4), sharey=True, constrained_layout=True)
for ax, f in zip(axes, (0.25, 0.45, 0.55, 0.7)):
    seen = min(f, 1 - f)
    ax.plot(t_a, np.cos(2 * np.pi * f * t_a), color=SCENERY, lw=1.1)
    ax.plot(t_a, np.cos(2 * np.pi * seen * t_a), color=pal.pink, lw=1.8)
    ax.plot(n_a, np.cos(2 * np.pi * f * n_a), "o", color=pal.cyan, ms=4)
    ax.set_title(f"true {f}, looks like {seen:.2f}", fontsize=10)
    ax.set_xlabel(r"$t$ (samples)")
axes[0].set_ylabel("value")
plt.show()
```

```{dropdown} Answer to the prediction
Slower: about 3.3 samples per cycle, a cosine at $1-0.7=0.3$ cycles per
sample (right panel of the still). The samples of the fast cosine are exactly
the samples of the slow one.
```

Below 0.5 cycles per sample the slowest cosine through the samples is the
true one, so the pink and gray curves coincide. Between 0.5 and 1 a slower
cosine, at $1-f$, fits the same samples. The reason is a property of cosines
at whole-number positions. For any integer $n$,

$$
\cos\!\big(2\pi(1-f)\,n\big)=\cos(2\pi n-2\pi f n)=\cos(-2\pi f n)=\cos(2\pi f n).
$$

The first step only expands the bracket. The second drops $2\pi n$, a whole
number of turns, which does not change a cosine. The third uses the symmetry
of the cosine, $\cos(-x)=\cos x$. In words: at the sample positions, a
cosine at $f$ and a cosine at $1-f$ take the same values, where 1 is the
sampling rate of one sample per unit. Seeing a fast cosine as a slow one is
called **aliasing**, and nothing in the samples reveals it.

The fast curve of the first figure is the same effect one step further: its
extra wave at 1 cycle per sample looks, at the samples, like a wave at
$1-1=0$ cycles per sample, and since it is zero at every sample it adds
nothing the samples can see. Above 1 cycle per sample the pattern repeats,
and the signal shifts in phase as well as frequency, but the demonstrations
here need only the range up to 1.

## The Nyquist limit: more than two samples per cycle

The two frequencies $f$ and $1-f$ are equal at $f=0.5$, two samples per
cycle, so 0.5 cycles per sample is the dividing line, the **Nyquist limit**.
A cosine below it is always the slowest cosine through its own samples, so
it cannot be mistaken for anything slower. The second requirement is to
**place the samples close enough that the bandwidth stays below the Nyquist
limit**.

One number says how much of that allowance a grid uses, the **oversampling
factor** $\rho=2f_{\max}$, the bandwidth as a fraction of the Nyquist limit.
Despite its name, a smaller $\rho$ means more samples per cycle than
needed, and $\rho=1$ is the limit itself. The test signal has
$\rho=2\times0.38=0.76$: at least 2.6 samples per cycle of any cosine it
can contain.

:::{admonition} Known result: the sampling theorem
:class: seealso

Among signals with no cosine at or above 0.5 cycles per sample, the samples
determine the signal completely, so whenever $\rho<1$ exactly one such curve
passes through them. The inequality must be strict: a cosine at exactly 0.5
cycles per sample can be sampled at its zero crossings and vanish from the
samples. The result is credited to Whittaker (1915), Nyquist (1928),
Kotelnikov (1933) and Shannon (1949); Goodman's *Introduction to Fourier
Optics*, section 2.4, gives the two-dimensional version used in optics. The
{doc}`next page <sinc-interpolation>` shows how the values between samples
are computed, which is also how the theorem is proved.
:::

These requirements assume noiseless samples on an evenly spaced grid. Energy
above the bandwidth, however small, would alias into the samples, and an
uneven grid would break the whole-number argument above.

## Try it

sinchole makes the same two checks for every delivered grid, once along each
axis: the bandwidth comes from the telescope pupil, and the grid spacing then
sets $\rho$. The exercises practice those checks.

1. A signal has a bandwidth of 0.3 cycles per sample. What is $\rho$, and how
   many samples per cycle does a cosine at the bandwidth get?
   ```{dropdown} Hint
   Use $\rho=2f_{\max}$, and samples per cycle $=1/f$.
   ```
   ```{dropdown} Answer
   $\rho=0.6$, and $1/0.3=3.3$ samples per cycle.
   ```
2. Predict, then change the tuple `(0.25, 0.45, 0.55, 0.7)` in the still's
   cell to end with `0.85`. Which slower cosine passes through the samples?
   ```{dropdown} Hint
   At the sample positions a cosine at $f$ matches one at $1-f$.
   ```
   ```{dropdown} Answer
   0.15 cycles per sample, a cosine that repeats every 6.7 samples.
   ```
3. A grid has $\rho=0.5$. How many samples does a cosine at the bandwidth get
   per cycle, and is the grid closer to or further from aliasing than the
   test signal's?
   ```{dropdown} Hint
   First recover $f_{\max}$ from $\rho$.
   ```
   ```{dropdown} Answer
   $f_{\max}=0.25$, so 4 samples per cycle, further from aliasing than the
   test signal's 2.6.
   ```

## What this page established

:::{admonition} What perfect interpolation needs, so far
:class: important

1. **A band limit.** The signal has no cosine above a known bandwidth
   $f_{\max}$.
2. **Dense enough samples.** More than two samples per cycle at the
   bandwidth, $\rho=2f_{\max}<1$.
3. *To be found on the next page: how to combine the samples.*
:::

With both requirements met, exactly one curve with no cosine at or above
0.5 cycles per sample passes through the samples. The next page builds the
procedure that finds it, starting from the single shape that each sample
contributes.

Next: {doc}`sinc-interpolation`.
