---
file_format: mystnb
kernelspec:
  name: python3
---

# Reproducible stencils

A read gathers 32 nodes per axis starting from `floor(t) - 15`, with
$t=(s-o)/h$, and whether those nodes exist decides whether the read is
served or refused. If two ways of running the same code (eager Python,
`jax.jit`, `jax.vmap`) computed different floors for the same query, the
same read could be accepted in one and refused in another, and receipts
would not be reproducible. This page shows that the obvious formula does
exactly that, and what sinchole does instead.

```{code-cell} python
:tags: [remove-cell]
import math
from fractions import Fraction

import hwostyle
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from sinchole.kernel import exact_quotient_floor_v1

jax.config.update("jax_enable_x64", True)
hwostyle.use("dark")
pal = hwostyle.palette
```

```{admonition} Predict before reading on
:class: tip
With spacing `h = 0.38`, what is `math.floor(1.9 / h)`? The literal 1.9 is
the double nearest $5\times0.38$.
```

```{dropdown} Answer
Python returns 5. The exact quotient of the two stored doubles is just below
5, so its floor is 4: the double nearest 1.9 is slightly smaller than five
times the double nearest 0.38. Python's division is correctly rounded, but
the correctly rounded quotient lands on the integer 5.
```

```{code-cell} python
h = 0.38
for q in (1.9, 1.14):
    exact = math.floor(Fraction(q) / Fraction(h))
    python = math.floor(q / h)
    jitted = int(jax.jit(lambda a: jnp.floor(a / h))(q))
    rule = int(jax.jit(lambda a: exact_quotient_floor_v1(a, 0.0, h))(q))
    print(f"q = {q}: exact {exact}, Python {python}, jax.jit {jitted}, sinchole {rule}")
```

Two different failures appear. At 1.9 both Python and `jax.jit` round the
quotient up to an integer the exact quotient does not reach. At 1.14, Python
gets the exact answer and `jax.jit` does not, because compiled code on CPU
replaces division by a constant with multiplication by its reciprocal, which
rounds twice. The second failure is the dangerous one, because it makes the
answer depend on how the code was run.

## Which method puts each query in the right cell?

The figure walks through the doubles nearest the cell edge at
$3\times0.38$, one unit in the last place (ULP) at a time, and marks the floor
each method computes. Exact arithmetic switches from 2 to 3 at one double, while
the naive formulas switch at others, and not at the same one.

```{code-cell} python
:tags: [hide-input]
edge = 3 * h
queries = [edge]
for _ in range(6):
    queries.insert(0, math.nextafter(queries[0], -math.inf))
    queries.append(math.nextafter(queries[-1], math.inf))
offsets = np.arange(-6, 7)
rows = {
    "exact": [math.floor(Fraction(q) / Fraction(h)) for q in queries],
    "Python q / h": [math.floor(q / h) for q in queries],
    "jax.jit q / h": list(np.asarray(jax.jit(lambda a: jnp.floor(a / h))(jnp.asarray(queries)))),
    "sinchole, jax.jit": list(np.asarray(
        jax.jit(lambda a: exact_quotient_floor_v1(a, 0.0, h))(jnp.asarray(queries)))),
}
fig, ax = plt.subplots(figsize=(8, 2.6), constrained_layout=True)
for y, (name, values) in enumerate(rows.items()):
    values = np.asarray(values, dtype=int)
    ax.scatter(offsets[values == 2], np.full((values == 2).sum(), y), color=pal.cyan, s=40)
    ax.scatter(offsets[values == 3], np.full((values == 3).sum(), y), color=pal.pink, s=40,
               marker="s")
ax.set_yticks(range(len(rows)), list(rows))
ax.invert_yaxis()
ax.set_xlabel("ULPs from the double nearest 3 x 0.38")
ax.set_ylim(3.5, -1.1)
ax.text(-6, -0.75, "cell 2", color=pal.cyan, ha="left", fontsize=9)
ax.text(6, -0.75, "cell 3", color=pal.pink, ha="right", fontsize=9)
plt.show()
```

Cyan dots are queries assigned to cell 2, pink squares to cell 3. The sinchole
row matches the exact row at every double, in eager and in compiled code.

## The remainder gives the exact floor

The sinchole reader computes the floor from the remainder instead of the quotient. It
forms $d=s-o$ once, takes the IEEE remainder $r=\operatorname{fmod}(d,h)$,
which is computed exactly, recovers the integer quotient as
$\operatorname{round}((d-r)/h)$ (a division whose exact result is an integer
far below $2^{53}$, so rounding cannot move it), and subtracts one when $d$
is negative with a nonzero remainder. The result is the floor of the exact
quotient of the two stored doubles $d$ and $h$, in every execution mode. The
supported range of spacings and origins, which keeps every intermediate a
normal number, is in {doc}`numerical-contract`.

On a test set of about 20,800 queries (20,000 uniform on $[-15,15]$ plus the
doubles at and within four ULPs of 81 cell edges, at $h=0.38$), naive
division under `jax.jit` chose a different floor from exact arithmetic 62
times, and the remainder rule never did, in eager code, under `jax.jit` and
under `jax.vmap`.

## A wrong floor costs a refusal, not accuracy

A floor that is off by one near a cell edge shifts the stencil by one node.
At an exact edge the node that leaves and the node that joins both have
weight exactly zero (a sinc evaluated at an integer), and a few ULPs away
their weights are near $10^{-20}$, so the image does not change measurably.

The harm is
elsewhere: every node of the stencil must exist, so near the edge of the
stored grid a shifted stencil can reach one node too far and turn an
accepted read into a refusal, in one execution mode but not another. A
library whose decisions depend on how it was compiled cannot give
reproducible receipts, which is why the floor rule is versioned in the
kernel profile identity (`exact_quotient_floor_v1`).
