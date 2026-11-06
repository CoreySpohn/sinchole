---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  render_markdown_format: myst
---

# What does a read cost?

A coronagraph yield code asks for a planet image at every orbit step of every
trial, so the cost of one read decides whether reading between grid nodes is
affordable. This page reports what version 0.1 costs for the physicaloptix
starter delivery on one laptop CPU, what a batch of $10^4$ positions costs,
and how that compares with yippy's current blend interpolation on the same
positions. Every number comes from `docs/scripts/benchmark.py`, which writes
`_static/validation/benchmark.json` beside the validation figures. Rerun it on
your own machine before relying on a number.

```{code-cell} python
:tags: [remove-cell]
import sys

sys.path.insert(0, "../scripts")
from IPython.display import Markdown, display
from validation_tables import load, sci

b = load("benchmark.json")
m = b["machine"]
f32, f64, yp = b["float32"], b["float64"], b["yippy_blend"]
ms = lambda s: f"{1e3 * s:.1f} ms"
per = lambda r: f"{1e3 * r['seconds_per_read']:.1f} ms"
bs32_first = f32["batch_small_first_call_seconds"]
```

## Where the numbers were taken

```{code-cell} python
:tags: [remove-input]
display(Markdown(
    f"- Machine: {m['chip']}, {m['total_number_of_cores']} cores, {m['memory']}.\n"
    f"- Software: Python {m['python']}, JAX {m['jax']} on the {m['backend']} backend.\n"
    f"- Delivery: 837 float32 nodes of 128 by 128 pixels, {f32['resident_mib']:.0f} MiB resident "
    f"({f64['resident_mib']:.0f} MiB if stored as float64).\n"
    f"- Load: the machine was shared with other jobs (load average "
    f"{m['load_average_at_start'][0]:.1f} at the start and {b['load_average_after_reads'][0]:.1f} "
    f"after the reads), so treat the timings as upper bounds on an idle machine. The float32 and "
    f"float64 readers were timed in alternating rounds, and float64 came out faster here, "
    f"which is within what contention and memory traffic can do and is not a claim about the kernel.\n"
    f"- GPU: untested. No GPU number exists for version 0.1."
))
```

## Cost of one read and of a batch

```{code-cell} python
:tags: [remove-input]
bs32, bs64 = f32["batch_small"], f64["batch_small"]
rows = [
    ("Admission, first call in the process", f"{b['admission_first_call_seconds']:.1f} s", "includes JAX start-up work"),
    ("Admission, later calls", f"{b['admission_later_calls_median_seconds']:.1f} s",
     f"minimum {b['admission_later_calls_min_seconds']:.1f} s over 4 calls"),
    ("Compile plus first single read, float32", f"{f32['single_first_call_seconds']:.2f} s", ""),
    ("Steady single read, float32", ms(f32["single_steady_median_seconds"]),
     f"range {ms(f32['single_steady_min_seconds'])} to {ms(f32['single_steady_max_seconds'])}"),
    ("Steady single read, float64 storage", ms(f64["single_steady_median_seconds"]),
     f"range {ms(f64['single_steady_min_seconds'])} to {ms(f64['single_steady_max_seconds'])}"),
    ("`read_batch`, 200 positions, float32", ms(bs32["seconds_per_read_median"]) + " per read",
     f"minimum {ms(bs32['seconds_per_read_min'])}, first call {bs32_first:.1f} s"),
    ("`read_batch`, 200 positions, float64 storage", ms(bs64["seconds_per_read_median"]) + " per read",
     f"minimum {ms(bs64['seconds_per_read_min'])}"),
    ("`read_batch`, 10 000 positions, float32", per(f32["batch_10000"]) + " per read",
     f"{f32['batch_10000']['seconds']:.0f} s total, one run, compile included"),
    ("`read_batch`, 10 000 positions, float64 storage", per(f64["batch_10000"]) + " per read",
     f"{f64['batch_10000']['seconds']:.0f} s total, one run, compile included"),
    ("yippy blend, single read", ms(yp["single_steady_median_seconds"]), ""),
    (f"yippy blend, {yp['batch']['positions']} positions in chunks of {yp['batch']['chunk']}",
     per(yp["batch"]) + " per read", f"{yp['batch']['seconds']:.0f} s total"),
]
display(Markdown("| Quantity | Result | Note |\n|---|---|---|\n" + "\n".join(f"| {a} | {r} | {n} |" for a, r, n in rows)))
```

```{code-cell} python
:tags: [remove-input]
ratio = yp["batch"]["seconds_per_read"] / f32["batch_10000"]["seconds_per_read"]
ratio64 = yp["batch"]["seconds_per_read"] / f64["batch_10000"]["seconds_per_read"]
display(Markdown(
    f"Over $10^4$ positions the sinc read took {per(f32['batch_10000'])} per read with float32 "
    f"storage and {per(f64['batch_10000'])} with float64 storage, and the yippy blend took "
    f"{per(yp['batch'])} per read. The blend is faster than the sinc read by a factor of "
    f"{1 / ratio:.0f} (float32) or {1 / ratio64:.0f} (float64). The blend reads four 128 by 128 "
    f"images and shifts each by an FFT phase ramp, about $10^{7}$ operations per read "
    f"(four padded forward and inverse transforms) against $1.7\\times10^{7}$ for the sinc "
    f"stencil. The two estimates are of the same order, so the blend's speed comes from the "
    f"efficiency of FFTs and not from fewer operations."
))
```

## Memory

```{code-cell} python
:tags: [remove-input]
display(Markdown(
    f"- Peak resident memory of the process after loading the delivery: {b['after_load_rss_mib']:.0f} MiB.\n"
    f"- After admission: {b['admission_peak_rss_mib']:.0f} MiB.\n"
    f"- After the float32 reads, including the $10^4$ batch: {f32['batch_10000']['peak_rss_mib_after']:.0f} MiB.\n"
    f"- After the float64 reads: {f64['batch_10000']['peak_rss_mib_after']:.0f} MiB.\n"
    f"- A batch of $10^4$ returns {f32['batch_10000']['output_gib']:.2f} GiB of float64 images, "
    f"so it needs a memory budget above the default one gibibyte of output, "
    f"so the benchmark sets two. Chunk larger campaigns."
))
```

The peak figures are the process's peak resident set size after each stage, so
they only rise. The resident delivery itself is the first row of the table
above, and the output of a batch dominates everything else.

## What to take from this

Reads cost tens of milliseconds on a loaded CPU because each pixel sums 1024
products, so one read is about $1.6\times10^4\times1024\approx1.7\times10^{7}$
multiply-adds. At a few $10^{9}$ operations per second that is the order of
10 ms, and the measured 20 to 50 ms on a busy machine is within a few times
of that estimate, so the cost is mostly the arithmetic. Batches of positions are the unit to optimize, and a GPU should help
most there, but no GPU measurement exists (GPU untested).

Next: the {doc}`../validation` page shows what those reads are worth.
