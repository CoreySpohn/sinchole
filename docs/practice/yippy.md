# Reading deliveries through yippy

Most users never call sinchole directly. A coronagraph designer delivers a
directory of FITS files with a JSON manifest, and
[yippy](https://yippy.readthedocs.io) opens it as a `Coronagraph` whose
off-axis PSFs come from sinchole. The yippy package owns everything about files (parsing,
units, hashes, caches and the choice of backend), and sinchole owns the
grid validation, the reads and their refusals. This page shows the yippy side
of that division.

:::{note}
The sinchole backend for yippy is optional and lives behind yippy's `sinc`
extra (`pip install "yippy[sinc]"`). Enable JAX 64-bit mode before
constructing a `Coronagraph` with sinc reads.
:::

## Choosing the interpolation backend

`Coronagraph` takes an `interpolation` argument with three values.

`"blend"` (the default)
: The existing yippy interpolation of tabulated PSFs. Legacy YIP packages,
  their caches and their results are unchanged bit for bit.

`"sinc"`
: Strict sinchole reads. The constructor raises if the delivery is not a
  versioned sinc delivery, if the sinchole extra is missing, if JAX 64-bit
  mode is off, or if admission refuses the grid.

`"auto"`
: The sinchole reader serves reads when the delivery is a versioned sinc delivery and admission
  succeeds, otherwise the blend backend with the reason recorded in
  `coro.interpolation_info`. Data errors (a corrupted hash, a header that
  disagrees with the manifest, an unknown unit) always raise, even in auto
  mode, because they mean the files are wrong rather than that sinc is
  unavailable.

```python
import jax

jax.config.update("jax_enable_x64", True)

from yippy import Coronagraph

coro = Coronagraph("path/to/delivery", interpolation="sinc")
image = coro.create_psf(4.0, 5.5)            # planet at (4.0, 5.5) lambda0/D
image, info = coro.create_psf_with_info(4.0, 5.5)
print(coro.interpolation_info.effective)     # "sinc"
print(coro.interpolation_info.receipt_id)    # the admission receipt identity
```

A quadrant delivery has no blend representation, so `"auto"` on a quadrant
delivery either reads with sinchole or raises, and it never falls back.

## Options

`SincOptions` selects a product and passes construction controls through to
sinchole.

```python
from yippy.interpolation import SincOptions
import sinchole as sc
import jax.numpy as jnp

options = SincOptions(
    product_id="mono-000",                     # required for multi-plane deliveries
    domain=sc.Domain(                          # optional narrower permission region
        "box", "closed", jnp.array([[-4.0, -4.0], [4.0, 4.0]]), jnp.zeros(2)
    ),
)
coro = Coronagraph("path/to/delivery", interpolation="sinc", sinc_options=options)
```

A caller domain narrower than the delivery's served region excludes the
delivered independent checks that fall outside it, and the receipt lists them
as excluded. Checks outside the delivery's own served region are a data error.
A bounded `ReadPolicy` never falls back to blend: version 0.1 cannot issue
bounds, so a bounded request refuses.

## What yippy checks before routing

Every declaration and integrity check runs before yippy decides between sinc
and blend, so a damaged delivery cannot slip through a fallback.

The checks
cover the SHA-256 hash of every referenced file (images, offsets, coordinate
lists, independent checks, mirror checks and reports), agreement between FITS
headers and the manifest, the order of domain bounds, consistency of layout
and reflection, a complete rectangle and authoritative integer indices.
Missing physical metadata, such as the coordinate model or a premise, is
classified as missing (auto falls back with that reason, strict sinc raises),
while metadata that defines units or geometry (detector shape, center and
pitch, sample kind, normalization and photon conversion) is always a data
error.

## Units and frames

Off-axis images keep the delivered units, photon fraction per pixel for a
pixel-integral delivery, and sinchole applies no normalization of its own.
Stellar and sky products keep their own frame and pixel scale, which may
differ from the off-axis frame. The `PIXSCALE` header is in reference
lambda/D per pixel. An empty header comment means that schema unit, a named
angular unit (arcsec, mas, deg, rad) is converted, and an unknown unit raises.

## The delivery manifest at a glance

A sinc delivery is a directory whose `sinc-delivery.json` manifest (schema
`sinc-delivery-0.1`) names every file and its hash. The designer-facing
checklist is on {doc}`designers`.

```text
sinc-delivery.json                   manifest: frame, layout, symmetry, hashes
starter_training.csv                 indexed positive-quadrant positions
starter_checks.csv                   independent interior check positions
starter_mirror_checks.csv            independent negative-position checks
mono-000/offax_psf.fits              training images [source, row, column]
mono-000/offax_psf_offset_list.fits  offsets [source, 2], x then y
checks/mono-000.fits                 independent check images
mirror-checks/mono-000.fits          mirror check images
mirror-report.json                   negative-position and axis comparisons
legacy-yip/                          stellar and sky products, unchanged
```
