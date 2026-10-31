# sinchole

Certified coronagraph field tables in JAX.

`sinchole` stores a coronagraph's complex focal-plane field on band-limited
lattices in source position and detector pixel, with a Chebyshev basis in
wavelength. Tables are read with windowed-sinc kernels that carry error
certificates, and a linked stellar wavefront-error layer accompanies them.
Tables export to the YIP format and plug into image simulators and
exposure-time calculators through an adapter. Built on
[JAX](https://github.com/google/jax) and
[Equinox](https://github.com/patrick-kidger/equinox).

```bash
pip install sinchole
```

```{toctree}
:maxdepth: 2
:caption: API Reference
:hidden:

autoapi/index
```
