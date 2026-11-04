# sinchole

Certified coronagraph field tables in JAX.

`sinchole` is a JAX/Equinox library for storing and reading a coronagraph's
complex focal-plane field. The field lives on band-limited lattices in source
position and detector pixel, with a Chebyshev basis in wavelength. Tables are
read with windowed-sinc kernels that carry error certificates, and a linked
stellar wavefront-error layer accompanies them. Tables export to the YIP
format and plug into image simulators and exposure-time calculators through an
adapter.

## Installation

```bash
pip install sinchole
```

Check the installed version:

```python
import sinchole

print(sinchole.__version__)
```

## License

MIT. See [LICENSE](LICENSE).
