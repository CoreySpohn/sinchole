# sinchole

<p align="center">
  <a href="https://github.com/CoreySpohn/sinchole/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/CoreySpohn/sinchole/ci.yml?branch=main&style=flat-square&logo=github&label=tests" alt="Tests"/></a>
  <a href="https://github.com/CoreySpohn/sinchole/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" alt="License"/></a>
  <a href="https://github.com/CoreySpohn/sinchole/blob/main/pyproject.toml"><img src="https://img.shields.io/badge/python-3.12%2B-blue.svg?style=flat-square&logo=python" alt="Python 3.12+"/></a>
</p>

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
