"""sinchole -- certified coronagraph field tables in JAX.

A coronagraph's complex focal-plane field is stored on band-limited lattices
in source position and detector pixel, with a Chebyshev basis in wavelength.
The tables are read with windowed-sinc kernels that carry error certificates,
and a linked stellar wavefront-error layer rides alongside. Tables export to
the YIP format and plug into image simulators and exposure-time calculators
through an adapter.
"""

try:
    from sinchole._version import __version__
except ImportError:  # pragma: no cover - source tree without a generated version file
    try:
        from importlib.metadata import PackageNotFoundError, version

        __version__ = version("sinchole")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
]
