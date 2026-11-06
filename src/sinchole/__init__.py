"""Sampling-admitted intensity grids with unknown error bounds."""

try:
    from sinchole._version import __version__
except ImportError:  # pragma: no cover - source tree without a generated version file
    try:
        from importlib.metadata import PackageNotFoundError, version

        __version__ = version("sinchole")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"

from sinchole.domain import Domain
from sinchole.evidence import (
    Admission,
    AdmissionRefused,
    BoundKind,
    LedgerEvidence,
    MemoryLimit,
    PrecisionUnavailable,
    ReadRefused,
    Reason,
    Receipt,
    Status,
    reason_names,
)
from sinchole.grid import (
    CheckSet,
    Frame,
    Lattice,
    MirrorCheckSet,
    Normalization,
    PixelGrid,
    Premises,
    Products,
    Reflection,
    SampledGrid,
    admit_grid,
)
from sinchole.kernel import KernelProfile
from sinchole.reader import (
    AbstractImageReader,
    ImageRead,
    Interpolator,
    MemoryBudget,
    ReadPolicy,
    Tiling,
    read_batch,
    require_read,
)

__all__ = [
    "AbstractImageReader",
    "Admission",
    "AdmissionRefused",
    "BoundKind",
    "CheckSet",
    "Domain",
    "Frame",
    "ImageRead",
    "Interpolator",
    "KernelProfile",
    "Lattice",
    "LedgerEvidence",
    "MemoryBudget",
    "MemoryLimit",
    "MirrorCheckSet",
    "Normalization",
    "PixelGrid",
    "PrecisionUnavailable",
    "Premises",
    "Products",
    "ReadPolicy",
    "ReadRefused",
    "Reason",
    "Receipt",
    "Reflection",
    "SampledGrid",
    "Status",
    "Tiling",
    "__version__",
    "admit_grid",
    "read_batch",
    "reason_names",
    "require_read",
]
