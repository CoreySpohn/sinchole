"""Sampling-admitted intensity grids with unknown error bounds."""

try:
    from sinchole._version import __version__
except ImportError:  # pragma: no cover - source tree without a generated version file
    try:
        from importlib.metadata import PackageNotFoundError, version

        __version__ = version("sinchole")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"

from sinchole.admission import (
    MIRROR_TOLERANCE,
    index_reasons,
    judge_checks,
    lattice_reasons,
    offset_evidence,
    plan_spacing,
    sampling_ratio,
    sampling_reasons,
    select_profile,
    spectral_reasons,
    valid_intensity,
    validate_declarations,
)
from sinchole.domain import Domain, coverage, restrict_domain, validate_check_positions
from sinchole.evidence import (
    Admission,
    AdmissionRefused,
    BoundKind,
    LedgerEvidence,
    MemoryLimit,
    PlanRefused,
    PrecisionUnavailable,
    ReadRefused,
    Reason,
    Receipt,
    Status,
    reason_names,
)
from sinchole.geometry import source_grid
from sinchole.grid import (
    CheckSet,
    MirrorCheckSet,
    Normalization,
    Premises,
    Products,
    Reflection,
    SampledGrid,
    admit_grid,
)
from sinchole.kernel import KernelProfile, PrefilteredKernelProfile
from sinchole.planner import (
    PrefilterPlan,
    plan_cell_checks,
    plan_mirror_bases,
    plan_prefilter,
    plan_radial_checks,
    plan_sampling,
)
from sinchole.reader import (
    AbstractImageReader,
    ImageRead,
    Interpolator,
    MemoryBudget,
    PrefilteredReader,
    ReadPolicy,
    Tiling,
    check_memory,
    estimate_memory,
    make_reader,
    memory_budget,
    read_batch,
    require_read,
)
from sinchole.scene import SceneRenderer

__all__ = [
    "MIRROR_TOLERANCE",
    "AbstractImageReader",
    "Admission",
    "AdmissionRefused",
    "BoundKind",
    "CheckSet",
    "Domain",
    "ImageRead",
    "Interpolator",
    "KernelProfile",
    "LedgerEvidence",
    "MemoryBudget",
    "MemoryLimit",
    "MirrorCheckSet",
    "Normalization",
    "PlanRefused",
    "PrecisionUnavailable",
    "PrefilterPlan",
    "PrefilteredKernelProfile",
    "PrefilteredReader",
    "Premises",
    "Products",
    "ReadPolicy",
    "ReadRefused",
    "Reason",
    "Receipt",
    "Reflection",
    "SampledGrid",
    "SceneRenderer",
    "Status",
    "Tiling",
    "__version__",
    "admit_grid",
    "check_memory",
    "coverage",
    "estimate_memory",
    "index_reasons",
    "judge_checks",
    "lattice_reasons",
    "make_reader",
    "memory_budget",
    "offset_evidence",
    "plan_cell_checks",
    "plan_mirror_bases",
    "plan_prefilter",
    "plan_radial_checks",
    "plan_sampling",
    "plan_spacing",
    "read_batch",
    "reason_names",
    "require_read",
    "restrict_domain",
    "sampling_ratio",
    "sampling_reasons",
    "select_profile",
    "source_grid",
    "spectral_reasons",
    "valid_intensity",
    "validate_check_positions",
    "validate_declarations",
]
